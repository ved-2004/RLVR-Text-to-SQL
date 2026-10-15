"""
GRPO post-training for text-to-SQL on Modal.

One-time setup:
    modal secret create wandb WANDB_API_KEY=<your key from wandb.ai/authorize>

Usage (in order):
    modal run spider_grpo_modal.py::smoke_train           # ~10 min, 20 examples, 5 steps
    modal run --detach spider_grpo_modal.py::train        # the real run

Then evaluate the adapter with spider_eval_modal.py (see notes at bottom).
"""

import modal

app = modal.App("spider-grpo")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install(
        "torch",
        "trl",
        "peft",
        "transformers",
        "accelerate",
        "datasets",
        "numpy",
        "sqlparse",
        "wandb",
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA = "/data"
SPIDER = f"{DATA}/spider_data"
DB_DIR = f"{SPIDER}/database"
TRAIN_JSON = f"{SPIDER}/train_spider.json"

PROMPT = (
    "Database schema:\n{schema}\n\nWrite a single SQLite query that answers "
    "the question. Return ONLY the query inside a ```sql code block.\n\nQuestion: {question}"
)

FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


# ---------------------------------------------------------------- verifier
# Byte-identical to the eval harness. If these ever diverge, the reward and
# the metric stop measuring the same thing.

def build_schema_string(db_path, max_tables=20):
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()][:max_tables]
    out = []
    for t in tables:
        cur.execute(f'PRAGMA table_info("{t}")')
        out.append(f"{t}({', '.join(r[1] for r in cur.fetchall())})")
    con.close()
    return "\n".join(out)


def extract_sql(t):
    import re
    if not isinstance(t, str):
        t = t[0]["content"] if t else ""
    m = (re.search(r"```sql\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
         or re.search(r"```\s*(.*?)```", t, re.DOTALL))
    return re.sub(r"\s+", " ", (m.group(1) if m else t).strip().rstrip(";").strip())


def safe_execute(db_path, sql, timeout=5.0):
    import sqlite3, time
    if not sql or any(k in sql.lower() for k in FORBIDDEN):
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    start = time.time()
    con.set_progress_handler(lambda: (time.time() - start) > timeout, 10000)
    try:
        return con.cursor().execute(sql).fetchall()
    except Exception:
        return None
    finally:
        con.close()


def exec_match(db_path, pred, gold):
    p = safe_execute(db_path, pred)
    if p is None:
        return 0.0
    g = safe_execute(db_path, gold)
    if g is None:
        return 0.0
    return 1.0 if set(map(tuple, p)) == set(map(tuple, g)) else 0.0


def sql_reward(completions, gold_sql, db_path, **kwargs):
    """Binary execution reward. Same control condition as the 1.5B runs."""
    return [exec_match(dbp, extract_sql(c), g)
            for c, g, dbp in zip(completions, gold_sql, db_path)]


def build_dataset(n_train):
    import json, os
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    raw = json.load(open(TRAIN_JSON))
    rows = []
    for ex in raw:
        dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
        if not os.path.exists(dbp):
            continue
        msg = [{"role": "user", "content": PROMPT.format(
            schema=build_schema_string(dbp), question=ex["question"])}]
        rows.append({"prompt": msg, "gold_sql": ex["query"], "db_path": dbp})
        if n_train and len(rows) >= n_train:
            break
    # gate: gold must score 1.0 against itself
    assert exec_match(rows[0]["db_path"], rows[0]["gold_sql"],
                      rows[0]["gold_sql"]) == 1.0, "VERIFIER BROKEN"
    print(f"{len(rows)} train examples, verifier gate OK")
    return Dataset.from_list(rows)


def run_training(n_train, max_steps, run_name, num_generations=8, save_steps=100):
    import os, json, glob
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    from transformers import TrainerCallback

    class CommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            vol.commit()

    os.environ["WANDB_PROJECT"] = "rlvr-text2sql"

    dataset = build_dataset(n_train)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg_kwargs = dict(
        output_dir=f"{DATA}/ckpt/{run_name}",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=num_generations,
        max_completion_length=256,
        learning_rate=1e-5,
        beta=0.04,
        temperature=0.9,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
    )
    if max_steps:
        cfg_kwargs["max_steps"] = max_steps
    else:
        cfg_kwargs["num_train_epochs"] = 1

    trainer = GRPOTrainer(
        model=MODEL_ID,
        args=GRPOConfig(**cfg_kwargs),
        train_dataset=dataset,
        reward_funcs=sql_reward,
        peft_config=peft_config,
        callbacks=[CommitCallback()],
    )

    ckpt_dir = f"{DATA}/ckpt/{run_name}"
    resume = bool(glob.glob(f"{ckpt_dir}/checkpoint-*"))
    print("resuming from checkpoint" if resume else "starting fresh")
    trainer.train(resume_from_checkpoint=resume)

    # SAVE THE ADAPTER FIRST — ~50MB, and it plus the base model is the
    # full reproduction path. Never merge before saving this.
    adapter_dir = f"{DATA}/adapters/{run_name}"
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    with open(f"{adapter_dir}/trainlog.json", "w") as f:
        json.dump(trainer.state.log_history, f)
    vol.commit()
    print(f"adapter saved to {adapter_dir}")
    return adapter_dir


# ---------------------------------------------------------------- entrypoints
@app.function(image=image, gpu="A100-80GB", volumes={DATA: vol},
              secrets=[modal.Secret.from_name("wandb")], timeout=3600)
def smoke_train():
    """20 examples, 5 steps. Proves the loop + reward + wandb before the real run."""
    run_training(n_train=20, max_steps=10, run_name="7b-grpo-smoke",
                 num_generations=4, save_steps=3)


@app.function(image=image, gpu="A100-80GB", volumes={DATA: vol},
              secrets=[modal.Secret.from_name("wandb")], timeout=28800)
def train(n_train: int = 7000, model_id: str = MODEL_ID, run_name: str = "7b-grpo-binary-7k"):
    global MODEL_ID
    MODEL_ID = model_id
    run_training(n_train=n_train, max_steps=0, run_name=run_name)


@app.local_entrypoint()
def main():
    train.remote()


# ----------------------------------------------------------------------
# To evaluate afterwards: in spider_eval_modal.py, load the base model with
# the adapter applied, e.g.
#
#   from peft import PeftModel
#   ... or pass the merged path to vLLM after merging on a CPU-only function.
#
# Simplest path: add a `merge` function that loads base + adapter, merges,
# writes to /data/merged/<run_name>, then point evaluate() at that path.

@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, timeout=1800)
def inspect(n_examples: int = 6, n_samples: int = 4):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    ds = build_dataset(n_examples)

    print("=" * 70)
    print("FULL PROMPT [0] — check the schema is actually here:")
    print(tok.apply_chat_template(ds[0]["prompt"], tokenize=False,
                                  add_generation_prompt=True)[:2000])
    print("=" * 70)

    for i in range(len(ds)):
        row = ds[i]
        text = tok.apply_chat_template(row["prompt"], tokenize=False,
                                       add_generation_prompt=True)
        enc = tok([text] * n_samples, return_tensors="pt",
                  padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=True,
                                 temperature=0.9, top_p=0.95,
                                 pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        print("\n" + "#" * 70)
        print(f"EXAMPLE {i}")
        print("GOLD SQL :", row["gold_sql"])
        print("GOLD ROWS:", str(safe_execute(row["db_path"], row["gold_sql"]))[:200])
        for j, t in enumerate(texts):
            sql = extract_sql(t)
            r = exec_match(row["db_path"], sql, row["gold_sql"])
            ntok = len(tok(t)["input_ids"])
            print(f"  --- sample {j} | tokens={ntok} | reward={r}")
            print(f"      RAW : {repr(t[:250])}")
            print(f"      SQL : {sql[:200]}")
            print(f"      ROWS: {str(safe_execute(row['db_path'], sql))[:200]}")

@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, timeout=3600)
def merge(model_id: str = MODEL_ID, run_name: str = "7b-grpo-binary-7k"):
    import torch, os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter = f"{DATA}/adapters/{run_name}"
    out = f"{DATA}/merged/{run_name}"
    print("adapter contents:", os.listdir(adapter))

    global MODEL_ID
    MODEL_ID = model_id

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()

    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(out)
    vol.commit()
    print("merged model at", out, "->", os.listdir(out)[:6])