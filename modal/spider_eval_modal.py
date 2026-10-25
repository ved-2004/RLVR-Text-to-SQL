"""
Spider evaluation on Modal.

Usage (in order):
    modal run spider_eval_modal.py::smoke          # ~2 min, verifies image + vLLM
    modal run spider_eval_modal.py::setup_data     # one time, unzips data into volume
    modal run spider_eval_modal.py::evaluate       # the real 7B baseline run

Data prep (once, from your laptop):
    modal volume create spider-data
    modal volume put spider-data spider_data.zip /spider_data.zip
    modal volume put spider-data test_suite_database.zip /test_suite_database.zip
"""

import modal

app = modal.App("spider-eval")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

# Everything the container needs, declared once. This is the whole point of Modal:
# if it works today it works identically in six months.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install(
        "vllm",              # unpinned: matches Modal's CUDA 13 driver
        "transformers",
        "numpy",             # <- the thing missing from hello_gpu.py
        "nltk",
        "datasets",
        "sqlparse", 
    )
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"}) 
    .run_commands(
        "git clone --depth 1 https://github.com/taoyds/spider /opt/spider",
        "git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval /opt/test-suite",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"",
    )
)

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA = "/data"
SPIDER = f"{DATA}/spider_data"
DB_DIR = f"{SPIDER}/database"
TABLES = f"{SPIDER}/tables.json"
DEV_JSON = f"{SPIDER}/dev.json"
TS_DB = f"{DATA}/ts_db/database"

PROMPT = (
    "Database schema:\n{schema}\n\nWrite a single SQLite query that answers "
    "the question. Return ONLY the query inside a ```sql code block.\n\nQuestion: {question}"
)


# ---------------------------------------------------------------- verifier
# Identical logic to the 1.5B notebooks, so numbers stay comparable.

FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


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


# ---------------------------------------------------------------- evaluator output
# Copied from reeval_official.py. Read metrics by section and row label, never by
# position: the official evaluator prints every exact-match failure before the table.

SECTIONS = {"EXECUTION ACCURACY": "exec", "EXACT MATCHING ACCURACY": "exact",
            "PARTIAL MATCHING ACCURACY": "partial_acc",
            "PARTIAL MATCHING RECALL": "partial_recall",
            "PARTIAL MATCHING F1": "partial_f1"}
LEVELS = ["easy", "medium", "hard", "extra", "all"]


def parse_table(out):
    """{(section, row_label): {level: value}} from evaluator stdout. Only lines
    after the 'count' header are read, so printed per-example errors are ignored."""
    import re
    res, section, started = {}, None, False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("count") and len(s.split()) == 6:
            started = True
            continue
        if not started:
            continue
        for key, name in SECTIONS.items():
            if key in s:
                section = name
                break
        else:
            m = re.match(r"^(.*?)\s+((?:-?\d+\.\d+\s*){5})$", s)
            if m and section:
                vals = [float(v) for v in m.group(2).split()]
                res[(section, m.group(1).strip())] = dict(zip(LEVELS, vals))
    return res


def get(t, section, label, level="all"):
    return t.get((section, label), {}).get(level)


# ---------------------------------------------------------------- one-time data setup
@app.function(image=image, volumes={DATA: vol}, timeout=1800)
def setup_data():
    """Unzip the uploaded archives into the volume. Run once."""
    import subprocess, os
    subprocess.run(f"unzip -q -o {DATA}/spider_data.zip -d {DATA}/", shell=True)
    os.makedirs(f"{DATA}/ts_db", exist_ok=True)
    subprocess.run(
        f"unzip -q -o {DATA}/test_suite_database.zip -d {DATA}/ts_db", shell=True)
    vol.commit()
    print("spider_data:", os.listdir(SPIDER)[:10])
    print("ts_db:", os.listdir(f"{DATA}/ts_db")[:10])
    # gate: gold must score 1.0 against itself, or paths are wrong
    import json
    ex = json.load(open(DEV_JSON))[0]
    dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
    print("VERIFIER GATE:", exec_match(dbp, ex["query"], ex["query"]))


# ---------------------------------------------------------------- cheap image check
@app.function(image=image, gpu="A100", timeout=900)
def smoke():
    """Prove vLLM loads and generates before spending on the full run."""
    from vllm import LLM, SamplingParams
    llm = LLM(model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
              dtype="bfloat16", max_model_len=2048, enforce_eager=True)
    out = llm.generate(["Write a SQL query selecting all rows from t."],
                       SamplingParams(temperature=0.0, max_tokens=32))
    print("SMOKE OK:", out[0].outputs[0].text.strip()[:120])


# ---------------------------------------------------------------- the real run
@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, timeout=7200)
def evaluate(model_id: str = MODEL_ID, limit: int = 0, k: int = 8):
    import json, os, subprocess
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_id)
    dev = json.load(open(DEV_JSON))
    if limit:
        dev = dev[:limit]

    examples = []
    for ex in dev:
        dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
        msg = [{"role": "user", "content": PROMPT.format(
            schema=build_schema_string(dbp), question=ex["question"])}]
        examples.append({
            "text": tok.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True),
            "gold": ex["query"], "db_id": ex["db_id"], "db_path": dbp,
        })
    print(f"{len(examples)} dev examples")

    llm = LLM(model=model_id, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096)
    prompts = [e["text"] for e in examples]

    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
    preds = [extract_sql(o.outputs[0].text) or "SELECT 1" for o in greedy]

    sampled = llm.generate(prompts, SamplingParams(
        temperature=0.8, top_p=0.95, n=k, max_tokens=256))

    tag = model_id.split("/")[-1]
    gold_p, pred_p = f"{DATA}/gold.txt", f"{DATA}/pred_{tag}.txt"
    with open(gold_p, "w") as gf, open(pred_p, "w") as pf:
        for e, p in zip(examples, preds):
            gf.write(f"{e['gold']}\t{e['db_id']}\n")
            pf.write(f"{p}\n")

    def run_eval(cwd, db, etype):
        r = subprocess.run(
            ["python", "evaluation.py", "--gold", gold_p, "--pred", pred_p,
             "--db", db, "--table", TABLES, "--etype", etype],
            cwd=cwd, capture_output=True, text=True)
        return r.stdout

    print("\n===== EX (official spider evaluator) =====")
    ex_out = run_eval("/opt/spider", DB_DIR, "all")
    print(ex_out[-2500:])

    print("\n===== test-suite evaluator, SINGLE db =====")
    ts_single = run_eval("/opt/test-suite", DB_DIR, "exec")
    print(ts_single[-900:])

    print("\n===== test-suite evaluator, TEST-SUITE dbs =====")
    ts_suite = run_eval("/opt/test-suite", TS_DB, "exec")
    print(ts_suite[-900:])

    p1 = sum(exec_match(e["db_path"], preds[i], e["gold"])
             for i, e in enumerate(examples)) / len(examples)
    pk = sum(
        1 for i, e in enumerate(examples)
        if any(exec_match(e["db_path"], extract_sql(c.text), e["gold"]) >= 1
               for c in sampled[i].outputs)
    ) / len(examples)
    print(f"\npass@1 (loose, full dev): {p1:.3f}   pass@{k}: {pk:.3f}")

    T_ex, T_single, T_suite = parse_table(ex_out), parse_table(ts_single), parse_table(ts_suite)
    metrics = {
        "EX_official_spider": get(T_ex, "exec", "execution"),
        "EM_official_spider": get(T_ex, "exact", "exact match"),
        "IUEN_partial_acc": get(T_ex, "partial_acc", "IUEN"),
        "EX_testsuite_single": get(T_single, "exec", "execution"),
        "TS_suite": get(T_suite, "exec", "execution"),
    }
    print("parsed:", metrics)

    # Full evaluator stdout, never a tail: a [-2500:] slice cut off the execution row.
    with open(f"{DATA}/summary_{tag}.json", "w") as f:
        json.dump({"model": model_id, "n_dev": len(examples),
                   "pass@1_loose": p1, f"pass@{k}_loose": pk, **metrics,
                   "ex_raw": ex_out, "ts_single_raw": ts_single,
                   "ts_suite_raw": ts_suite}, f, indent=2)
    vol.commit()
    print(f"\nsaved: summary_{tag}.json")


@app.local_entrypoint()
def main():
    evaluate.remote()

@app.function(image=image, volumes={DATA: vol}, timeout=600)
def check_ts():
    import os, subprocess
    for p in [f"{DATA}/ts_db", f"{DATA}/ts_db/database"]:
        print(p, "->", os.listdir(p)[:8] if os.path.exists(p) else "MISSING")
    r = subprocess.run(
        ["python", "evaluation.py", "--gold", f"{DATA}/gold.txt",
         "--pred", f"{DATA}/pred_Qwen2.5-Coder-7B-Instruct.txt",
         "--db", TS_DB, "--table", TABLES, "--etype", "exec"],
        cwd="/opt/test-suite", capture_output=True, text=True)
    print("STDOUT:", r.stdout[-800:])
    print("STDERR:", r.stderr[-1500:])   # the real error lives here