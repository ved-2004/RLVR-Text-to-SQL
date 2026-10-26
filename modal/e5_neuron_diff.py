"""
E5: Which of GRPO's weight changes carry the 1.5B gain?

Background. E4 steered neurons by adding vectors to their activations. The doses
were too large, most runs produced no valid SQL, and the random controls told us
nothing. E5 intervenes with GRPO's own change instead. GRPO is a LoRA (r=16) on
every linear layer, so its exact change to each weight is
    dW = (lora_alpha / r) * B @ A
read in fp32 from /data/adapters/<run>. For MLP neuron n, GRPO changed row n of
gate_proj and up_proj and column n of down_proj, and nothing else about it.

Every E5 model is base + (part of dW). "Copy in" applies dW only to a chosen set
of layers, modules or neurons. "Remove" applies all of dW except that set, which
is GRPO with that part undone. Each model lies between base and GRPO, so nothing
is pushed off-distribution, and each one is scored with the main eval suite.

Step 1  diff   (one A100-40GB)
    a. Weight score per MLP neuron, relative to the base weights:
           s_n = |dW_gate[n,:]| / |W_gate[n,:]| + |dW_up[n,:]| / |W_up[n,:]|
               + |dW_down[:,n]| / |W_down[:,n]|
       plus relative |dW| per layer and module, and per attention head.
    b. Activation score per MLP neuron: teacher-force both models on the same
       text (base and GRPO dev predictions) and take
           mean |a_GRPO - a_base| / std(a_base)   at the input of down_proj.
    c. Cross-check with E4: ranks of the E4 neurons, top-50 overlap with E4's
       AtP list, and Spearman correlation between the three rankings.
    d. Writes the configs, with explicit neuron lists, for step 2.

Step 2  sweep  (one A100-40GB per config, in parallel)
    Builds the config's model in the container's /tmp (never on the volume),
    then runs the main eval: greedy for EX / TS / official EX / set-op, and 8
    samples at temperature 0.8 for pass@1 / pass@8, plus style statistics.

Configs
    base, full                               sanity: must match the main eval
    attn_only, mlp_only                      dW on one module type, all layers
    L00-06, L07-13, L14-20, L21-27           dW on one layer group, both types
    top256, top2048, top16384                dW on the top-k MLP neurons by s_n
    rand2048, rand16384                      random neurons, same count per layer
    grpo_minus_top2048, grpo_minus_rand2048  GRPO with those neurons undone

Commands
    modal run --detach e5_neuron_diff.py::pipeline
    modal run --detach e5_neuron_diff.py::pipeline --n-dev 50   # smoke run
    modal run e5_neuron_diff.py::summarize                      # prints the table again

Outputs on the volume
    /data/e5/diff_1.5b.json, /data/e5/scores_1.5b.npz, /data/e5/configs_1.5b.json
    /data/e5/runs/<config>/{pred.txt,gold.txt,*_eval.txt,row.json}
    /data/e5/summary_1.5b.json
"""

import modal

app = modal.App("spider-e5")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

# Same image as E3: unpinned vLLM and transformers, which read the merged
# checkpoint's rope_parameters correctly.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip")
    .pip_install("vllm", "transformers", "numpy", "nltk", "datasets", "sqlparse")
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .run_commands(
        "git clone --depth 1 https://github.com/taoyds/spider /opt/spider",
        "git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval /opt/test-suite",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"",
    )
)

DATA = "/data"
SPIDER = f"{DATA}/spider_data"
DB_DIR = f"{SPIDER}/database"
TABLES = f"{SPIDER}/tables.json"
DEV_JSON = f"{SPIDER}/dev.json"
TS_DB = f"{DATA}/ts_db/database"

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RL = "/data/merged/1.5b-grpo-binary-7k"
ADAPTER = "/data/adapters/1.5b-grpo-binary-7k"
PRED = {"base": f"{DATA}/pred_Qwen2.5-Coder-1.5B-Instruct.txt",
        "rl": f"{DATA}/pred_1.5b-grpo-binary-7k.txt"}
# Reference greedy numbers for the gate: the rescore of the main-eval predictions.
REEVAL = f"{DATA}/reeval/summary.json"
REEVAL_TAG = {"base": "Qwen2.5-Coder-1.5B-Instruct", "full": "1.5b-grpo-binary-7k"}
DEVICE = "cuda"

N_LAYERS, D_FF, N_HEADS, HEAD_DIM = 28, 8960, 12, 128
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")
E4_NEURONS = [(27, 3289), (26, 2897), (19, 2499)]   # E4: two T-writers, top AtP neuron
TOP_K = (256, 2048, 16384)
LAYER_GROUPS = [(0, 6), (7, 13), (14, 20), (21, 27)]

# Byte-identical to spider_eval_modal.PROMPT.
PROMPT = (
    "Database schema:\n{schema}\n\nWrite a single SQLite query that answers "
    "the question. Return ONLY the query inside a ```sql code block.\n\nQuestion: {question}"
)
FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


# ====================================================================== SQL helpers
# Copied from spider_eval_modal.py, so every number is scored the same way.
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
    g = safe_execute(db_path, gold) if p is not None else None
    return p is not None and g is not None and set(map(tuple, p)) == set(map(tuple, g))


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


def run_eval(cwd, gold_p, pred_p, db, etype):
    import subprocess
    r = subprocess.run(
        ["python", "evaluation.py", "--gold", gold_p, "--pred", pred_p,
         "--db", db, "--table", TABLES, "--etype", etype],
        cwd=cwd, capture_output=True, text=True)
    return r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")


def style_of(sql):
    """Same regexes as E2/E3/E4, so numbers are comparable."""
    import re
    low = sql.lower()
    return {"alias": len(re.findall(r"\bt\d+\b", low)),
            "qual": len(re.findall(r"\b[a-z_]+\.[a-z_]+", low)),
            "join": len(re.findall(r"\bjoin\b", low))}


def make_prompt(tok, ex, cache):
    dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
    if dbp not in cache:
        cache[dbp] = build_schema_string(dbp)
    msg = [{"role": "user", "content": PROMPT.format(schema=cache[dbp], question=ex["question"])}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True), dbp


# ====================================================================== LoRA delta
def module_of(key):
    """'model.layers.3.mlp.up_proj.weight' -> (3, 'up_proj')."""
    import re
    m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.weight$", key)
    return int(m.group(1)), m.group(2)


def load_lora(adapter_dir):
    """{base checkpoint key: (B, A, scale)}, fp32, so dW = scale * B @ A exactly."""
    import json
    from safetensors.torch import load_file
    cfg = json.load(open(f"{adapter_dir}/adapter_config.json"))
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "plain LoRA only"
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(f"{adapter_dir}/adapter_model.safetensors")
    out = {}
    for k, a in sd.items():
        if k.endswith(".lora_A.weight"):
            key = k[len("base_model.model."):-len(".lora_A.weight")] + ".weight"
            out[key] = (sd[k.replace(".lora_A.", ".lora_B.")].float(), a.float(), scale)
    return out


def load_base_state(model_id=BASE):
    """The base checkpoint's tensors, as stored (bf16), keyed as in model.safetensors."""
    import glob
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    d = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    sd = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        sd.update(load_file(f))
    return sd


def row_norms(B, A, scale):
    """|(scale * B @ A)[n, :]| for every row n, without forming the matrix."""
    return scale * ((B @ (A @ A.T)) * B).sum(1).clamp_min(0).sqrt()


def col_norms(B, A, scale):
    """|(scale * B @ A)[:, n]| for every column n."""
    return scale * (((B.T @ B) @ A) * A).sum(0).clamp_min(0).sqrt()


def apply_config(base_sd, lora, cfg):
    """base + the part of GRPO's dW that cfg selects, in the base dtype.
    cfg: {"attn": [layers], "mlp": [layers], "neurons": [[L, n], ...],
          "neuron_mode": "only" | "except"}.
    "only" applies dW to the listed MLP neurons; "except" applies it to all
    MLP neurons but those. Attention follows cfg["attn"] in both cases."""
    import torch
    sel = {}
    for L, n in cfg.get("neurons") or []:
        sel.setdefault(L, []).append(n)
    mode = cfg.get("neuron_mode", "only")
    out = dict(base_sd)
    for key, (B, A, s) in lora.items():
        L, mod = module_of(key)
        if mod in ATTN:
            if L not in cfg["attn"]:
                continue
            d = s * B @ A
        elif L in cfg["mlp"]:
            d = s * B @ A
        elif L in sel or (mode == "except" and cfg.get("neurons")):
            idx = torch.tensor(sorted(sel.get(L, [])), dtype=torch.long)
            if mode == "except":
                keep = torch.ones(D_FF, dtype=torch.bool)
                keep[idx] = False
                idx = keep.nonzero().squeeze(1)
            d = torch.zeros(B.shape[0], A.shape[1])
            if mod == "down_proj":
                d[:, idx] = s * B @ A[:, idx]
            else:
                d[idx] = s * B[idx] @ A
        else:
            continue
        out[key] = (base_sd[key].float() + d).to(base_sd[key].dtype)
    return out


def write_model(sd, out_dir, like_dir=RL):
    """Weights from sd; config, tokenizer and generation config from like_dir."""
    import os, shutil
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(like_dir):
        if not f.endswith(".safetensors") and not f.endswith(".index.json"):
            shutil.copy(f"{like_dir}/{f}", out_dir)
    save_file({k: v.contiguous() for k, v in sd.items()}, f"{out_dir}/model.safetensors",
              metadata={"format": "pt"})


# ====================================================================== scores
def weight_scores(base_sd, lora):
    """Per-neuron relative weight change, per-neuron |dW|^2, per-module and per-head
    relative |dW|_F. Everything is computed from A and B, never from merged weights."""
    import torch
    rel = torch.zeros(N_LAYERS, D_FF)
    sq = torch.zeros(N_LAYERS, D_FF)
    modules, heads = {}, {}
    q_o = {}
    for key, (B, A, s) in lora.items():
        L, mod = module_of(key)
        W = base_sd[key].float()
        if mod in ("down_proj", "o_proj"):    # neurons / heads are input columns here
            d, w = col_norms(B, A, s), W.norm(dim=0)
        else:
            d, w = row_norms(B, A, s), W.norm(dim=1)
        modules[f"{L}.{mod}"] = (d.square().sum().sqrt() / W.norm()).item()
        if mod in MLP:
            rel[L] += d / w
            sq[L] += d.square()
        if mod in ("q_proj", "o_proj"):
            # q rows and o columns of head h are its HEAD_DIM-wide slice
            q_o[(L, mod)] = (d.square().view(N_HEADS, HEAD_DIM).sum(1),
                             w.square().view(N_HEADS, HEAD_DIM).sum(1))
    for L in range(N_LAYERS):
        dq, wq = q_o[(L, "q_proj")]
        do, wo = q_o[(L, "o_proj")]
        heads[L] = ((dq + do) / (wq + wo)).sqrt().tolist()
    return {"rel": rel, "sq": sq, "modules": modules, "heads": heads}


def acts_of(model, ids, P):
    """[n_layers, completion tokens, d_ff] activations entering down_proj, float32."""
    import torch
    acts, handles = {}, []
    for L, layer in enumerate(model.model.layers):
        def hook(mod, inp, L=L):
            acts[L] = inp[0][0, P:].float()
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([acts[L] for L in range(len(acts))])


def activation_scores(tok, base, rl, texts, device=DEVICE):
    """texts: [(prompt, completion)]. Both models read the same tokens; the score
    is mean |a_rl - a_base| / std(a_base) over completion tokens."""
    import torch
    nL = len(base.model.layers)
    s_abs, s_d, s_b, s_b2 = (torch.zeros(nL, D_FF, device=device) for _ in range(4))
    n = 0
    for prompt, comp in texts:
        P = len(tok(prompt, add_special_tokens=False).input_ids)
        ids = tok(prompt + comp, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        ab, ar = acts_of(base, ids, P), acts_of(rl, ids, P)
        s_abs += (ar - ab).abs().sum(1)
        s_d += (ar - ab).sum(1)
        s_b += ab.sum(1)
        s_b2 += ab.square().sum(1)
        n += ab.shape[1]
    std = (s_b2 / n - (s_b / n).square()).clamp_min(1e-12).sqrt()
    return {"abs": (s_abs / n / std).cpu(), "mean": (s_d / n).cpu(), "std": std.cpu(), "tokens": n}


def ranks_desc(x):
    """1 = largest, over the flattened array."""
    import numpy as np
    flat = np.asarray(x).ravel()
    r = np.empty(flat.size, dtype=np.int64)
    r[np.argsort(-flat, kind="stable")] = np.arange(1, flat.size + 1)
    return r.reshape(np.asarray(x).shape)


def spearman(a, b):
    import numpy as np
    return float(np.corrcoef(ranks_desc(a).ravel(), ranks_desc(b).ravel())[0, 1])


def top_neurons(score, k):
    """[[layer, neuron], ...] of the k largest entries."""
    import numpy as np
    idx = np.argsort(-np.asarray(score).ravel(), kind="stable")[:k]
    return [[int(i // D_FF), int(i % D_FF)] for i in idx]


def random_like(neurons, seed):
    """Random neurons with the same count per layer as `neurons`."""
    import random
    rng = random.Random(seed)
    per = {}
    for L, _ in neurons:
        per[L] = per.get(L, 0) + 1
    return [[L, n] for L in sorted(per) for n in sorted(rng.sample(range(D_FF), per[L]))]


def build_configs(rel, seed=0):
    every = list(range(N_LAYERS))
    c = lambda name, attn=(), mlp=(), neurons=None, mode="only": {
        "name": name, "attn": list(attn), "mlp": list(mlp),
        "neurons": neurons or [], "neuron_mode": mode}
    top = {k: top_neurons(rel, k) for k in TOP_K}
    rand = {k: random_like(top[k], seed) for k in TOP_K[1:]}
    cfgs = [c("base"), c("full", every, every),
            c("attn_only", attn=every), c("mlp_only", mlp=every)]
    for a, b in LAYER_GROUPS:
        g = range(a, b + 1)
        cfgs.append(c(f"L{a:02d}-{b:02d}", g, g))
    cfgs += [c(f"top{k}", neurons=top[k]) for k in TOP_K]
    cfgs += [c(f"rand{k}", neurons=rand[k]) for k in TOP_K[1:]]
    km = TOP_K[1]                                  # the removal tests use the middle k
    cfgs += [c(f"grpo_minus_top{km}", attn=every, neurons=top[km], mode="except"),
             c(f"grpo_minus_rand{km}", attn=every, neurons=rand[km], mode="except")]
    return cfgs


# ====================================================================== step 1
def load_hf(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, memory=32768,
              timeout=3 * 3600)
def diff(tag: str = "1.5b", n_act: int = 400, seed: int = 0):
    import json, os, random
    import numpy as np
    from transformers import AutoTokenizer

    base_sd = load_base_state()
    lora = load_lora(ADAPTER)
    w = weight_scores(base_sd, lora)
    rel, sq = w["rel"].numpy(), w["sq"].numpy()
    del base_sd
    print("weight scores done")

    tok = AutoTokenizer.from_pretrained(BASE)
    dev = json.load(open(DEV_JSON))
    preds = {k: open(p).read().splitlines() for k, p in PRED.items()}
    pick = sorted(random.Random(seed).sample(range(len(dev)), min(n_act, len(dev))))
    cache, texts = {}, []
    for i in pick:
        prompt = make_prompt(tok, dev[i], cache)[0]
        for who in ("base", "rl"):
            texts.append((prompt, f"```sql\n{preds[who][i]}\n```"))
    base, rl = load_hf(BASE), load_hf(RL)
    act = activation_scores(tok, base, rl, texts)
    a_abs = act["abs"].numpy()
    print(f"activation scores done: {len(texts)} sequences, {act['tokens']} tokens")

    os.makedirs(f"{DATA}/e5", exist_ok=True)
    out = {"base_id": BASE, "rl_path": RL, "adapter": ADAPTER,
           "n_act_questions": len(pick), "n_act_tokens": act["tokens"]}

    # ---- where the change sits
    flat = np.sort(sq.ravel())[::-1]
    out["concentration"] = {f"top{p}%": float(flat[:max(1, int(flat.size * p / 100))].sum() / flat.sum())
                            for p in (0.1, 1, 10)}
    out["layer_rel"] = {
        kind: [float(np.sqrt(sum(w["modules"][f"{L}.{m}"] ** 2 for m in mods) / len(mods)))
               for L in range(N_LAYERS)]
        for kind, mods in (("attn", ATTN), ("mlp", MLP))}
    out["module_rel"] = w["modules"]
    out["head_rel"] = w["heads"]

    # ---- the rankings, and how they agree with each other and with E4
    r_w, r_a = ranks_desc(rel), ranks_desc(a_abs)
    e4_path = f"{DATA}/e4/scores_{tag}.npz"
    atp = np.load(e4_path)["atp"] if os.path.exists(e4_path) else None
    r_atp = ranks_desc(np.abs(atp)) if atp is not None else None
    out["spearman"] = {"weight_vs_act": spearman(rel, a_abs)}
    if atp is not None:
        out["spearman"]["weight_vs_atp"] = spearman(rel, np.abs(atp))
        out["spearman"]["act_vs_atp"] = spearman(a_abs, np.abs(atp))
    out["e4_neurons"] = [{
        "layer": L, "neuron": n,
        "weight_rank": int(r_w[L, n]), "act_rank": int(r_a[L, n]),
        "atp_abs_rank": int(r_atp[L, n]) if r_atp is not None else None,
        "act_mean_shift": float(act["mean"][L, n])} for L, n in E4_NEURONS]
    e4_find = f"{DATA}/e4/find_{tag}.json"
    if os.path.exists(e4_find):
        e4_top = {(x["layer"], x["neuron"]) for x in json.load(open(e4_find))["top_atp"][:50]}
        for name, score in (("weight", rel), ("act", a_abs)):
            mine = {tuple(x) for x in top_neurons(score, 50)}
            out[f"e4_top50_overlap_{name}"] = len(mine & e4_top)
    out["top_weight"] = [{"layer": L, "neuron": n, "rel": float(rel[L, n]),
                          "act_rank": int(r_a[L, n]),
                          "atp_abs_rank": int(r_atp[L, n]) if r_atp is not None else None}
                         for L, n in top_neurons(rel, 50)]
    out["top_act"] = [{"layer": L, "neuron": n, "act": float(a_abs[L, n]),
                       "weight_rank": int(r_w[L, n])} for L, n in top_neurons(a_abs, 50)]
    for k in TOP_K:
        layers = [L for L, _ in top_neurons(rel, k)]
        out[f"top{k}_per_layer"] = [layers.count(L) for L in range(N_LAYERS)]

    cfgs = build_configs(rel, seed)
    np.savez_compressed(f"{DATA}/e5/scores_{tag}.npz", weight_rel=rel, weight_sq=sq,
                        act_abs=a_abs, act_mean=act["mean"].numpy(), act_std=act["std"].numpy())
    json.dump(out, open(f"{DATA}/e5/diff_{tag}.json", "w"), indent=2)
    json.dump(cfgs, open(f"{DATA}/e5/configs_{tag}.json", "w"))
    vol.commit()
    print(json.dumps({k: out[k] for k in ("concentration", "spearman", "e4_neurons")}, indent=2))
    print("configs:", [x["name"] for x in cfgs])


# ====================================================================== step 2
@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, memory=32768,
              timeout=3 * 3600)
def run(cfg: dict, n_dev: int = 0, k: int = 8):
    import gc, json, os, re
    from collections import Counter
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_dir = f"/tmp/e5_{cfg['name']}"
    write_model(apply_config(load_base_state(), load_lora(ADAPTER), cfg), model_dir)
    gc.collect()

    tok = AutoTokenizer.from_pretrained(model_dir)
    dev = json.load(open(DEV_JSON))
    if n_dev:
        dev = dev[:n_dev]
    cache = {}
    prompts, dbps = zip(*[make_prompt(tok, ex, cache) for ex in dev])
    llm = LLM(model=model_dir, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096)
    greedy = llm.generate(list(prompts), SamplingParams(temperature=0.0, max_tokens=256))
    preds = [extract_sql(o.outputs[0].text) or "SELECT 1" for o in greedy]
    sampled = llm.generate(list(prompts), SamplingParams(
        temperature=0.8, top_p=0.95, n=k, max_tokens=256))

    d = f"{DATA}/e5/runs/{cfg['name']}"
    os.makedirs(d, exist_ok=True)
    gold_p, pred_p = f"{d}/gold.txt", f"{d}/pred.txt"
    with open(gold_p, "w") as f:
        f.writelines(f"{ex['query']}\t{ex['db_id']}\n" for ex in dev)
    with open(pred_p, "w") as f:
        f.write("\n".join(preds) + "\n")
    outs = {"spider_all": run_eval("/opt/spider", gold_p, pred_p, DB_DIR, "all"),
            "ts_single": run_eval("/opt/test-suite", gold_p, pred_p, DB_DIR, "exec"),
            "ts_suite": run_eval("/opt/test-suite", gold_p, pred_p, TS_DB, "exec")}
    for name, text in outs.items():
        open(f"{d}/{name}_eval.txt", "w").write(text)
    T = {name: parse_table(text) for name, text in outs.items()}

    gold = [ex["query"] for ex in dev]
    hit1 = [exec_match(db, p, g) for db, p, g in zip(dbps, preds, gold)]
    hitk = [any(exec_match(dbps[i], extract_sql(c.text), gold[i]) for c in sampled[i].outputs)
            for i in range(len(dev))]
    setop = [i for i, g in enumerate(gold) if re.search(r"\b(intersect|union|except)\b", g.lower())]
    st = [style_of(p) for p in preds]
    row = {
        "name": cfg["name"], "n_dev": len(dev),
        "n_neurons": len(cfg.get("neurons") or []), "neuron_mode": cfg.get("neuron_mode"),
        "attn_layers": cfg["attn"], "mlp_layers": cfg["mlp"],
        "EX_testsuite_single": get(T["ts_single"], "exec", "execution"),
        "TS_suite": get(T["ts_suite"], "exec", "execution"),
        "EX_official_spider": get(T["spider_all"], "exec", "execution"),
        "EM_official_spider": get(T["spider_all"], "exact", "exact match"),
        "setop_n": len(setop),
        "setop_exec_acc_loose": float(np.mean([hit1[i] for i in setop])) if setop else None,
        "pass@1_loose": float(np.mean(hit1)),
        f"pass@{k}_loose": float(np.mean(hitk)),
        "valid_sql": float(np.mean([safe_execute(db, p) is not None for db, p in zip(dbps, preds)])),
        "alias_per_q": float(np.mean([s["alias"] for s in st])),
        "qual_per_q": float(np.mean([s["qual"] for s in st])),
        "join_per_q": float(np.mean([s["join"] for s in st])),
        "examples": [{"q": ex["question"], "sql": p} for ex, p in zip(dev[:5], preds[:5])],
    }
    json.dump(row, open(f"{d}/row.json", "w"), indent=2)
    vol.commit()
    print({k_: v for k_, v in row.items() if k_ != "examples"})
    return row


ACC = ("EX_testsuite_single", "TS_suite", "EX_official_spider", "EM_official_spider",
       "setop_exec_acc_loose", "pass@1_loose", "pass@8_loose")
STYLE_COLS = ("valid_sql", "alias_per_q", "qual_per_q", "join_per_q")
HOLD_TOL = 0.005   # "no metric below base" allows 0.5 points of decoding noise


def add_recovery(rows):
    """Per accuracy metric: percent of the base -> full gain this config recovers,
    and whether it stays at or above base on every metric."""
    by = {r["name"]: r for r in rows}
    b, f = by.get("base"), by.get("full")
    for r in rows:
        if not (b and f):
            break
        rec = {}
        for m in ACC:
            x, lo, hi = r.get(m), b.get(m), f.get(m)
            rec[m] = None if None in (x, lo, hi) or hi == lo else 100 * (x - lo) / (hi - lo)
        r["recovered_pct"] = rec
        r["holds_base"] = all(r.get(m) is not None and b.get(m) is not None
                              and r[m] >= b[m] - HOLD_TOL for m in ACC)
    return rows


def gate(rows):
    """base and full must reproduce the main eval (greedy metrics only)."""
    import json, os
    if not os.path.exists(REEVAL):
        return
    by = {r["name"]: r for r in rows}
    refs = {r["tag"]: r for r in json.load(open(REEVAL))}
    for name, tag in REEVAL_TAG.items():
        r, ref = by.get(name), refs.get(tag, {}).get("EX_testsuite_single")
        if not r or r["n_dev"] != 1034 or ref is None or r["EX_testsuite_single"] is None:
            continue
        ok = abs(r["EX_testsuite_single"] - ref) <= HOLD_TOL
        print(f"gate {name}: EX {r['EX_testsuite_single']:.3f} vs main eval {ref:.3f} "
              f"-> {'ok' if ok else 'FAIL, do not read the other rows'}")


def print_table(rows):
    cols = ACC + STYLE_COLS
    print(f"\n{'config':22}" + "".join(f"{c.replace('_loose', '')[:10]:>11}" for c in cols)
          + f"{'holds':>7}")
    for r in rows:
        print(f"{r['name']:22}" + "".join(
            f"{r[c]:>11.3f}" if isinstance(r.get(c), float) else f"{'n/a':>11}" for c in cols)
            + f"{str(r.get('holds_base', '')):>7}")


def write_summary(rows, tag):
    import json, os
    vol.reload()                                   # see dirs/files the workers committed
    os.makedirs(f"{DATA}/e5", exist_ok=True)
    order = [c["name"] for c in json.load(open(f"{DATA}/e5/configs_{tag}.json"))]
    rows = add_recovery(sorted(rows, key=lambda r: order.index(r["name"])
                               if r["name"] in order else len(order)))
    json.dump(rows, open(f"{DATA}/e5/summary_{tag}.json", "w"), indent=2)
    vol.commit()
    print_table(rows)
    gate(rows)


@app.function(image=image, volumes={DATA: vol}, timeout=8 * 3600)
def sweep(tag: str = "1.5b", n_dev: int = 0, k: int = 8):
    import json
    cfgs = json.load(open(f"{DATA}/e5/configs_{tag}.json"))
    rows = list(run.starmap([(c, n_dev, k) for c in cfgs], return_exceptions=True))
    for c, r in zip(cfgs, rows):
        if not isinstance(r, dict):
            print(f"{c['name']} failed: {r!r}")
    write_summary([r for r in rows if isinstance(r, dict)], tag)


@app.function(image=image, volumes={DATA: vol}, timeout=12 * 3600)
def pipeline(tag: str = "1.5b", n_dev: int = 0, n_act: int = 400, k: int = 8):
    """diff, then sweep, in one detached run."""
    diff.remote(tag=tag, n_act=n_act)
    vol.reload()
    sweep.remote(tag=tag, n_dev=n_dev, k=k)


@app.function(image=image, volumes={DATA: vol})
def summarize(tag: str = "1.5b"):
    import json, glob
    vol.reload()
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{DATA}/e5/runs/*/row.json"))]
    write_summary(rows, tag)
