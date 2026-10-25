"""
E4: Which MLP neurons make Qwen2.5-Coder-1.5B write table aliases (T1, T2),
and what happens to accuracy when we steer them?

Background. Base 1.5B writes 2.63 alias mentions per query; after GRPO it writes
0.36 and switches to table.column. The reward never mentions style. E4 looks
for the neurons behind that preference and intervenes on them directly.

Step 1  find   (one A100-80GB, ~30-45 min)
    a. On Spider TRAIN questions (dev is kept clean for evaluation), generate the
       base model's answer twice: once told to use aliases, once told not to.
       Keep pairs where both versions are correct by execution, the alias
       version has >= 2 alias mentions and the other has none.
    b. Alias preference of a model on a pair, under the ORIGINAL prompt:
           m = log p(alias version) - log p(qualified version)
       m > 0 means the model prefers aliases. We report m for base and GRPO.
    c. For every MLP neuron (input of down_proj, 28 x 8960 = 250,880 neurons):
         steer score  = d m / d(constant added to the neuron at every position),
                        scaled by the steering recipe below. Ranks neurons by
                        how much steering them should move alias preference.
         AtP score    = sum_pos (a_GRPO - a_base) * d m / d a  (attribution
                        patching). Estimates how much of the GRPO style shift
                        runs through each neuron.
         zero-abl     = sum_pos (0 - a_base) * d m / d a.
    d. Screen the top candidates for real: steer each one alone at dose 1 on
       held-out pairs and measure the actual change in m.
    e. Logit lens for the winners: which tokens each neuron writes toward.

Step 2  sweep  (one container per config, in parallel)
    Steering recipe (from the "evil Qwen" post), applied to the activation
    entering down_proj, at every position:
        a_n += sign * dose * median(||resid leaving layer L||, tokens 1:) / ||W_down[:, n]||
    so the vector added to the residual stream has norm dose * median residual norm.
    Generates greedy SQL on the full Spider dev set under each config and
    reports EX, test-suite accuracy, valid-SQL rate and alias statistics.
    Controls: random neurons in the same layers at the same dose, the
    opposite sign (more aliases), and the same neurons on the GRPO model.

Commands
    modal run --detach e4_alias_neurons.py::find
    modal run --detach e4_alias_neurons.py::sweep        # after find finishes
    modal run e4_alias_neurons.py::summarize             # prints the table again

Outputs on the volume
    /data/e4/find_1.5b.json, /data/e4/scores_1.5b.npz, /data/e4/pairs_1.5b.jsonl
    /data/e4/runs/<config>/{pred.txt,row.json}, /data/e4/sweep_1.5b.json
"""

import modal

app = modal.App("spider-e4")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "numpy", "nltk", "sqlparse")
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
TRAIN_JSON = f"{SPIDER}/train_spider.json"
TS_DB = f"{DATA}/ts_db/database"

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RL = "/data/merged/1.5b-grpo-binary-7k"
DEVICE = "cuda"

# Byte-identical to every earlier eval when style == "".
PROMPT_TMPL = (
    "Database schema:\n{schema}\n\nWrite a single SQLite query that answers "
    "the question. Return ONLY the query inside a ```sql code block.{style}"
    "\n\nQuestion: {question}"
)
STYLE = {
    "default": "",
    "no_alias": (" Do not use table aliases such as T1 or T2. Write full table "
                 "names and refer to columns as table.column."),
    "alias": (" Give every table an alias T1, T2, ... using AS, and refer to "
              "every column through its alias, for example T1.name."),
}
FENCE_OPEN, FENCE_CLOSE = "```sql\n", "\n```"
FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


# ====================================================================== SQL helpers
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


def style_of(sql):
    """Same regexes as E2/E3, so numbers are comparable."""
    import re
    low = sql.lower()
    return {"alias": len(re.findall(r"\bt\d+\b", low)),
            "qual": len(re.findall(r"\b[a-z_]+\.[a-z_]+", low)),
            "join": len(re.findall(r"\bjoin\b", low))}


def run_eval(cwd, gold_p, pred_p, db, etype):
    import subprocess
    r = subprocess.run(
        ["python", "evaluation.py", "--gold", gold_p, "--pred", pred_p,
         "--db", db, "--table", TABLES, "--etype", etype],
        cwd=cwd, capture_output=True, text=True)
    return r.stdout


def parse_row(out, label):
    for line in out.splitlines():
        if line.strip().startswith(label):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


# ====================================================================== model helpers
def load(model_id):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    m = m.to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def make_prompt(tok, ex, style, cache):
    dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
    if dbp not in cache:
        cache[dbp] = build_schema_string(dbp)
    msg = [{"role": "user", "content": PROMPT_TMPL.format(
        schema=cache[dbp], style=STYLE[style], question=ex["question"])}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True), dbp


def generate(model, tok, prompts, max_new=256, bs=32):
    """Greedy, batched, left-padded. Overrides Qwen's generation_config
    (repetition_penalty etc.) so this matches plain greedy decoding."""
    import torch
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    with torch.no_grad():
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(DEVICE)
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               temperature=None, top_p=None, top_k=None,
                               repetition_penalty=1.0, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(g[:, enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
            for i, t in zip(idx, texts):
                out[i] = t
    return out


def layers_of(model):
    return model.model.layers


class Capture:
    """Records the activation entering every down_proj (the MLP neurons)."""

    def __init__(self, model, grad):
        self.model, self.grad, self.acts, self.handles = model, grad, {}, []

    def __enter__(self):
        for L, layer in enumerate(layers_of(self.model)):
            def hook(mod, inp, L=L):
                x = inp[0]
                if self.grad and x.requires_grad:
                    x.retain_grad()
                self.acts[L] = x
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


class Intervene:
    """Edits neuron activations entering down_proj, at every position.
    spec items: {"layer", "neuron", "mode": "add" | "set", "value"}."""

    def __init__(self, model, spec):
        self.model, self.spec, self.handles = model, spec or [], []

    def __enter__(self):
        import torch
        by_layer = {}
        for s in self.spec:
            by_layer.setdefault(s["layer"], []).append(s)
        for L, items in by_layer.items():
            def pick(mode):
                sel = [s for s in items if s["mode"] == mode]
                return (torch.tensor([s["neuron"] for s in sel], dtype=torch.long, device=DEVICE),
                        torch.tensor([s["value"] for s in sel], dtype=torch.float32, device=DEVICE))
            ai, av = pick("add")
            si, sv = pick("set")

            def hook(mod, inp, ai=ai, av=av, si=si, sv=sv):
                x = inp[0].clone()
                if ai.numel():
                    x[..., ai] += av.to(x.dtype)
                if si.numel():
                    x[..., si] = sv.to(x.dtype)
                return (x,)
            self.handles.append(
                layers_of(self.model)[L].mlp.down_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def resid_medians(model, tok, prompts):
    """median ||residual stream leaving layer L|| over tokens 1: (skips the
    attention-sink token 0), one value per layer."""
    import torch
    norms = {L: [] for L in range(len(layers_of(model)))}
    handles = []
    for L, layer in enumerate(layers_of(model)):
        def hook(mod, inp, out, L=L):
            h = out[0] if isinstance(out, tuple) else out
            norms[L].append(h[0, 1:].float().norm(dim=-1).cpu())
        handles.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
            model(ids)
    for h in handles:
        h.remove()
    return [torch.cat(norms[L]).median().item() for L in sorted(norms)]


def steer_spec(model, neurons, dose, med):
    """The recipe: offset = sign * dose * median_resid[L] / ||W_down[:, n]||."""
    spec = []
    for x in neurons:
        L, n = x["layer"], x["neuron"]
        w = layers_of(model)[L].mlp.down_proj.weight[:, n].float().norm().item()
        spec.append({"layer": L, "neuron": n, "mode": "add",
                     "value": x["sign"] * dose * med[L] / w})
    return spec


def completion_logp(model, ids, P, grad=False):
    """Sum of log p(completion tokens). ids = prompt (P tokens) + completion."""
    import torch
    import torch.nn.functional as F
    C = ids.shape[1] - P
    with torch.set_grad_enabled(grad):
        logits = model(ids, logits_to_keep=C + 1).logits[0, :-1].float()
        return F.log_softmax(logits, -1).gather(-1, ids[0, P:, None]).sum()


def pair_pass(model, ids, P, grad):
    """Returns (logp, acts[L] over tokens 1:, grads[L] or None), float32."""
    with Capture(model, grad=grad) as cap:
        lp = completion_logp(model, ids, P, grad=grad)
        if grad:
            lp.backward()
    acts = [cap.acts[L][0, 1:].detach().float() for L in range(len(cap.acts))]
    grads = ([cap.acts[L].grad[0, 1:].float() for L in range(len(cap.acts))]
             if grad else None)
    return lp.item(), acts, grads


def accumulate_pair(base, rl, idsA, PA, idsQ, PQ, acc):
    """One alias/qualified pair. m = logp(A) - logp(Q). Adds per-neuron
    statistics into acc (dict of [n_layers, d_ff] float32 tensors)."""
    lpA_b, aA_b, gA = pair_pass(base, idsA, PA, grad=True)
    lpQ_b, aQ_b, gQ = pair_pass(base, idsQ, PQ, grad=True)
    lpA_r, aA_r, _ = pair_pass(rl, idsA, PA, grad=False)
    lpQ_r, aQ_r, _ = pair_pass(rl, idsQ, PQ, grad=False)
    for L in range(len(gA)):
        acc["steer_grad"][L] += gA[L].sum(0) - gQ[L].sum(0)
        acc["atp"][L] += ((aA_r[L] - aA_b[L]) * gA[L]).sum(0) \
            - ((aQ_r[L] - aQ_b[L]) * gQ[L]).sum(0)
        acc["zero_abl"][L] += (-aA_b[L] * gA[L]).sum(0) - (-aQ_b[L] * gQ[L]).sum(0)
        acc["dact"][L] += (aA_r[L] - aA_b[L]).mean(0) + (aQ_r[L] - aQ_b[L]).mean(0)
    return lpA_b - lpQ_b, lpA_r - lpQ_r


def encode_pair(tok, prompt, sql):
    import torch
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(FENCE_OPEN + sql + FENCE_CLOSE, add_special_tokens=False).input_ids
    return torch.tensor([p + c], device=DEVICE), len(p)


def logit_lens(model, tok, L, n, k=8):
    import torch
    v = layers_of(model)[L].mlp.down_proj.weight[:, n].float() * model.model.norm.weight.float()
    s = (model.lm_head.weight.float() @ v)
    up = tok.convert_ids_to_tokens(s.topk(k).indices.tolist())
    down = tok.convert_ids_to_tokens((-s).topk(k).indices.tolist())
    return up, down


# ====================================================================== step 1
@app.function(image=image, gpu="A100-80GB", volumes={DATA: vol}, timeout=4 * 3600)
def find(base_id: str = BASE, rl_path: str = RL, tag: str = "1.5b",
         n_train: int = 800, n_screen: int = 32, n_cand: int = 40, seed: int = 0):
    import json, os, random
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    os.makedirs(f"{DATA}/e4", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_id)
    base, rl = load(base_id), load(rl_path)
    base.enable_input_require_grads()
    nL = len(layers_of(base))
    d_ff = layers_of(base)[0].mlp.down_proj.weight.shape[1]

    # ---- a. paired generations on train questions
    train = json.load(open(TRAIN_JSON))
    sample = random.Random(seed).sample(train, n_train)
    cache = {}
    P_def, dbps = zip(*[make_prompt(tok, ex, "default", cache) for ex in sample])
    P_ali = [make_prompt(tok, ex, "alias", cache)[0] for ex in sample]
    P_qua = [make_prompt(tok, ex, "no_alias", cache)[0] for ex in sample]
    ga = [extract_sql(t) for t in generate(base, tok, P_ali)]
    gq = [extract_sql(t) for t in generate(base, tok, P_qua)]
    pairs = []
    for i, ex in enumerate(sample):
        if (style_of(ga[i])["alias"] >= 2 and style_of(gq[i])["alias"] == 0
                and exec_match(dbps[i], ga[i], ex["query"])
                and exec_match(dbps[i], gq[i], ex["query"])):
            pairs.append({"q": ex["question"], "db": dbps[i], "prompt": P_def[i],
                          "alias": ga[i], "qual": gq[i]})
    print(f"pairs kept: {len(pairs)} / {n_train}")
    with open(f"{DATA}/e4/pairs_{tag}.jsonl", "w") as f:
        for p in pairs:
            f.write(json.dumps({k: v for k, v in p.items() if k != "prompt"}) + "\n")
    if len(pairs) < n_screen + 50:
        raise RuntimeError("too few pairs; raise n_train")
    attr_pairs, screen_pairs = pairs[:-n_screen], pairs[-n_screen:]

    # ---- recipe scale
    med = resid_medians(base, tok, list(P_def[:32]))

    # ---- b, c. attribution
    acc = {k: torch.zeros(nL, d_ff, device=DEVICE)
           for k in ("steer_grad", "atp", "zero_abl", "dact")}
    m_b, m_r = [], []
    for j, p in enumerate(attr_pairs):
        idsA, PA = encode_pair(tok, p["prompt"], p["alias"])
        idsQ, PQ = encode_pair(tok, p["prompt"], p["qual"])
        mb, mr = accumulate_pair(base, rl, idsA, PA, idsQ, PQ, acc)
        m_b.append(mb); m_r.append(mr)
        if j % 50 == 0:
            print(f"attr {j}/{len(attr_pairs)}  m_base={np.mean(m_b):.2f}  m_rl={np.mean(m_r):.2f}")
    n = len(attr_pairs)
    acc = {k: (v / n).cpu() for k, v in acc.items()}
    wnorm = torch.stack([l.mlp.down_proj.weight.float().norm(dim=0).cpu()
                         for l in layers_of(base)])                       # [nL, d_ff]
    scale = torch.tensor(med)[:, None] / wnorm                            # offset per unit dose
    pred_dm = -(acc["steer_grad"].abs() * scale)                          # steer toward fewer aliases
    np.savez_compressed(f"{DATA}/e4/scores_{tag}.npz",
                        **{k: v.numpy() for k, v in acc.items()},
                        pred_dm=pred_dm.numpy(), wnorm=wnorm.numpy())

    def top(t, k, largest):
        flat = t.flatten()
        idx = flat.topk(k, largest=largest).indices.tolist()
        return [(i // d_ff, i % d_ff, flat[i].item()) for i in idx]

    atp_rank = acc["atp"].flatten().argsort().argsort().view(nL, d_ff)   # 0 = most alias-reducing

    # ---- d. screen candidates by real single-neuron steering on held-out pairs
    def m_on(pairs_, spec):
        out = []
        with Intervene(base, spec):
            for p in pairs_:
                idsA, PA = encode_pair(tok, p["prompt"], p["alias"])
                idsQ, PQ = encode_pair(tok, p["prompt"], p["qual"])
                a = completion_logp(base, idsA, PA).item()
                q = completion_logp(base, idsQ, PQ).item()
                out.append((a, q))
        return out

    ref = m_on(screen_pairs, [])
    m_ref = np.mean([a - q for a, q in ref])
    q_ref = np.mean([q for _, q in ref])
    screened = []
    for L, nn, pd in top(pred_dm, n_cand, largest=False):
        sign = -1 if acc["steer_grad"][L, nn] > 0 else 1
        res = m_on(screen_pairs, steer_spec(base, [{"layer": L, "neuron": nn, "sign": sign}], 1.0, med))
        up, down = logit_lens(base, tok, L, nn)
        screened.append({
            "layer": L, "neuron": nn, "sign": sign,
            "pred_dm": pd,
            "dm": float(np.mean([a - q for a, q in res]) - m_ref),
            "d_logp_qual": float(np.mean([q for _, q in res]) - q_ref),
            "atp": acc["atp"][L, nn].item(),
            "atp_rank": int(atp_rank[L, nn]),
            "rl_shift": acc["dact"][L, nn].item(),
            "rl_shift_agrees": bool((acc["dact"][L, nn].item() > 0) == (sign > 0)),
            "lens_up": up, "lens_down": down,
        })
        print(f"L{L:>2} N{nn:<5} sign {sign:+d}  pred {pd:+.2f}  real dm {screened[-1]['dm']:+.2f}"
              f"  atp_rank {screened[-1]['atp_rank']}")
    screened.sort(key=lambda r: r["dm"])

    summary = {
        "base_id": base_id, "rl_path": rl_path, "n_pairs_attr": n,
        "n_pairs_screen": len(screen_pairs),
        "m_base_mean": float(np.mean(m_b)), "m_rl_mean": float(np.mean(m_r)),
        "m_base_frac_prefers_alias": float(np.mean(np.array(m_b) > 0)),
        "m_rl_frac_prefers_alias": float(np.mean(np.array(m_r) > 0)),
        "resid_median": med,
        "screened": screened,
        "top_atp": [{"layer": L, "neuron": nn, "atp": s} for L, nn, s in top(acc["atp"], 50, False)],
        "top_zero_abl": [{"layer": L, "neuron": nn, "score": s}
                         for L, nn, s in top(acc["zero_abl"], 50, False)],
        "atp_total": acc["atp"].sum().item(),
        "atp_top100_share": float(-torch.tensor([s for *_, s in top(acc["atp"], 100, False)]).sum()
                                  / -acc["atp"].clamp(max=0).sum()),
    }
    json.dump(summary, open(f"{DATA}/e4/find_{tag}.json", "w"), indent=2)
    vol.commit()

    print(f"\nalias preference m (log-odds, pairs={n}): base {summary['m_base_mean']:+.2f}"
          f"  GRPO {summary['m_rl_mean']:+.2f}")
    print(f"base prefers alias on {summary['m_base_frac_prefers_alias']:.0%} of pairs,"
          f" GRPO on {summary['m_rl_frac_prefers_alias']:.0%}")
    print(f"GRPO shift explained by top-100 neurons (AtP, linear est.): {summary['atp_top100_share']:.0%}")
    print("\nbest neurons by real effect on alias preference (dose 1):")
    for r in screened[:8]:
        print(f"  L{r['layer']} N{r['neuron']} {r['sign']:+d}  dm {r['dm']:+.2f}  "
              f"d logp(qual) {r['d_logp_qual']:+.2f}  GRPO moved it same way: {r['rl_shift_agrees']}"
              f"  writes toward {r['lens_up'][:4] if r['sign'] > 0 else r['lens_down'][:4]}")


# ====================================================================== step 2
@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, timeout=3 * 3600)
def steer(cfg: dict):
    import json, os
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = load(cfg["model"])
    cache = {}
    train = json.load(open(TRAIN_JSON))
    med = resid_medians(model, tok, [make_prompt(tok, ex, "default", cache)[0]
                                     for ex in train[:32]])
    if cfg["mode"] == "add":
        spec = steer_spec(model, cfg["neurons"], cfg["dose"], med)
    elif cfg["mode"] == "zero":
        spec = [{"layer": x["layer"], "neuron": x["neuron"], "mode": "set", "value": 0.0}
                for x in cfg["neurons"]]
    else:
        spec = []

    dev = json.load(open(DEV_JSON))
    if cfg.get("n_dev"):
        dev = dev[:cfg["n_dev"]]
    prompts, dbps = zip(*[make_prompt(tok, ex, "default", cache) for ex in dev])
    with Intervene(model, spec):
        outs = generate(model, tok, list(prompts))
    preds = [extract_sql(o) or "SELECT 1" for o in outs]

    d = f"{DATA}/e4/runs/{cfg['name']}"
    os.makedirs(d, exist_ok=True)
    gold_p, pred_p = f"{d}/gold.txt", f"{d}/pred.txt"
    with open(gold_p, "w") as f:
        f.writelines(f"{ex['query']}\t{ex['db_id']}\n" for ex in dev)
    with open(pred_p, "w") as f:
        f.write("\n".join(preds) + "\n")

    st = [style_of(p) for p in preds]
    row = {
        "name": cfg["name"], "model": cfg["model"], "mode": cfg["mode"],
        "dose": cfg.get("dose"), "neurons": cfg.get("neurons", []),
        "EX": parse_row(run_eval("/opt/spider", gold_p, pred_p, DB_DIR, "all"), "execution"),
        "TS_single": parse_row(run_eval("/opt/test-suite", gold_p, pred_p, DB_DIR, "exec"), "execution"),
        "TS_suite": parse_row(run_eval("/opt/test-suite", gold_p, pred_p, TS_DB, "exec"), "execution"),
        "valid_sql": float(np.mean([safe_execute(db, p) is not None for db, p in zip(dbps, preds)])),
        "alias_per_q": float(np.mean([s["alias"] for s in st])),
        "frac_any_alias": float(np.mean([s["alias"] > 0 for s in st])),
        "qual_per_q": float(np.mean([s["qual"] for s in st])),
        "join_per_q": float(np.mean([s["join"] for s in st])),
        "examples": [{"q": ex["question"], "sql": p} for ex, p in zip(dev[:5], preds[:5])],
    }
    json.dump(row, open(f"{d}/row.json", "w"), indent=2)
    vol.commit()
    print(row)
    return row


def build_configs(f, n_dev, seed=0):
    import random
    rng = random.Random(seed)
    B, R = f["base_id"], f["rl_path"]
    keep = lambda xs: [{"layer": x["layer"], "neuron": x["neuron"], "sign": x["sign"]} for x in xs]
    t2, t8 = keep(f["screened"][:2]), keep(f["screened"][:8])
    flip = lambda xs: [{**x, "sign": -x["sign"]} for x in xs]
    rand_like = lambda xs: [{"layer": x["layer"], "neuron": rng.randrange(8960),
                             "sign": rng.choice([-1, 1])} for x in xs]
    c = lambda name, model, mode, neurons=None, dose=None: {
        "name": name, "model": model, "mode": mode, "neurons": neurons or [],
        "dose": dose, "n_dev": n_dev}
    return [
        c("base", B, "none"),
        c("base_top2_d0.5", B, "add", t2, 0.5),
        c("base_top2_d1", B, "add", t2, 1.0),       # the recipe, as posted
        c("base_top8_d0.25", B, "add", t8, 0.25),
        c("base_top8_d0.5", B, "add", t8, 0.5),
        c("base_zero_top8", B, "zero", t8),
        c("base_rand2_d1", B, "add", rand_like(t2), 1.0),      # control
        c("base_rand8_d0.5", B, "add", rand_like(t8), 0.5),    # control
        c("base_top2_d1_more_alias", B, "add", flip(t2), 1.0),  # opposite direction
        c("grpo", R, "none"),
        c("grpo_top2_d1_more_alias", R, "add", flip(t2), 1.0),  # undo GRPO's style?
    ]


COLS = ("EX", "TS_single", "TS_suite", "valid_sql", "alias_per_q", "frac_any_alias",
        "qual_per_q", "join_per_q")


def print_table(rows):
    print(f"\n{'config':28}" + "".join(f"{c[:9]:>10}" for c in COLS))
    for r in rows:
        print(f"{r['name']:28}" + "".join(
            f"{r[c]:>10.3f}" if isinstance(r.get(c), float) else f"{'n/a':>10}" for c in COLS))


@app.function(image=image, volumes={DATA: vol}, timeout=6 * 3600)
def sweep(tag: str = "1.5b", n_dev: int = 0):
    import json
    f = json.load(open(f"{DATA}/e4/find_{tag}.json"))
    rows = list(steer.map(build_configs(f, n_dev), return_exceptions=True))
    rows = [r for r in rows if isinstance(r, dict)]
    import os
    vol.reload()
    os.makedirs(f"{DATA}/e4", exist_ok=True)
    json.dump(rows, open(f"{DATA}/e4/sweep_{tag}.json", "w"), indent=2)
    vol.commit()
    print_table(rows)


@app.function(image=image, volumes={DATA: vol}, timeout=10 * 3600)
def pipeline(tag: str = "1.5b", n_train: int = 800, n_dev: int = 0):
    """find, then sweep, in one detached run."""
    find.remote(tag=tag, n_train=n_train)
    vol.reload()
    sweep.remote(tag=tag, n_dev=n_dev)


@app.function(image=image, volumes={DATA: vol})
def summarize(tag: str = "1.5b"):
    import json, glob
    vol.reload()
    rows = [json.load(open(p)) for p in sorted(glob.glob(f"{DATA}/e4/runs/*/row.json"))]
    json.dump(rows, open(f"{DATA}/e4/sweep_{tag}.json", "w"), indent=2)
    vol.commit()
    print_table(rows)
