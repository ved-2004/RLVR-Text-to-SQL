"""
E2 — Is RLVR sharpening away from gold, or just changing style?

Three analyses:
  A. PREFERENCE  at each gold token, what does the RL model actually prefer,
                 and is that preference a valid alternative (alias<->qualified
                 name, equivalent aggregate) or a genuine error?
  B. CALIBRATION token-level ECE and reliability bins for base vs RL, where
                 "correct" = argmax equals the gold token.
  C. SURFACE     style statistics of the two models' actual predictions
                 (alias use, JOIN count, DISTINCT, length, tables referenced),
                 plus how often each model's output is executionally correct
                 while textually differing from gold.

Outputs on the volume:
    /data/e2/<tag>_prefs.jsonl
    /data/e2/<tag>_summary.json
"""

import modal

app = modal.App("spider-e2")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "numpy", "sqlparse", "accelerate")
)

DATA = "/data"
SPIDER = f"{DATA}/spider_data"
DB_DIR = f"{SPIDER}/database"
DEV_JSON = f"{SPIDER}/dev.json"

PROMPT = (
    "Database schema:\n{schema}\n\nWrite a single SQLite query that answers "
    "the question. Return ONLY the query inside a ```sql code block.\n\nQuestion: {question}"
)
FENCE_OPEN, FENCE_CLOSE = "```sql\n", "\n```"
FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")
AGG = {"count", "avg", "min", "max", "sum"}


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


def schema_names(db_path):
    """Return (tables, columns) lowercased, for judging valid alternatives."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'")
    tables = [r[0].lower() for r in cur.fetchall()]
    cols = set()
    for t in tables:
        try:
            cur.execute(f'PRAGMA table_info("{t}")')
            cols |= {r[1].lower() for r in cur.fetchall()}
        except Exception:
            pass
    con.close()
    return set(tables), cols


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


def norm_sql(s):
    import re
    return re.sub(r"\s+", " ", s.strip().rstrip(";")).lower()


def surface_stats(sql):
    """Style fingerprint of a query."""
    import re
    low = sql.lower()
    return {
        "len_chars": len(sql),
        "n_alias": len(re.findall(r"\bt\d+\b", low)),
        "n_as": len(re.findall(r"\bas\b", low)),
        "n_join": len(re.findall(r"\bjoin\b", low)),
        "n_distinct": len(re.findall(r"\bdistinct\b", low)),
        "n_agg": sum(len(re.findall(rf"\b{a}\s*\(", low)) for a in AGG),
        "n_setop": len(re.findall(r"\b(intersect|union|except)\b", low)),
        "n_subq": low.count("(select"),
        "qualified": len(re.findall(r"\b[a-z_]+\.[a-z_]+", low)),
    }


@app.function(image=image, gpu="A100-80GB", volumes={DATA: vol}, timeout=7200)
def analyze(base_id: str, rl_path: str, tag: str,
            pred_base: str, pred_rl: str, limit: int = 0, topk: int = 5):
    import json, os, re, math
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import defaultdict, Counter

    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rl = AutoModelForCausalLM.from_pretrained(
        rl_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    dev = json.load(open(DEV_JSON))
    if limit:
        dev = dev[:limit]
    pb = open(pred_base).read().splitlines()
    pr = open(pred_rl).read().splitlines()

    os.makedirs(f"{DATA}/e2", exist_ok=True)
    pref_f = open(f"{DATA}/e2/{tag}_prefs.jsonl", "w")

    @torch.no_grad()
    def forward(model, ids, P, L):
        logits = model(ids).logits[0, P - 1: P - 1 + L].float()
        return F.log_softmax(logits, dim=-1)

    # ---- accumulators
    pref_counts = Counter()          # what RL prefers when it differs from gold
    role_pref = defaultdict(Counter)
    NB, NR = 20, 20                  # calibration bins
    bins_b = [[0, 0.0] for _ in range(NB)]   # [n_correct, sum_conf]
    cnt_b = [0] * NB
    bins_r = [[0, 0.0] for _ in range(NR)]
    cnt_r = [0] * NR
    surf_base, surf_rl, surf_gold = Counter(), Counter(), Counter()
    exec_ok_text_diff = {"base": 0, "rl": 0}
    exec_ok = {"base": 0, "rl": 0}

    alias_re = re.compile(r"^t\d+$", re.I)

    for i, ex in enumerate(dev):
        q = ex["query"]
        dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
        tables, cols = schema_names(dbp)

        # ---------- C. surface stats on the actual predictions
        for name, pred, acc in (("base", pb[i], surf_base), ("rl", pr[i], surf_rl)):
            for k, v in surface_stats(pred).items():
                acc[k] += v
            m = exec_match(dbp, pred, q)
            exec_ok[name] += int(m)
            if m and norm_sql(pred) != norm_sql(q):
                exec_ok_text_diff[name] += 1
        for k, v in surface_stats(q).items():
            surf_gold[k] += v

        # ---------- A + B. teacher-forced preference and calibration
        msg = [{"role": "user", "content": PROMPT.format(
            schema=build_schema_string(dbp), question=ex["question"])}]
        prompt_text = tok.apply_chat_template(msg, tokenize=False,
                                              add_generation_prompt=True)
        comp_text = FENCE_OPEN + q + FENCE_CLOSE
        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        c_enc = tok(comp_text, add_special_tokens=False, return_offsets_mapping=True)
        c_ids, offs = c_enc["input_ids"], c_enc["offset_mapping"]
        ids = torch.tensor([p_ids + c_ids], device="cuda")
        P, L = len(p_ids), len(c_ids)

        lp_b = forward(base, ids, P, L)
        lp_r = forward(rl, ids, P, L)
        pb_top = lp_b.exp().topk(topk, dim=-1)
        pr_top = lp_r.exp().topk(topk, dim=-1)
        tgt = torch.tensor(c_ids, device="cuda")
        lpg_b = lp_b.gather(1, tgt[:, None]).squeeze(1)
        lpg_r = lp_r.gather(1, tgt[:, None]).squeeze(1)

        off = len(FENCE_OPEN)
        for k in range(L):
            gold_id = c_ids[k]
            gold_tok = tok.decode([gold_id])
            b_arg = pb_top.indices[k, 0].item()
            r_arg = pr_top.indices[k, 0].item()
            b_conf = pb_top.values[k, 0].item()
            r_conf = pr_top.values[k, 0].item()

            # calibration bins
            bi = min(int(b_conf * NB), NB - 1)
            cnt_b[bi] += 1
            bins_b[bi][0] += int(b_arg == gold_id)
            bins_b[bi][1] += b_conf
            ri = min(int(r_conf * NR), NR - 1)
            cnt_r[ri] += 1
            bins_r[ri][0] += int(r_arg == gold_id)
            bins_r[ri][1] += r_conf

            d_lp = lpg_r[k].item() - lpg_b[k].item()
            if d_lp >= -0.05 or r_arg == gold_id:
                continue   # only inspect positions where RL moved away from gold

            a, b = offs[k]
            if a < off:
                continue
            c = a - off
            while c < b - off and c < len(q) and q[c].isspace():
                c += 1
            if c >= len(q):
                continue

            r_tok = tok.decode([r_arg]).strip()
            g_strip = gold_tok.strip()
            rl_low, g_low = r_tok.lower(), g_strip.lower()

            # categorise the RL model's preference
            if alias_re.match(g_low) and (rl_low in tables or rl_low in cols):
                cat = "ALIAS->NAME"          # gold used T1, RL wants the real name
            elif alias_re.match(rl_low) and (g_low in tables or g_low in cols):
                cat = "NAME->ALIAS"
            elif rl_low in tables and g_low in tables:
                cat = "OTHER_TABLE"
            elif rl_low in cols and g_low in cols:
                cat = "OTHER_COLUMN"
            elif rl_low in AGG and g_low in AGG:
                cat = "OTHER_AGG"
            elif g_low.upper() == rl_low.upper():
                cat = "CASE_ONLY"
            elif not r_tok or r_tok in ("\n", " "):
                cat = "WHITESPACE"
            elif rl_low in tables or rl_low in cols:
                cat = "SCHEMA_NAME_OTHER"
            else:
                cat = "OTHER"

            pref_counts[cat] += 1
            rec = {"ex": i, "pos": k, "gold": gold_tok, "rl_pref": r_tok,
                   "base_pref": tok.decode([b_arg]), "cat": cat,
                   "d_lp_gold": d_lp, "rl_conf": r_conf, "base_conf": b_conf}
            pref_f.write(json.dumps(rec) + "\n")
            role_pref[cat][f"{g_strip}->{r_tok}"] += 1

        if i % 100 == 0:
            print(f"{i}/{len(dev)}")
    pref_f.close()

    def ece(bins, cnt):
        tot = sum(cnt) or 1
        e = 0.0
        rows = []
        for j in range(len(cnt)):
            if cnt[j] == 0:
                continue
            acc = bins[j][0] / cnt[j]
            conf = bins[j][1] / cnt[j]
            e += (cnt[j] / tot) * abs(acc - conf)
            rows.append({"bin": j, "n": cnt[j], "conf": conf, "acc": acc})
        return e, rows

    ece_b, rows_b = ece(bins_b, cnt_b)
    ece_r, rows_r = ece(bins_r, cnt_r)

    n = len(dev)
    summary = {
        "tag": tag, "n_examples": n,
        "preference_categories": dict(pref_counts.most_common()),
        "top_swaps": {c: dict(v.most_common(10)) for c, v in role_pref.items()},
        "ece": {"base": ece_b, "rl": ece_r},
        "reliability_base": rows_b, "reliability_rl": rows_r,
        "surface_per_query": {
            "gold": {k: v / n for k, v in surf_gold.items()},
            "base": {k: v / n for k, v in surf_base.items()},
            "rl": {k: v / n for k, v in surf_rl.items()},
        },
        "exec_correct": exec_ok,
        "exec_correct_but_text_differs": exec_ok_text_diff,
    }
    json.dump(summary, open(f"{DATA}/e2/{tag}_summary.json", "w"), indent=2)
    vol.commit()

    print(f"\n===== {tag}: what RL prefers where it moved away from gold =====")
    tot = sum(pref_counts.values()) or 1
    for c, v in pref_counts.most_common():
        print(f"  {c:20} {v:>7}  {v/tot*100:>5.1f}%")

    print(f"\n===== token-level ECE (argmax == gold) =====")
    print(f"  base {ece_b:.4f}   rl {ece_r:.4f}   delta {ece_r-ece_b:+.4f}")

    print(f"\n===== surface style, per query =====")
    keys = ["len_chars", "n_alias", "n_as", "n_join", "n_distinct",
            "n_agg", "n_setop", "n_subq", "qualified"]
    print(f"  {'stat':12}{'gold':>9}{'base':>9}{'rl':>9}{'rl-base':>9}")
    for k in keys:
        g = summary["surface_per_query"]["gold"][k]
        b = summary["surface_per_query"]["base"][k]
        r = summary["surface_per_query"]["rl"][k]
        print(f"  {k:12}{g:>9.2f}{b:>9.2f}{r:>9.2f}{r-b:>+9.2f}")

    print(f"\n===== executionally correct =====")
    print(f"  base {exec_ok['base']}/{n}   rl {exec_ok['rl']}/{n}")
    print(f"  correct but text != gold:  base {exec_ok_text_diff['base']}   "
          f"rl {exec_ok_text_diff['rl']}")
    print(f"\nsaved /data/e2/{tag}_summary.json")