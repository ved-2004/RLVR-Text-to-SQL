"""
E1 — Where does RLVR sharpen?  (v2: fixed BPE leading-space tagging bug)

Teacher-forces the gold SQL through the base model and the RL model, records
per-token divergence between them, tags every token with its SQL role
(TABLE / JOIN / COLUMN / OP / VALUE / ...), and aggregates by role and by
outcome (FIXED / BROKEN / BOTH_RIGHT / BOTH_WRONG).

Outputs on the volume:
    /data/e1/<tag>_tokens.jsonl   one record per completion token
    /data/e1/<tag>_summary.json   per-role and per-outcome aggregates
"""

import modal

app = modal.App("spider-e1")
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


# ---------------------------------------------------------------- verifier
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


# ---------------------------------------------------------------- SQL role tagger
CLAUSE = {
    "select": "SELECT", "from": "FROM", "where": "WHERE",
    "group": "GROUPBY", "order": "ORDERBY", "by": "BY",
    "having": "HAVING", "limit": "LIMIT",
    "join": "JOIN", "on": "ON", "as": "AS",
    "and": "BOOL", "or": "BOOL", "not": "BOOL",
    "distinct": "DISTINCT", "desc": "ORDERDIR", "asc": "ORDERDIR",
    "in": "OP", "like": "OP", "between": "OP", "is": "OP",
    "null": "VALUE",
    "intersect": "SETOP", "union": "SETOP", "except": "SETOP",
    "inner": "JOIN", "left": "JOIN", "right": "JOIN", "outer": "JOIN",
}
AGG = {"count", "avg", "min", "max", "sum"}


def tag_sql_roles(sql):
    """Return [(start_char, end_char, role)] over the SQL string."""
    import re, sqlparse
    from sqlparse import tokens as T
    spans, pos, expect_table = [], 0, False
    for tok in sqlparse.parse(sql)[0].flatten():
        v = tok.value
        s, e = pos, pos + len(v)
        pos = e
        if tok.is_whitespace:
            continue
        low = v.lower()
        if low in CLAUSE:
            role = CLAUSE[low]
            expect_table = low in ("from", "join", "inner", "left", "right", "outer")
        elif low in AGG:
            role, expect_table = "AGG", False
        elif tok.ttype in T.Keyword:
            role, expect_table = "KEYWORD", False
        elif tok.ttype in T.Name or tok.ttype is None:
            if "." in v:
                role = "COLUMN"
            elif re.fullmatch(r"t\d+", low):
                role = "ALIAS"
            elif expect_table:
                role = "TABLE"
            else:
                role = "COLUMN"
            expect_table = False
        elif tok.ttype in T.Literal:
            role, expect_table = "VALUE", False
        elif tok.ttype in T.Operator or tok.ttype in T.Comparison:
            role, expect_table = "OP", False
        elif tok.ttype in T.Wildcard:
            role, expect_table = "STAR", False
        elif tok.ttype in T.Punctuation:
            role = "PUNCT"
        else:
            role, expect_table = "OTHER", False
        spans.append((s, e, role))
    return spans


def role_for_char(spans, c):
    for s, e, r in spans:
        if s <= c < e:
            return r
    return "WS"


# ---------------------------------------------------------------- analysis
@app.function(image=image, gpu="A100-80GB", volumes={DATA: vol}, timeout=7200)
def analyze(base_id: str, rl_path: str, tag: str,
            pred_base: str = "", pred_rl: str = "", limit: int = 0):
    import json, os
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collections import defaultdict

    tok = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    rl = AutoModelForCausalLM.from_pretrained(
        rl_path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    dev = json.load(open(DEV_JSON))
    if limit:
        dev = dev[:limit]

    outcomes = [None] * len(dev)
    if pred_base and pred_rl:
        pb = open(pred_base).read().splitlines()
        pr = open(pred_rl).read().splitlines()
        for i, ex in enumerate(dev):
            dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
            b = exec_match(dbp, pb[i], ex["query"])
            r = exec_match(dbp, pr[i], ex["query"])
            outcomes[i] = {(0, 1): "FIXED", (1, 0): "BROKEN",
                           (1, 1): "BOTH_RIGHT", (0, 0): "BOTH_WRONG"}[(int(b), int(r))]

    os.makedirs(f"{DATA}/e1", exist_ok=True)
    out_f = open(f"{DATA}/e1/{tag}_tokens.jsonl", "w")

    @torch.no_grad()
    def per_token(model, ids, P, L):
        logits = model(ids).logits[0, P - 1: P - 1 + L].float()
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        top2 = p.topk(2, dim=-1).values
        return logp, ent, top2[:, 0] - top2[:, 1]

    records = []
    for i, ex in enumerate(dev):
        dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
        msg = [{"role": "user", "content": PROMPT.format(
            schema=build_schema_string(dbp), question=ex["question"])}]
        prompt_text = tok.apply_chat_template(msg, tokenize=False,
                                              add_generation_prompt=True)
        q = ex["query"]
        comp_text = FENCE_OPEN + q + FENCE_CLOSE

        p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        c_enc = tok(comp_text, add_special_tokens=False, return_offsets_mapping=True)
        c_ids, offs = c_enc["input_ids"], c_enc["offset_mapping"]
        ids = torch.tensor([p_ids + c_ids], device="cuda")
        P, L = len(p_ids), len(c_ids)

        lp_b, ent_b, mg_b = per_token(base, ids, P, L)
        lp_r, ent_r, mg_r = per_token(rl, ids, P, L)
        kl = (lp_r.exp() * (lp_r - lp_b)).sum(-1)
        tgt = torch.tensor(c_ids, device="cuda")
        lpg_b = lp_b.gather(1, tgt[:, None]).squeeze(1)
        lpg_r = lp_r.gather(1, tgt[:, None]).squeeze(1)

        spans = tag_sql_roles(q)
        off = len(FENCE_OPEN)

        if i == 0:
            print("\n--- TAGGER CHECK (example 0) ---")
            print("gold:", q)

        for k in range(L):
            a, b = offs[k]
            if a < off:
                role = "FORMAT"
            else:
                c, end = a - off, b - off
                # BPE attaches the leading space to the token; skip it so
                # ' FROM' tags as FROM rather than as whitespace.
                while c < end and c < len(q) and q[c].isspace():
                    c += 1
                if c >= len(q):
                    role = "FORMAT"
                elif c >= end:
                    role = "WS"
                else:
                    role = role_for_char(spans, c)

            if i == 0:
                print(f"  {role:9} {tok.decode([c_ids[k]])!r}")

            rec = {
                "ex": i, "pos": k, "role": role, "outcome": outcomes[i],
                "tok": tok.decode([c_ids[k]]),
                "kl": kl[k].item(),
                "ent_base": ent_b[k].item(), "ent_rl": ent_r[k].item(),
                "margin_base": mg_b[k].item(), "margin_rl": mg_r[k].item(),
                "lp_gold_base": lpg_b[k].item(), "lp_gold_rl": lpg_r[k].item(),
            }
            records.append(rec)
            out_f.write(json.dumps(rec) + "\n")

        if i % 100 == 0:
            print(f"{i}/{len(dev)}")
    out_f.close()

    def agg(recs):
        d = defaultdict(list)
        for r in recs:
            d[r["role"]].append(r)
        total_kl = sum(r["kl"] for r in recs) or 1e-9
        n_total = max(len(recs), 1)
        rows = {}
        for role, rs in d.items():
            n = len(rs)
            row = {
                "n": n,
                "token_share": n / n_total,
                "kl_share": sum(r["kl"] for r in rs) / total_kl,
                "mean_kl": sum(r["kl"] for r in rs) / n,
                "d_entropy": sum(r["ent_rl"] - r["ent_base"] for r in rs) / n,
                "d_margin": sum(r["margin_rl"] - r["margin_base"] for r in rs) / n,
                "d_lp_gold": sum(r["lp_gold_rl"] - r["lp_gold_base"] for r in rs) / n,
            }
            row["concentration"] = row["kl_share"] / max(row["token_share"], 1e-9)
            rows[role] = row
        return rows

    summary = {"tag": tag, "base": base_id, "rl": rl_path,
               "n_examples": len(dev),
               "overall_mean_kl": sum(r["kl"] for r in records) / max(len(records), 1),
               "all": agg(records)}
    if any(outcomes):
        summary["by_outcome"] = {
            oc: agg([r for r in records if r["outcome"] == oc])
            for oc in ("FIXED", "BROKEN", "BOTH_RIGHT", "BOTH_WRONG")
            if any(r["outcome"] == oc for r in records)}
        summary["outcome_counts"] = {
            oc: sum(1 for o in outcomes if o == oc)
            for oc in ("FIXED", "BROKEN", "BOTH_RIGHT", "BOTH_WRONG")}

    json.dump(summary, open(f"{DATA}/e1/{tag}_summary.json", "w"), indent=2)
    vol.commit()

    def show(rows, title):
        print(f"\n===== {title} =====")
        print(f"{'role':10} {'n':>6} {'tok%':>6} {'KL%':>6} {'conc':>6} "
              f"{'meanKL':>8} {'dEnt':>8} {'dMargin':>8} {'dLPgold':>8}")
        for role, r in sorted(rows.items(), key=lambda x: -x[1]["kl_share"]):
            print(f"{role:10} {r['n']:>6} {r['token_share']*100:>6.1f} "
                  f"{r['kl_share']*100:>6.1f} {r['concentration']:>6.2f} "
                  f"{r['mean_kl']:>8.4f} {r['d_entropy']:>8.4f} "
                  f"{r['d_margin']:>8.4f} {r['d_lp_gold']:>8.4f}")

    print(f"\noverall mean KL: {summary['overall_mean_kl']:.5f}")
    show(summary["all"], f"{tag} — ALL examples")
    if "by_outcome" in summary:
        print("\noutcome counts:", summary["outcome_counts"])
        for oc, rows in summary["by_outcome"].items():
            show(rows, f"{tag} — {oc}")
    print(f"\nsaved /data/e1/{tag}_summary.json")