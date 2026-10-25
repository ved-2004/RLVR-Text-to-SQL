"""
E3: Is the style change causal, and can the gain be had without training?

Runs the same Spider dev harness as spider_eval_modal.py, but varies two things:

  style   default   original prompt (byte-identical to every earlier eval)
          no_alias  + "do not use table aliases, write table.column"
          alias     + "alias every table as T1, T2 and write T1.column"

  decode  greedy    temperature 0 (what EX normally measures)
          sc8       sample 8, execute each, majority-vote on result sets.
                    Uses no gold answer, so it is a legitimate inference-time method.

For every (style, decode) cell it reports official EX, exact match, IUEN,
test-suite accuracy (single DB and suite DBs), and the alias / qualified-name
rate of the outputs, which is the check that the style instruction actually worked.

Commands:
    modal run --detach e3_style_and_sampling.py::sweep      # 2 models x 3 styles, 6 GPUs in parallel
    modal run e3_style_and_sampling.py::summarize           # rebuild the table from rows/

Outputs on the volume:
    /data/e3/pred_<tag>_<style>_<decode>.txt
    /data/e3/rows/<tag>_<style>_<decode>.json   one file per cell
    /data/e3/summary.json                       all 12 cells
"""

import modal

app = modal.App("spider-e3")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

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

# With style="" this is byte-identical to the prompt used in every earlier run.
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

FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


# ---------------------------------------------------------------- verifier (unchanged)
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


# ---------------------------------------------------------------- helpers
def run_eval(cwd, gold_p, pred_p, db, etype):
    import subprocess
    r = subprocess.run(
        ["python", "evaluation.py", "--gold", gold_p, "--pred", pred_p,
         "--db", db, "--table", TABLES, "--etype", etype],
        cwd=cwd, capture_output=True, text=True)
    return r.stdout


def parse_row(out, label):
    """Return the 'all' column of the first row starting with `label`."""
    for line in out.splitlines():
        if line.strip().startswith(label):
            try:
                return float(line.split()[-1])
            except ValueError:
                return None
    return None


def style_stats(preds):
    import re
    n = max(len(preds), 1)
    alias = sum(len(re.findall(r"\bt\d+\b", p.lower())) for p in preds) / n
    qual = sum(len(re.findall(r"\b[a-z_]+\.[a-z_]+", p.lower())) for p in preds) / n
    return alias, qual


# ---------------------------------------------------------------- main
@app.function(image=image, gpu="A100-40GB", volumes={DATA: vol}, timeout=10800)
def run(model_id: str, tag: str, styles: str = "default,no_alias,alias", k: int = 8):
    import json, os
    from collections import Counter
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_id)
    dev = json.load(open(DEV_JSON))
    llm = LLM(model=model_id, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096)

    os.makedirs(f"{DATA}/e3", exist_ok=True)
    gold_p = f"{DATA}/e3/gold_{tag}_{styles.replace(',', '-')}.txt"   # unique per container
    with open(gold_p, "w") as gf:
        for ex in dev:
            gf.write(f"{ex['query']}\t{ex['db_id']}\n")

    schema_cache, results = {}, []

    for style in styles.split(","):
        style = style.strip()
        prompts, dbps = [], []
        for ex in dev:
            dbp = f"{DB_DIR}/{ex['db_id']}/{ex['db_id']}.sqlite"
            if dbp not in schema_cache:
                schema_cache[dbp] = build_schema_string(dbp)
            msg = [{"role": "user", "content": PROMPT_TMPL.format(
                schema=schema_cache[dbp], style=STYLE[style],
                question=ex["question"])}]
            prompts.append(tok.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True))
            dbps.append(dbp)

        greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=256))
        g_preds = [extract_sql(o.outputs[0].text) or "SELECT 1" for o in greedy]

        sampled = llm.generate(prompts, SamplingParams(
            temperature=0.8, top_p=0.95, n=k, max_tokens=256))
        sc_preds = []
        for i, out in enumerate(sampled):
            votes, first = Counter(), {}
            for o in out.outputs:
                sql = extract_sql(o.text)
                res = safe_execute(dbps[i], sql)
                if res is None:
                    continue
                key = frozenset(map(tuple, res))
                votes[key] += 1
                first.setdefault(key, sql)
            sc_preds.append(first[votes.most_common(1)[0][0]] if votes else g_preds[i])

        for decode, preds in (("greedy", g_preds), (f"sc{k}", sc_preds)):
            name = f"{tag}_{style}_{decode}"
            pred_p = f"{DATA}/e3/pred_{name}.txt"
            with open(pred_p, "w") as pf:
                pf.write("\n".join(preds) + "\n")

            ex_out = run_eval("/opt/spider", gold_p, pred_p, DB_DIR, "all")
            ts1 = run_eval("/opt/test-suite", gold_p, pred_p, DB_DIR, "exec")
            tsn = run_eval("/opt/test-suite", gold_p, pred_p, TS_DB, "exec")
            alias, qual = style_stats(preds)

            row = {
                "tag": tag, "style": style, "decode": decode,
                "EX": parse_row(ex_out, "execution"),
                "EM": parse_row(ex_out, "exact match"),
                "IUEN": parse_row(ex_out, "IUEN"),
                "TS_single": parse_row(ts1, "execution"),
                "TS_suite": parse_row(tsn, "execution"),
                "alias_per_q": alias,
                "qualified_per_q": qual,
            }
            results.append(row)
            os.makedirs(f"{DATA}/e3/rows", exist_ok=True)
            json.dump(row, open(f"{DATA}/e3/rows/{name}.json", "w"), indent=2)
            print(f"done {name}: {row}")
            vol.commit()

    print_table(results, tag)
    return results


def print_table(results, title):
    print(f"\n===== {title} =====")
    print(f"{'tag':12}{'style':10}{'decode':8}{'EX':>7}{'EM':>7}{'IUEN':>7}"
          f"{'TS1':>7}{'TSn':>7}{'alias':>7}{'qual':>7}")
    for r in results:
        f = lambda v: f"{v:>7.3f}" if isinstance(v, float) else f"{'n/a':>7}"
        print(f"{r['tag']:12}{r['style']:10}{r['decode']:8}{f(r['EX'])}{f(r['EM'])}{f(r['IUEN'])}"
              f"{f(r['TS_single'])}{f(r['TS_suite'])}"
              f"{r['alias_per_q']:>7.2f}{r['qualified_per_q']:>7.2f}")


MODELS = {"1.5b-base": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
          "1.5b-rl": "/data/merged/1.5b-grpo-binary-7k"}


@app.function(image=image, volumes={DATA: vol}, timeout=4 * 3600)
def sweep(k: int = 8):
    """One container per (model, style): 6 in parallel. Each writes its own
    files, so nothing is overwritten. Then one combined summary."""
    import json
    jobs = [(mid, tag, style, k) for tag, mid in MODELS.items()
            for style in ("default", "no_alias", "alias")]
    rows = []
    for out in run.starmap(jobs, return_exceptions=True):
        if isinstance(out, list):
            rows.extend(out)
        else:
            print(f"job failed: {out}")
    import os
    vol.reload()                                   # see dirs/files the workers committed
    os.makedirs(f"{DATA}/e3", exist_ok=True)
    json.dump(rows, open(f"{DATA}/e3/summary.json", "w"), indent=2)
    vol.commit()
    print_table(rows, "E3 all")
    print(f"\n{len(rows)}/12 cells done. saved /data/e3/summary.json")


@app.function(image=image, volumes={DATA: vol})
def summarize():
    """Rebuild the table from per-cell files, e.g. if sweep was interrupted."""
    import json, glob
    vol.reload()
    rows = [json.load(open(q)) for q in sorted(glob.glob(f"{DATA}/e3/rows/*.json"))]
    json.dump(rows, open(f"{DATA}/e3/summary.json", "w"), indent=2)
    vol.commit()
    print_table(rows, "E3 from rows/")
    print(f"\n{len(rows)}/12 cells found. saved /data/e3/summary.json")
