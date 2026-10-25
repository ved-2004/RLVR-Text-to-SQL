"""
Re-score the four existing prediction files. CPU only, no generation.

Why: spider_eval_modal.py saved only the last 2,500 characters of the official
Spider evaluator's output (ex_raw). The evaluator prints every exact-match failure
before its score table, so that window held only the tail of the table. The
numbers we called "EX" (0.521, 0.580, ...) were the SELECT row of PARTIAL MATCHING
RECALL, and "IUEN" was the IUEN recall row. The test-suite numbers were saved in
full and are fine.

This job re-runs, on the exact same pred files:
  - official Spider evaluator (etype all): execution, exact match, IUEN rows
  - test-suite evaluator on the original dev DBs (single) and on the suite DBs
  - execution accuracy on the set-operation subset (gold has INTERSECT/UNION/EXCEPT)
and saves the FULL evaluator output, parsed section by section.

    modal run --detach reeval_official.py::main

Outputs: /data/reeval/<tag>_{spider_all,ts_single,ts_suite}.txt, /data/reeval/summary.json
"""

import modal

app = modal.App("spider-reeval")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("nltk", "sqlparse", "numpy")
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
GOLD = f"{DATA}/gold.txt"

TAGS = ["Qwen2.5-Coder-1.5B-Instruct", "1.5b-grpo-binary-7k",
        "Qwen2.5-Coder-7B-Instruct", "7b-grpo-binary-7k"]

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


FORBIDDEN = ("drop", "delete", "update", "insert", "alter", "create",
             "replace", "attach", "pragma")


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


@app.function(image=image, volumes={DATA: vol}, cpu=4, timeout=3 * 3600)
def one(tag: str):
    import json, os, re, subprocess
    pred_p = f"{DATA}/pred_{tag}.txt"
    dev = json.load(open(DEV_JSON))
    preds = open(pred_p).read().splitlines()
    gold_lines = open(GOLD).read().splitlines()
    assert len(preds) == len(dev) == len(gold_lines) == 1034, (len(preds), len(gold_lines))

    def run(cwd, db, etype):
        r = subprocess.run(["python", "evaluation.py", "--gold", GOLD, "--pred", pred_p,
                            "--db", db, "--table", TABLES, "--etype", etype],
                           cwd=cwd, capture_output=True, text=True)
        return r.stdout + ("\n[stderr]\n" + r.stderr if r.returncode else "")

    os.makedirs(f"{DATA}/reeval", exist_ok=True)
    outs = {"spider_all": run("/opt/spider", DB_DIR, "all"),
            "ts_single": run("/opt/test-suite", DB_DIR, "exec"),
            "ts_suite": run("/opt/test-suite", TS_DB, "exec")}
    for k, v in outs.items():
        open(f"{DATA}/reeval/{tag}_{k}.txt", "w").write(v)
    T = {k: parse_table(v) for k, v in outs.items()}

    setop = [i for i, ex in enumerate(dev)
             if re.search(r"\b(intersect|union|except)\b", ex["query"].lower())]
    db = lambda i: f"{DB_DIR}/{dev[i]['db_id']}/{dev[i]['db_id']}.sqlite"
    setop_acc = sum(exec_match(db(i), preds[i], dev[i]["query"]) for i in setop) / len(setop)

    row = {
        "tag": tag,
        "EX_official_spider": get(T["spider_all"], "exec", "execution"),
        "EM_official_spider": get(T["spider_all"], "exact", "exact match"),
        "EX_testsuite_single": get(T["ts_single"], "exec", "execution"),
        "TS_suite": get(T["ts_suite"], "exec", "execution"),
        "IUEN_partial_acc": get(T["spider_all"], "partial_acc", "IUEN"),
        "IUEN_partial_recall": get(T["spider_all"], "partial_recall", "IUEN"),
        "select_partial_recall": get(T["spider_all"], "partial_recall", "select"),
        "setop_n": len(setop),
        "setop_exec_acc_loose": setop_acc,
        "by_level_EX_official": T["spider_all"].get(("exec", "execution")),
        "by_level_TS_single": T["ts_single"].get(("exec", "execution")),
    }
    json.dump(row, open(f"{DATA}/reeval/{tag}.json", "w"), indent=2)
    vol.commit()
    return row


@app.function(image=image, volumes={DATA: vol}, timeout=4 * 3600)
def main():
    import json
    rows = [r for r in one.map(TAGS, return_exceptions=True) if isinstance(r, dict)]
    json.dump(rows, open(f"{DATA}/reeval/summary.json", "w"), indent=2)
    vol.commit()
    cols = ["EX_official_spider", "EX_testsuite_single", "TS_suite", "EM_official_spider",
            "select_partial_recall", "IUEN_partial_recall", "setop_exec_acc_loose"]
    print(f"\n{'model':30}" + "".join(f"{c[:12]:>13}" for c in cols))
    for r in rows:
        print(f"{r['tag']:30}" + "".join(
            f"{r[c]:>13.3f}" if isinstance(r[c], float) else f"{'n/a':>13}" for c in cols))
    print(f"\nset-op subset size: {rows[0]['setop_n'] if rows else 'n/a'}")
    print("check: select_partial_recall should equal the old 'EX' (0.521 / 0.580 / 0.587 / 0.596),")
    print("       EX_testsuite_single should equal the old 'TS single' (0.591 / 0.718 / 0.768 / 0.794).")
