"""
Per-question correctness for greedy prediction files, paired McNemar tests, and the
teacher-gap table that decides whether distilling the 7B into the 1.5B is worth a
GPU run. CPU only, no generation, no GPU.

Why: every result file so far stores aggregates only. Paired tests, the sets of
questions GRPO fixed or broke, and any comparison between a distilled student and
GRPO need one 0/1 per question per model. This job re-scores existing pred files
with the same test-suite evaluator call that spider_eval_modal.py and
reeval_official.py use, and records the evaluator's own verdict for each question by
wrapping evaluation.eval_exec_match. Nothing on the EX / TS path is re-implemented.

Self-checks. A tag that fails any of them is reported as FAIL and left out of
analysis.json:
  1. exactly one recorded evaluator call per dev question, in dev order (the gold
     string of call i equals gold.txt line i);
  2. the per-question mean equals the aggregate the evaluator printed in the same run;
  3. for the 4 main models, it also equals reeval/summary.json to 3 decimals
     (EX 0.591 / 0.718 / 0.768 / 0.794, TS 0.510 / 0.632 / 0.695 / 0.730);
  4. the loose per-question mean equals pass@1_loose in the model's summary json,
     and on the 80 set-op questions equals setop_exec_acc_loose;
  5. at 1.5B, loose FIXED / BROKEN equal E1's outcome_counts (155 / 28).

    bash modal/run_experiments.sh perq                     # if added to the launcher
    modal run --detach per_question.py::main               # the 4 main models
    modal run --detach per_question.py::main --tags "a,b"  # any pred_<tag>.txt on /data
    modal run per_question.py::summarize                   # rebuild from per-cell files

Outputs:
  /data/perq/<tag>_{single,suite}.json  per-cell verdicts (one file per worker)
  /data/perq/<tag>_{single,suite}.log   full evaluator stdout, never truncated
  /data/perq/<tag>.jsonl                one row per question: i, db_id, hardness,
                                        setop, ex, ts, loose
  /data/perq/checks.json                every self-check with PASS / FAIL
  /data/perq/analysis.json              accuracies, paired tests, teacher-gap table

Metric names: "ex" is the test-suite evaluator on the original dev DBs (the report's
EX), "ts" is the same evaluator on the distilled suite (TS), "loose" is the set-based
check on the original DB (the report's pass@1 and set-op EX). Note that pass@1_loose
in spider_eval_modal.py scores the GREEDY prediction, not one of the 8 samples.
"""

import contextlib
import io
import json
import math
import os
import re
import sys

import modal

app = modal.App("spider-perq")
vol = modal.Volume.from_name("spider-data", create_if_missing=True)

# Same image as reeval_official.py, so the evaluator code is the same commit family.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("nltk", "sqlparse", "numpy", "tqdm")
    .run_commands(
        "git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval /opt/test-suite",
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab')\"",
    )
)

DATA = "/data"
SPIDER = f"{DATA}/spider_data"
DB_DIR = f"{SPIDER}/database"
DEV_JSON = f"{SPIDER}/dev.json"
TS_DB = f"{DATA}/ts_db/database"
GOLD = f"{DATA}/gold.txt"
OUT = f"{DATA}/perq"
EVAL_DIR = os.environ.get("TS_EVAL_DIR", "/opt/test-suite")
N_DEV = 1034

MAIN_TAGS = ["Qwen2.5-Coder-1.5B-Instruct", "1.5b-grpo-binary-7k",
             "Qwen2.5-Coder-7B-Instruct", "7b-grpo-binary-7k"]
SHORT = {"Qwen2.5-Coder-1.5B-Instruct": "1.5b_base", "1.5b-grpo-binary-7k": "1.5b_grpo",
         "Qwen2.5-Coder-7B-Instruct": "7b_base", "7b-grpo-binary-7k": "7b_grpo"}

# (a, b): a's wins are questions a gets right and b gets wrong.
PAIRS = [("1.5b_grpo", "1.5b_base"), ("7b_grpo", "7b_base"),
         ("7b_base", "1.5b_base"), ("7b_base", "1.5b_grpo"), ("7b_grpo", "1.5b_grpo")]
# (student, rl, teacher) for the teacher-gap table.
GAPS = [("1.5b_base", "1.5b_grpo", "7b_base"), ("1.5b_base", "1.5b_grpo", "7b_grpo")]
LEVELS = ["easy", "medium", "hard", "extra"]
TOL = 5e-4      # the evaluator prints 3 decimals


# ---------------------------------------------------------------- scoring (pure)

# Copied from reeval_official.py (identical to spider_eval_modal.py and
# e1_token_analysis.py up to the return type), so "loose" is scored the same way.
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


# Copied from reeval_official.py: parse evaluator stdout by section, never by tail.
SECTIONS = {"EXECUTION ACCURACY": "exec", "EXACT MATCHING ACCURACY": "exact",
            "PARTIAL MATCHING ACCURACY": "partial_acc",
            "PARTIAL MATCHING RECALL": "partial_recall",
            "PARTIAL MATCHING F1": "partial_f1"}
_LEVELS5 = ["easy", "medium", "hard", "extra", "all"]


def parse_table(out):
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
                res[(section, m.group(1).strip())] = dict(zip(_LEVELS5, vals))
    return res


def load_evaluator(eval_dir=EVAL_DIR):
    """Import test-suite-sql-eval's evaluation module (it imports its siblings)."""
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    import evaluation
    return evaluation


def read_gold(path):
    """gold.txt as the evaluator reads it: (g_str, db_id) per non-blank line."""
    rows = [l.strip().split("\t") for l in open(path).read().splitlines()]
    assert all(len(r) == 2 for r in rows), "gold.txt must be '<sql>\\t<db_id>' per line"
    return rows


def score_cell(ev, gold_p, pred_p, db_dir, eval_dir=EVAL_DIR):
    """Run ev.evaluate(etype='exec') exactly as the CLI does and record the verdict of
    every eval_exec_match call. Returns (calls, printed_all, stdout)."""
    preds = open(pred_p).read().splitlines()
    # A blank pred line would split the evaluator's single session in two.
    assert all(p.strip() for p in preds), "blank line in pred file"
    calls, orig = [], ev.eval_exec_match

    def recorder(db, p_str, g_str, plug_value, keep_distinct,
                 progress_bar_for_each_datapoint):
        s = orig(db=db, p_str=p_str, g_str=g_str, plug_value=plug_value,
                 keep_distinct=keep_distinct,
                 progress_bar_for_each_datapoint=progress_bar_for_each_datapoint)
        calls.append({"db_id": os.path.basename(os.path.dirname(db)),
                      "g": g_str, "ok": int(bool(s))})
        return s

    buf = io.StringIO()
    ev.eval_exec_match = recorder
    try:
        # Same defaults as the CLI: no plug_value, no keep_distinct, no progress bar.
        with contextlib.redirect_stdout(buf), contextlib.chdir(eval_dir):
            ev.evaluate(gold_p, pred_p, db_dir, "exec", None, False, False, False)
    finally:
        ev.eval_exec_match = orig
    out = buf.getvalue()
    printed = parse_table(out).get(("exec", "execution"), {}).get("all")
    return calls, printed, out


def check_alignment(calls, gold):
    assert len(calls) == len(gold), f"{len(calls)} evaluator calls for {len(gold)} questions"
    for i, (c, (g, db_id)) in enumerate(zip(calls, gold)):
        assert c["g"] == g and c["db_id"] == db_id, f"misaligned at question {i}"


def meta_and_loose(ev, gold, preds, db_dir):
    """Hardness (the evaluator's own rule), set-op flag and loose verdict per question."""
    hardness, setop, loose = [], [], []
    for i, ((g, db_id), p) in enumerate(zip(gold, preds)):
        db = os.path.join(db_dir, db_id, db_id + ".sqlite")
        schema = ev.Schema(ev.get_schema(db))
        hardness.append(ev.Evaluator().eval_hardness(ev.get_sql(schema, g)))
        setop.append(int(bool(re.search(r"\b(intersect|union|except)\b", g.lower()))))
        loose.append(int(bool(exec_match(db, p, g))))
    return hardness, setop, loose


# ---------------------------------------------------------------- analysis (pure)

def mcnemar_p(b, c):
    """Exact two-sided McNemar: binomial test on the b + c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def paired(a, b):
    """a, b: 0/1 lists. Wins = a right and b wrong."""
    w = sum(1 for x, y in zip(a, b) if x and not y)
    l = sum(1 for x, y in zip(a, b) if y and not x)
    n = len(a)
    return {"wins": w, "losses": l, "delta_pts": 100 * (w - l) / n, "p": mcnemar_p(w, l)}


def teacher_gap(S, R, T, hardness, setop):
    """S = student base, R = student after GRPO, T = teacher. 0/1 lists."""
    idx = range(len(S))
    fixed = [i for i in idx if not S[i] and R[i]]
    broken = [i for i in idx if S[i] and not R[i]]
    t_not_r = [i for i in idx if T[i] and not R[i]]
    by_level = {lv: sum(1 for i in t_not_r if hardness[i] == lv) for lv in LEVELS}
    so = [i for i in idx if setop[i]]
    return {
        "n": len(S),
        "fixed": len(fixed),
        "broken": len(broken),
        "fixed_teacher_right": sum(T[i] for i in fixed),
        "broken_teacher_right": sum(T[i] for i in broken),
        "teacher_right_student_wrong": sum(1 for i in idx if T[i] and not S[i]),
        "teacher_right_rl_wrong": len(t_not_r),
        "rl_right_teacher_wrong": sum(1 for i in idx if R[i] and not T[i]),
        "teacher_only": sum(1 for i in idx if T[i] and not S[i] and not R[i]),
        "all_wrong": sum(1 for i in idx if not (S[i] or R[i] or T[i])),
        "oracle_rl_or_teacher": sum(1 for i in idx if R[i] or T[i]) / len(S),
        "teacher_right_rl_wrong_by_level": by_level,
        "setop": {"n": len(so),
                  "fixed": sum(1 for i in so if not S[i] and R[i]),
                  "fixed_teacher_right": sum(1 for i in so if not S[i] and R[i] and T[i]),
                  "teacher_right_rl_wrong": sum(1 for i in so if T[i] and not R[i])},
        "ids_teacher_right_rl_wrong": t_not_r,
    }


def analyze(vecs):
    """vecs: {short_tag: {"ex": [...], "ts": [...], "loose": [...],
    "hardness": [...], "setop": [...]}} for every tag that passed its checks."""
    res = {"acc": {t: {m: sum(v[m]) / len(v[m]) for m in ("ex", "ts", "loose")}
                   for t, v in vecs.items()},
           "paired": {}, "teacher_gap": {}}
    for a, b in PAIRS:
        if a in vecs and b in vecs:
            res["paired"][f"{a}_vs_{b}"] = {m: paired(vecs[a][m], vecs[b][m])
                                            for m in ("ex", "ts", "loose")}
    for s, r, t in GAPS:
        if all(x in vecs for x in (s, r, t)):
            res["teacher_gap"][f"{t}_over_{r}"] = {
                m: teacher_gap(vecs[s][m], vecs[r][m], vecs[t][m],
                               vecs[s]["hardness"], vecs[s]["setop"])
                for m in ("ex", "ts", "loose")}
    return res


def run_checks(tag, single, suite, ref_reeval, ref_summary, ref_e1, base_loose=None):
    """Return [(name, ok, detail)] for one tag. ref_* may be None for new tags."""
    n = len(single["ok"])
    ex, ts = sum(single["ok"]) / n, sum(suite["ok"]) / n
    lo = sum(single["loose"]) / n
    out = [("ex == evaluator printed", abs(ex - single["printed"]) < TOL, (ex, single["printed"])),
           ("ts == evaluator printed", abs(ts - suite["printed"]) < TOL, (ts, suite["printed"]))]
    if ref_reeval:
        out += [("ex == reeval", abs(round(ex, 3) - ref_reeval["EX_testsuite_single"]) < 1e-9,
                 (ex, ref_reeval["EX_testsuite_single"])),
                ("ts == reeval", abs(round(ts, 3) - ref_reeval["TS_suite"]) < 1e-9,
                 (ts, ref_reeval["TS_suite"]))]
        so = [i for i in range(n) if single["setop"][i]]
        so_acc = sum(single["loose"][i] for i in so) / len(so)
        out.append(("setop loose == reeval", abs(so_acc - ref_reeval["setop_exec_acc_loose"]) < 1e-9,
                    (so_acc, ref_reeval["setop_exec_acc_loose"])))
    if ref_summary and "pass@1_loose" in ref_summary:
        out.append(("loose == pass@1_loose", abs(lo - ref_summary["pass@1_loose"]) < 1e-9,
                    (lo, ref_summary["pass@1_loose"])))
    if ref_e1 and base_loose is not None:
        b, r = base_loose, single["loose"]
        got = {"FIXED": sum(1 for x, y in zip(b, r) if not x and y),
               "BROKEN": sum(1 for x, y in zip(b, r) if x and not y)}
        want = {k: ref_e1["outcome_counts"][k] for k in got}
        out.append(("loose fixed/broken == E1", got == want, (got, want)))
    return out


# ---------------------------------------------------------------- Modal

@app.function(image=image, volumes={DATA: vol}, cpu=2, timeout=3 * 3600)
def cell(tag: str, which: str):
    ev = load_evaluator()
    pred_p = f"{DATA}/pred_{tag}.txt"
    gold = read_gold(GOLD)
    preds = open(pred_p).read().splitlines()
    assert len(preds) == len(gold) == N_DEV, (len(preds), len(gold))
    dev = json.load(open(DEV_JSON))
    assert [(d["query"].strip(), d["db_id"]) for d in dev] == [tuple(g) for g in gold], \
        "gold.txt is not dev.json in order"
    db_dir = DB_DIR if which == "single" else TS_DB
    calls, printed, out = score_cell(ev, GOLD, pred_p, db_dir)
    check_alignment(calls, gold)
    res = {"tag": tag, "which": which, "ok": [c["ok"] for c in calls], "printed": printed}
    if which == "single":
        res["hardness"], res["setop"], res["loose"] = meta_and_loose(ev, gold, preds, DB_DIR)
        res["db_id"] = [g[1] for g in gold]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{tag}_{which}.json", "w"))
    open(f"{OUT}/{tag}_{which}.log", "w").write(out)
    vol.commit()
    return {"tag": tag, "which": which, "acc": sum(res["ok"]) / len(res["ok"]),
            "printed": printed}


@app.function(image=image, volumes={DATA: vol}, timeout=4 * 3600)
def main(tags: str = ""):
    tags = [t for t in tags.split(",") if t] or MAIN_TAGS
    jobs = [(t, w) for t in tags for w in ("single", "suite")]
    for r in cell.starmap(jobs, return_exceptions=True):
        print(r)
    build(tags)


@app.function(image=image, volumes={DATA: vol})
def summarize(tags: str = ""):
    build([t for t in tags.split(",") if t] or MAIN_TAGS)


def _load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def build(tags):
    vol.reload()                         # see the files the workers committed
    reeval = {r["tag"]: r for r in (_load(f"{DATA}/reeval/summary.json") or [])}
    e1 = _load(f"{DATA}/e1/1.5b_summary.json")
    base_single = _load(f"{OUT}/Qwen2.5-Coder-1.5B-Instruct_single.json")
    checks, vecs = {}, {}
    for tag in tags:
        single, suite = _load(f"{OUT}/{tag}_single.json"), _load(f"{OUT}/{tag}_suite.json")
        if not (single and suite):
            checks[tag] = [("cells present", False, (bool(single), bool(suite)))]
            continue
        cs = run_checks(tag, single, suite, reeval.get(tag),
                        _load(f"{DATA}/summary_{tag}.json"),
                        e1 if tag == "1.5b-grpo-binary-7k" else None,
                        base_single["loose"] if base_single else None)
        checks[tag] = cs
        if not all(ok for _, ok, _ in cs):
            continue
        rows = [{"i": i, "db_id": single["db_id"][i], "hardness": single["hardness"][i],
                 "setop": single["setop"][i], "ex": single["ok"][i], "ts": suite["ok"][i],
                 "loose": single["loose"][i]} for i in range(len(single["ok"]))]
        with open(f"{OUT}/{tag}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        vecs[SHORT.get(tag, tag)] = {"ex": single["ok"], "ts": suite["ok"],
                                     "loose": single["loose"],
                                     "hardness": single["hardness"], "setop": single["setop"]}
    json.dump({t: [{"check": n, "ok": ok, "detail": d} for n, ok, d in cs]
               for t, cs in checks.items()}, open(f"{OUT}/checks.json", "w"), indent=2)
    analysis = analyze(vecs)
    analysis["tags_passed"] = sorted(vecs)
    json.dump(analysis, open(f"{OUT}/analysis.json", "w"), indent=2)
    vol.commit()
    report(checks, analysis)


def report(checks, a):
    for tag, cs in checks.items():
        for name, ok, d in cs:
            print(f"{'PASS' if ok else 'FAIL'}  {tag:30} {name:28} {d}")
    print("\naccuracy (ex / ts / loose)")
    for t, v in a["acc"].items():
        print(f"  {t:12} {v['ex']:.3f} / {v['ts']:.3f} / {v['loose']:.3f}")
    print("\npaired, ex: wins / losses, delta pts, McNemar p")
    for k, v in a["paired"].items():
        e = v["ex"]
        print(f"  {k:24} {e['wins']:4} / {e['losses']:<4} {e['delta_pts']:+6.1f}  p={e['p']:.2g}")
    for k, v in a["teacher_gap"].items():
        g = v["ex"]
        print(f"\nteacher gap, ex, {k}:")
        for f in ("fixed", "fixed_teacher_right", "broken", "broken_teacher_right",
                  "teacher_right_rl_wrong", "rl_right_teacher_wrong", "teacher_only",
                  "all_wrong", "oracle_rl_or_teacher"):
            print(f"  {f:26} {g[f]}")
        print(f"  {'by level (t right, rl wrong)':26} {g['teacher_right_rl_wrong_by_level']}")
        print(f"  {'set-op':26} {g['setop']}")
