"""
CPU test for per_question.py. No GPU, no Modal account, no Spider download.

Needs a local clone of test-suite-sql-eval and NLTK punkt / punkt_tab:
    git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval /tmp/test-suite
    TS_EVAL_DIR=/tmp/test-suite python modal/test_per_question_cpu.py

It builds two toy Spider-style databases plus a two-file test suite, with questions
chosen so EX, TS and the loose check disagree on purpose, runs the real evaluator
through the same Modal cell function (via .local()), and checks every per-question
verdict, the self-checks, McNemar against the handoff's p-values, and the
teacher-gap counts.
"""

import json, os, shutil, sqlite3, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import per_question as P

EVAL_DIR = os.environ.get("TS_EVAL_DIR", "/tmp/test-suite")
assert os.path.exists(f"{EVAL_DIR}/evaluation.py"), f"clone test-suite-sql-eval to {EVAL_DIR}"
P.EVAL_DIR = EVAL_DIR
P.load_evaluator.__defaults__ = (EVAL_DIR,)
P.score_cell.__defaults__ = (EVAL_DIR,)


def make_db(path, singers, pets):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    if singers is not None:
        con.execute("CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, country TEXT)")
        con.executemany("INSERT INTO singer VALUES (?,?,?,?)", singers)
    if pets is not None:
        con.execute("CREATE TABLE pet (pet_id INTEGER PRIMARY KEY, kind TEXT, weight REAL)")
        con.executemany("INSERT INTO pet VALUES (?,?,?)", pets)
    con.commit()
    con.close()


root = tempfile.mkdtemp()
data = f"{root}/data"
singers = [(1, "Ann", 25, "US"), (2, "Bo", 34, "US"), (3, "Cy", 41, "UK"), (4, "Di", 29, "FR")]
singers_perturbed = [(1, "Ann", 25, "US"), (2, "Bo", 31, "US"), (3, "Cy", 41, "UK"), (4, "Di", 29, "FR")]
pets = [(1, "dog", 12.0), (2, "cat", 4.0), (3, "dog", 15.0), (4, "bird", 1.0), (5, "cat", 11.0)]
make_db(f"{data}/spider_data/database/concert/concert.sqlite", singers, None)
make_db(f"{data}/spider_data/database/pets/pets.sqlite", None, pets)
make_db(f"{data}/ts_db/database/concert/concert.sqlite", singers, None)
make_db(f"{data}/ts_db/database/concert/concert_1.sqlite", singers_perturbed, None)
make_db(f"{data}/ts_db/database/pets/pets.sqlite", None, pets)

# (gold, db_id, pred, expected ex, expected ts, expected loose, set-op)
CASES = [
    ("SELECT count(*) FROM singer", "concert",
     "SELECT count(*) FROM singer", 1, 1, 1, 0),
    # spurious: right on the original DB (no age 31), wrong on the perturbed suite DB
    ("SELECT name FROM singer WHERE age > 30", "concert",
     "SELECT name FROM singer WHERE age > 31", 1, 0, 1, 0),
    # order matters when gold has ORDER BY; loose (set) check ignores order
    ("SELECT name FROM singer ORDER BY age DESC", "concert",
     "SELECT name FROM singer ORDER BY age ASC", 0, 0, 1, 0),
    # set-op, alias style differs from gold, same result
    ("SELECT kind FROM pet WHERE weight > 10 INTERSECT SELECT kind FROM pet WHERE weight < 20", "pets",
     "SELECT T1.kind FROM pet AS T1 WHERE T1.weight > 10 INTERSECT SELECT kind FROM pet WHERE weight < 20",
     1, 1, 1, 1),
    # invalid SQL: the evaluator's parser fails, execution fails
    ("SELECT name FROM singer WHERE country = 'US'", "concert",
     "SELEC name FROM singer", 0, 0, 0, 0),
    # bag vs set: duplicates matter to the evaluator, not to the loose check
    ("SELECT country FROM singer", "concert",
     "SELECT country FROM singer GROUP BY country", 0, 0, 1, 0),
]
N = len(CASES)
open(f"{data}/gold.txt", "w").write("".join(f"{g}\t{d}\n" for g, d, *_ in CASES))
json.dump([{"query": g, "db_id": d, "question": "q"} for g, d, *_ in CASES],
          open(f"{data}/spider_data/dev.json", "w"))

P.DATA, P.N_DEV = data, N
P.SPIDER, P.DB_DIR = f"{data}/spider_data", f"{data}/spider_data/database"
P.DEV_JSON, P.TS_DB = f"{data}/spider_data/dev.json", f"{data}/ts_db/database"
P.GOLD, P.OUT = f"{data}/gold.txt", f"{data}/perq"


class FakeVol:
    def reload(self): pass
    def commit(self): pass


P.vol = FakeVol()

# 1. one cell end to end through the real evaluator
tag = "Qwen2.5-Coder-1.5B-Instruct"
open(f"{data}/pred_{tag}.txt", "w").write("".join(c[2] + "\n" for c in CASES))
r_single = P.cell.local(tag, "single")
r_suite = P.cell.local(tag, "suite")
single = json.load(open(f"{data}/perq/{tag}_single.json"))
suite = json.load(open(f"{data}/perq/{tag}_suite.json"))
assert single["ok"] == [c[3] for c in CASES], single["ok"]
assert suite["ok"] == [c[4] for c in CASES], suite["ok"]
assert single["loose"] == [c[5] for c in CASES], single["loose"]
assert single["setop"] == [c[6] for c in CASES], single["setop"]
assert abs(single["printed"] - 3 / 6) < P.TOL and abs(suite["printed"] - 2 / 6) < P.TOL
assert all(h in P.LEVELS for h in single["hardness"]), single["hardness"]
assert "EXECUTION ACCURACY" in open(f"{data}/perq/{tag}_single.log").read()
print("cell ok: ex", single["ok"], "ts", suite["ok"], "loose", single["loose"],
      "hardness", single["hardness"])

# 2. misalignment and blank pred lines are caught
gold = P.read_gold(P.GOLD)
calls, _, _ = P.score_cell(P.load_evaluator(), P.GOLD, f"{data}/pred_{tag}.txt", P.DB_DIR)
P.check_alignment(calls, gold)
for bad in (gold[1:] + gold[:1], gold[:-1]):
    try:
        P.check_alignment(calls, bad)
        raise SystemExit("misalignment not caught")
    except AssertionError:
        pass
open(f"{root}/blank.txt", "w").write("SELECT 1\n\nSELECT 1\n")
try:
    P.score_cell(P.load_evaluator(), P.GOLD, f"{root}/blank.txt", P.DB_DIR)
    raise SystemExit("blank pred line not caught")
except AssertionError:
    pass
print("alignment and blank-line guards ok")

# 3. McNemar matches the p-values in the handoff (computed there with scipy-free code
#    on the looser re-execution; we check the formula at the precision printed there)
for b, c, want in [(157, 26, "5e-24"), (55, 9, "4e-9"), (22, 8, "0.016"), (73, 22, "1e-7"),
                   (35, 7, "2e-5"), (35, 17, "0.018"), (9, 4, "0.27"), (5, 1, "0.22")]:
    sig = len(want.split("e")[0].replace(".", "").lstrip("0"))
    got = P.mcnemar_p(b, c)
    assert float(f"{got:.{sig}g}") == float(want), (b, c, got, want)
assert P.mcnemar_p(0, 0) == 1.0 and P.mcnemar_p(3, 3) == 1.0
print("mcnemar ok, e.g. 157/26 ->", f"{P.mcnemar_p(157, 26):.2g}", " 22/8 ->", f"{P.mcnemar_p(22, 8):.2g}")

# 4. teacher gap on hand-built vectors
S = [0, 0, 0, 1, 1, 0, 0, 1]
R = [1, 1, 0, 1, 0, 0, 1, 1]
T = [1, 0, 1, 1, 1, 0, 1, 0]
hard = ["easy", "hard", "extra", "easy", "medium", "hard", "easy", "medium"]
setop = [0, 1, 1, 0, 0, 0, 0, 0]
g = P.teacher_gap(S, R, T, hard, setop)
assert (g["fixed"], g["broken"]) == (3, 1)                    # fixed: 0,1,6  broken: 4
assert (g["fixed_teacher_right"], g["broken_teacher_right"]) == (2, 1)
assert g["teacher_right_student_wrong"] == 3                  # 0, 2, 6
assert g["teacher_right_rl_wrong"] == 2 and g["ids_teacher_right_rl_wrong"] == [2, 4]
assert g["rl_right_teacher_wrong"] == 2                       # 1, 7
assert (g["teacher_only"], g["all_wrong"]) == (1, 1)          # 2 ; 5
assert abs(g["oracle_rl_or_teacher"] - 7 / 8) < 1e-12
assert g["teacher_right_rl_wrong_by_level"] == {"easy": 0, "medium": 1, "hard": 0, "extra": 1}
assert g["setop"] == {"n": 2, "fixed": 1, "fixed_teacher_right": 0, "teacher_right_rl_wrong": 1}
print("teacher gap ok")

# 5. build(): four tags, references that pass, then one that fails
preds = {
    "Qwen2.5-Coder-1.5B-Instruct": [c[2] for c in CASES],
    "1.5b-grpo-binary-7k": [c[0] for c in CASES],                       # gold: all right
    "Qwen2.5-Coder-7B-Instruct": [c[0] for c in CASES[:4]] + ["SELECT 1", "SELECT 1"],
    "7b-grpo-binary-7k": [c[0] for c in CASES[:5]] + ["SELECT 1"],
}
for t, ps in preds.items():
    open(f"{data}/pred_{t}.txt", "w").write("".join(p + "\n" for p in ps))
    for w in ("single", "suite"):
        P.cell.local(t, w)
cells = {t: json.load(open(f"{data}/perq/{t}_single.json")) for t in preds}
suites = {t: json.load(open(f"{data}/perq/{t}_suite.json")) for t in preds}
os.makedirs(f"{data}/reeval", exist_ok=True)
json.dump([{"tag": t, "EX_testsuite_single": round(sum(cells[t]["ok"]) / N, 3),
            "TS_suite": round(sum(suites[t]["ok"]) / N, 3),
            "setop_exec_acc_loose": float(cells[t]["loose"][3])} for t in preds],
          open(f"{data}/reeval/summary.json", "w"))
for t in preds:
    json.dump({"pass@1_loose": sum(cells[t]["loose"]) / N}, open(f"{data}/summary_{t}.json", "w"))
b, r = cells["Qwen2.5-Coder-1.5B-Instruct"]["loose"], cells["1.5b-grpo-binary-7k"]["loose"]
os.makedirs(f"{data}/e1", exist_ok=True)
json.dump({"outcome_counts": {"FIXED": sum(1 for x, y in zip(b, r) if not x and y),
                              "BROKEN": sum(1 for x, y in zip(b, r) if x and not y),
                              "BOTH_RIGHT": 0, "BOTH_WRONG": 0}},
          open(f"{data}/e1/1.5b_summary.json", "w"))

P.build(list(preds))
a = json.load(open(f"{data}/perq/analysis.json"))
chk = json.load(open(f"{data}/perq/checks.json"))
assert a["tags_passed"] == sorted(P.SHORT.values()), a["tags_passed"]
assert all(c["ok"] for cs in chk.values() for c in cs)
assert a["acc"]["1.5b_grpo"]["ex"] == 1.0 and a["acc"]["1.5b_base"]["ex"] == 0.5
pe = a["paired"]["1.5b_grpo_vs_1.5b_base"]["ex"]
assert (pe["wins"], pe["losses"]) == (3, 0)
gx = a["teacher_gap"]["7b_base_over_1.5b_grpo"]["ex"]
assert gx["fixed"] == 3 and gx["fixed_teacher_right"] == 1    # 7B base gets q2 only of 2,4,5
rows = [json.loads(l) for l in open(f"{data}/perq/1.5b-grpo-binary-7k.jsonl")]
assert len(rows) == N and set(rows[0]) == {"i", "db_id", "hardness", "setop", "ex", "ts", "loose"}
print("build ok: 4 tags passed, analysis and jsonl written")

# a wrong reference must fail the tag and keep it out of the analysis
ref = json.load(open(f"{data}/reeval/summary.json"))
ref[2]["EX_testsuite_single"] = 0.999
json.dump(ref, open(f"{data}/reeval/summary.json", "w"))
P.build(list(preds))
a = json.load(open(f"{data}/perq/analysis.json"))
chk = json.load(open(f"{data}/perq/checks.json"))
assert "7b_base" not in a["tags_passed"]
assert any(not c["ok"] for c in chk["Qwen2.5-Coder-7B-Instruct"])
assert "7b_base_over_1.5b_grpo" not in a["teacher_gap"]
print("failing reference ok: 7b_base excluded")

shutil.rmtree(root)
print("\nALL PER-QUESTION CPU TESTS PASSED")
