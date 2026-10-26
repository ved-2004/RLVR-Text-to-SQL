"""
Writes report/numbers.tex from the result files in results/modal/. The report
uses only these macros, so no evaluation number is ever typed by hand.

    python report/make_numbers.py            # run from the repo root

Reads (missing files become a red ?? in the PDF, never a guess):
    results/modal/{1.5b_base,1.5b_grpo,7b_base,7b_grpo}/summary_*.json
        ts_single_raw, ts_suite_raw  -> test-suite evaluator, "execution" row, "all" column
        pass@1_loose, pass@8_loose
    results/modal/reeval/summary.json   (from modal/reeval_official.py)
        EX_official_spider, setop_exec_acc_loose, setop_n
    results/modal/e3/summary.json       (from modal/e3_style_and_sampling.py)
    results/modal/e4/{find,sweep}_1.5b.json   (from modal/e4_alias_neurons.py)

Never read ex_raw: it is a truncated log (see the Evaluation note in the report).
Numbers from E1/E2 (KL, style counts, ECE) live in report/numbers_manual.tex.
"""

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "modal")
OUT = os.path.join(ROOT, "report", "numbers.tex")

MODELS = {"SB": "1.5b_base", "SR": "1.5b_grpo", "LB": "7b_base", "LR": "7b_grpo"}
REEVAL_TAGS = {"SB": "Qwen2.5-Coder-1.5B-Instruct", "SR": "1.5b-grpo-binary-7k",
               "LB": "Qwen2.5-Coder-7B-Instruct", "LR": "7b-grpo-binary-7k"}
MISSING = r"\textcolor{red}{??}"

# E3: (tag, style, decode) -> macro prefix Ethree{model}{style}{decode}
E3_TAGS = {"SB": "1.5b-base", "SR": "1.5b-rl"}
E3_STYLES = {"Def": "default", "No": "no_alias", "Al": "alias"}
E3_DECODES = {"G": "greedy", "V": "sc8"}
E3_METRICS = {"Ex": ("EX", ".3f"), "Ts": ("TS_single", ".3f"), "Suite": ("TS_suite", ".3f"),
              "Alias": ("alias_per_q", ".2f"), "Qual": ("qualified_per_q", ".2f")}
# E4 sweep rows -> macro prefix Efour{code}
E4_RUNS = {"Base": "base", "Grpo": "grpo",
           "TopTwoHalf": "base_top2_d0.5", "TopTwoOne": "base_top2_d1",
           "TopEightQuarter": "base_top8_d0.25", "TopEightHalf": "base_top8_d0.5",
           "RandTwoOne": "base_rand2_d1", "RandEightHalf": "base_rand8_d0.5",
           "ZeroEight": "base_zero_top8",
           "TopTwoOneRev": "base_top2_d1_more_alias", "GrpoTopTwoOneRev": "grpo_top2_d1_more_alias"}
E4_METRICS = {"Ex": ("EX", ".3f"), "Ts": ("TS_single", ".3f"), "Suite": ("TS_suite", ".3f"),
              "Valid": ("valid_sql", ".3f"), "Alias": ("alias_per_q", ".2f"),
              "Qual": ("qual_per_q", ".2f")}


def load_json(*parts):
    p = os.path.join(RES, *parts)
    return json.load(open(p)) if os.path.exists(p) else None


def share(part_lo, part_hi, lo, hi):
    """Percent of the base-to-GRPO gain (lo -> hi) covered by part_lo -> part_hi."""
    if None in (part_lo, part_hi, lo, hi) or hi == lo:
        return None
    return 100 * (part_hi - part_lo) / (hi - lo)


def e3_e4_macros():
    """(name, value, format) for E3 and E4. Formats are explicit, unlike main()."""
    out = []
    e3 = {(r["tag"], r["style"], r["decode"]): r for r in (load_json("e3", "summary.json") or [])}
    g = {}
    for mc, tag in E3_TAGS.items():
        for sc, style in E3_STYLES.items():
            for dc, dec in E3_DECODES.items():
                r = e3.get((tag, style, dec), {})
                for kc, (key, f) in E3_METRICS.items():
                    g[mc + sc + dc + kc] = r.get(key)
                    out.append((f"Ethree{mc}{sc}{dc}{kc}", r.get(key), f))

    def d(a, b):
        return None if a is None or b is None else 100 * (a - b)

    for m in ("Ts", "Suite", "Ex"):
        lo, hi = g[f"SBDefG{m}"], g[f"SRDefG{m}"]
        out += [(f"EthreeVoteShare{m}", share(lo, g[f"SBDefV{m}"], lo, hi), ".0f"),
                (f"EthreeVoteNoShare{m}", share(lo, g[f"SBNoV{m}"], lo, hi), ".0f")]
        for mc in E3_TAGS:
            out += [(f"dEthreeAl{mc}{m}", d(g[f"{mc}AlG{m}"], g[f"{mc}DefG{m}"]), "+.1f"),
                    (f"dEthreeNo{mc}{m}", d(g[f"{mc}NoG{m}"], g[f"{mc}DefG{m}"]), "+.1f"),
                    (f"dEthreeVote{mc}{m}", d(g[f"{mc}DefV{m}"], g[f"{mc}DefG{m}"]), "+.1f")]

    sweep = {r["name"]: r for r in (load_json("e4", "sweep_1.5b.json") or [])}
    s = {}
    for rc, name in E4_RUNS.items():
        r = sweep.get(name, {})
        for kc, (key, f) in E4_METRICS.items():
            s[rc + kc] = r.get(key)
            out.append((f"Efour{rc}{kc}", r.get(key), f))
    steered = [r for r in sweep.values() if r.get("mode") == "add"]
    out += [(f"EfourZeroShare{m}", share(s[f"Base{m}"], s[f"ZeroEight{m}"],
                                         s[f"Base{m}"], s[f"Grpo{m}"]), ".0f") for m in ("Ts", "Suite")]
    out += [(f"dEfourZero{m}", d(s[f"ZeroEight{m}"], s[f"Base{m}"]), "+.1f") for m in ("Ts", "Suite")]
    out += [(f"dEfourTopTwoHalf{m}", d(s[f"TopTwoHalf{m}"], s[f"Base{m}"]), "+.1f") for m in ("Ts", "Suite")]
    out += [("EfourSteered", len(steered) if sweep else None, "d"),
            ("EfourBroken", sum(r["valid_sql"] == 0 for r in steered) if sweep else None, "d")]

    f = load_json("e4", "find_1.5b.json") or {}
    sc = f.get("screened") or []
    top = f.get("top_atp") or []
    layers = [t["layer"] for t in top]
    top_layer = max(set(layers), key=layers.count) if layers else None
    dm = None if f.get("m_rl_mean") is None else f["m_rl_mean"] - f["m_base_mean"]
    pct = lambda x: None if x is None else 100 * x
    out += [("EfourPairs", f.get("n_pairs_attr"), "d"),
            ("EfourScreenPairs", f.get("n_pairs_screen"), "d"),
            ("EfourMBase", f.get("m_base_mean"), ".2f"),
            ("EfourMRl", f.get("m_rl_mean"), ".2f"),
            ("EfourDm", dm, ".2f"),
            ("EfourAtpTotal", f.get("atp_total"), ".2f"),
            ("EfourPrefAliasBase", pct(f.get("m_base_frac_prefers_alias")), ".1f"),
            ("EfourPrefAliasRl", pct(f.get("m_rl_frac_prefers_alias")), ".1f"),
            ("EfourTopShare", pct(f.get("atp_top100_share")), ".0f"),
            ("EfourTopN", len(top) if top else None, "d"),
            ("EfourTopLayer", top_layer, "d"),
            ("EfourTopLayerCount", layers.count(top_layer) if layers else None, "d"),
            ("EfourScreened", len(sc) if sc else None, "d"),
            ("EfourAgree", sum(x["rl_shift_agrees"] for x in sc) if sc else None, "d")]
    for i, x in enumerate(sc[:2]):
        out += [(f"EfourScr{'AB'[i]}Layer", x["layer"], "d"),
                (f"EfourScr{'AB'[i]}Neuron", x["neuron"], "d"),
                (f"EfourScr{'AB'[i]}AtpRank", x["atp_rank"], "d")]
    return out


def exec_all(raw):
    """'all' column of the execution row, read only after the count header."""
    started = False
    for line in (raw or "").splitlines():
        s = line.strip()
        if s.startswith("count") and len(s.split()) == 6:
            started = True
        elif started and s.startswith("execution"):
            vals = re.findall(r"-?\d+\.\d+", s)
            if len(vals) == 5:
                return float(vals[-1])
    return None


def load_summary(folder):
    files = glob.glob(os.path.join(RES, folder, "summary_*.json"))
    if len(files) != 1:
        print(f"warning: expected 1 summary in {folder}, found {len(files)}", file=sys.stderr)
        return {}
    return json.load(open(files[0]))


def main():
    v = {}
    for code, folder in MODELS.items():
        s = load_summary(folder)
        v[f"Ex{code}"] = exec_all(s.get("ts_single_raw"))
        v[f"TsSuite{code}"] = exec_all(s.get("ts_suite_raw"))
        v[f"PassOne{code}"] = s.get("pass@1_loose")
        v[f"PassEight{code}"] = s.get("pass@8_loose")

    re_path = os.path.join(RES, "reeval", "summary.json")
    rows = {r["tag"]: r for r in json.load(open(re_path))} if os.path.exists(re_path) else {}
    setop_n = None
    for code, tag in REEVAL_TAGS.items():
        r = rows.get(tag, {})
        v[f"ExOfficial{code}"] = r.get("EX_official_spider")
        v[f"SetOp{code}"] = r.get("setop_exec_acc_loose")
        setop_n = r.get("setop_n", setop_n)
        # cross-check: the re-scored test-suite number must equal the stored one
        ts = r.get("EX_testsuite_single")
        if ts is not None and v[f"Ex{code}"] is not None and abs(ts - v[f"Ex{code}"]) > 1e-3:
            sys.exit(f"MISMATCH {code}: summary EX {v[f'Ex{code}']} vs reeval {ts}")

    def pts(a, b):
        return None if a is None or b is None else 100 * (a - b)

    for m in ("Ex", "TsSuite", "PassOne", "PassEight", "ExOfficial", "SetOp"):
        for size in ("S", "L"):
            v[f"d{m}{size}"] = pts(v[f"{m}{size}R"], v[f"{m}{size}B"])
    for code in MODELS:
        v[f"Gap{code}"] = pts(v[f"Ex{code}"], v[f"TsSuite{code}"])
    for size in ("S", "L"):
        gr, gb = v[f"Gap{size}R"], v[f"Gap{size}B"]
        v[f"dGap{size}"] = None if gr is None or gb is None else gr - gb   # already in points
        v[f"Head{size}"] = pts(v[f"PassEight{size}B"], v[f"PassOne{size}B"])
        # in percent, for prose
        for code in (f"{size}B", f"{size}R"):
            v[f"ExPct{code}"] = None if v[f"Ex{code}"] is None else 100 * v[f"Ex{code}"]

    def fmt(name, x):
        if x is None:
            return MISSING
        if name.startswith("d"):
            return f"${x:+.1f}$"
        if name.startswith(("Gap", "Head", "ExPct")):
            return f"{x:.1f}"
        return f"{x:.3f}"

    lines = ["% GENERATED by report/make_numbers.py. Do not edit by hand.",
             f"% source: {os.path.relpath(RES, ROOT)}"]
    for name in sorted(v):
        lines.append(f"\\newcommand{{\\{name}}}{{{fmt(name, v[name])}}}")
    lines.append(f"\\newcommand{{\\SetOpN}}{{{setop_n if setop_n else MISSING}}}")
    for name in sorted(k for k in v if k.startswith(("PassOne", "PassEight"))):
        x = v[name]
        lines.append(f"\\newcommand{{\\Plot{name}}}{{{'nan' if x is None else f'{x:.3f}'}}}")
    extra = e3_e4_macros()
    for name, x, f in extra:
        val = MISSING if x is None else f"{x:{f}}"
        if f.startswith("+") and x is not None:
            val = f"${val}$"
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    open(OUT, "w").write("\n".join(lines) + "\n")
    missing = sorted(k for k, x in v.items() if x is None) + [n for n, x, _ in extra if x is None]
    print(f"wrote {OUT}: {len(v) + len(extra)} macros, {len(missing)} missing")
    if missing:
        print("  missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
