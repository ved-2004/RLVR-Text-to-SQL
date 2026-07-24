# RLVR for Text-to-SQL

Post-training a small code LLM with reinforcement learning from verifiable rewards (GRPO + execution feedback), and measuring **whether RL extends what the model can do or just makes it more reliable at what it already could.**

## Results

Spider dev set (1,034 examples), Qwen2.5-Coder-1.5B-Instruct, scored with the official Spider evaluator.

### Execution accuracy

| difficulty | base | GRPO 500 ex | GRPO 7,000 ex |
|---|---|---|---|
| easy | 0.665 | 0.810 | 0.847 |
| medium | 0.258 | 0.359 | 0.455 |
| hard | 0.253 | 0.299 | 0.408 |
| extra | 0.199 | 0.217 | 0.223 |
| **all** | **0.345** | **0.434** | **0.504** |

Exact-match accuracy moved in step: 0.285 → 0.368 → 0.423.

Going from 500 to 7,000 training examples, easy questions barely improved (+3.7 — saturating) while medium gained +9.6 and hard +10.9. The additional data bought progress on harder queries rather than more polish on easy ones.

### Capability vs. reliability

pass@1 = one greedy attempt. pass@8 = sample 8 times, count it if any is correct. pass@8 approximates the set of problems the model can reach at all, which separates "learned something new" from "got better at picking."

Full dev set, base vs. GRPO 7,000:

| | base | GRPO 7,000 ex | change |
|---|---|---|---|
| pass@1 | 0.496 | 0.677 | **+0.181** |
| pass@8 | 0.733 | 0.809 | **+0.075** |
| gap | 0.237 | 0.132 | −0.105 |

At 500 training examples, pass@8 moved **+1.0 point** (measured on a 200-question subset). At 7,000 it moved **+7.5**.

### Set operations

IUEN accuracy (INTERSECT / UNION / EXCEPT / nested queries):

| | base | GRPO 500 ex | GRPO 7,000 ex |
|---|---|---|---|
| hard | 0.000 | 0.000 | 0.143 |
| extra | 0.000 | 0.000 | 0.059 |
| all | 0.000 | 0.000 | 0.105 |

The model never produced a correct set operation at base or at 500 examples. At 7,000 it starts to.

### False-positive rate

A wrong query can return the right rows on one database by coincidence (`WHERE age > 34` vs `>= 34` when nobody is exactly 34). The [test-suite metric](https://arxiv.org/abs/2010.02840) runs each query against many perturbed copies of the database and only counts it correct if it matches on all of them.

Same evaluator, same predictions — only the databases change:

| | single database | test suite | gap |
|---|---|---|---|
| GRPO 500 ex | 0.601 | 0.519 | 8.2 pts |
| GRPO 7,000 ex | 0.697 | 0.607 | 9.0 pts |

Roughly one in seven "correct" answers is only correct by coincidence. This gap is where reward hacking shows up: since the training reward *is* execution feedback, every false positive is a reward earned without being right. If the gap widens as training gets more aggressive, the model is learning to exploit the verifier.

## Findings

**Both sharpening and extension happen, at different data scales.** pass@1 rose more than twice as much as pass@8 (+18.1 vs +7.5), so most of the gain is the model becoming more reliable at queries it could already reach — the gap between its greedy answer and its best-of-8 answer narrowed from 23.7 points to 13.2. But pass@8 rising at all means the reachable set itself expanded, and the set-operation result agrees: a construct that was flat-zero at two training scales starts working at the third.

At 500 examples only the sharpening was visible. It took 14× the data for the capability gain to appear.

## Setup

- **Model:** Qwen2.5-Coder-1.5B-Instruct — a general code model, deliberately not SQL-specialized, so "did RL help?" stays interpretable
- **Algorithm:** GRPO via TRL `GRPOTrainer`, LoRA r=16 / alpha=32 on all linear layers, 8 rollouts per prompt, 1 epoch, lr 1e-5, KL beta 0.04, temperature 0.9
- **Reward:** binary execution. Run the generated SQL, run the gold SQL, compare result sets. Match → 1, anything else (wrong rows, error, unparseable) → 0. No shaping.
- **Data:** Spider train split, evaluated on the full dev set (1,034). Spider's test set is held by the authors for the leaderboard and is not used.
- **Eval:** official Spider evaluator for execution and exact-match accuracy; test-suite evaluator for the false-positive audit.

## Caveats

1.5B model, single seed, one epoch. This demonstrates the pipeline and the effect; it is not a tuned result.

The pass@k numbers use a simplified correctness check (set comparison, ignoring row order and duplicates) rather than the official evaluator. Both sides of every pass@k comparison use the same check, so the deltas are valid, but the absolute values are more permissive than the headline execution accuracy and should not be compared to it. The 500-example pass@k was measured on a 200-question subset.

**Note on the two evaluators:** the original Spider evaluator and the test-suite evaluator use different comparison logic and return different numbers on identical predictions. Any gap must be measured with a single evaluator, varying only the databases.

## Repo

```
notebooks/
  01_baseline.ipynb      # evaluate the untrained model → 0.345
  02_grpo_smoke.ipynb    # GRPO, 500 training examples → 0.434
  03_grpo_full.ipynb     # GRPO, full 7,000-example split → 0.504
results/
  summary.json           # every number from the runs, in one file
  gold.txt               # Spider dev reference queries
  pred_base.txt          # base model predictions
  pred_grpo_7k.txt       # GRPO 7,000 predictions
  ex_base.txt            # official evaluator output, base
  ex_grpo_7k.txt         # official evaluator output, GRPO 7,000
  ts_grpo_7k.txt         # test-suite evaluator output
  grpo_7k_trainlog.json  # per-step reward, KL, loss
```

The prediction files are the reproducibility path: with `gold.txt` and either `pred_*.txt`, anyone can re-run the official Spider evaluator and verify these numbers without a GPU.

## Next

- Scale to Qwen2.5-Coder-7B
- Controlled comparison: **GRPO vs DAPO vs PPO**, same task, same reward — does algorithm choice or reward design matter more?
- Reward shaping: replace the binary signal with denser partial credit, and check whether the false-positive gap widens (i.e. whether the model starts gaming the verifier)
- Move pass@k onto the official evaluator so it's directly comparable to the headline accuracy

## References

- Spider — [Yu et al., 2018](https://arxiv.org/abs/1809.08887) · [dataset](https://yale-lily.github.io/spider) · [evaluator](https://github.com/taoyds/spider)
- Test-suite accuracy — [Zhong, Yu & Klein, 2020](https://arxiv.org/abs/2010.02840) · [code](https://github.com/taoyds/test-suite-sql-eval)
- GRPO — [DeepSeekMath, Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- pass@k and the reasoning boundary — [Yue et al., 2025](https://arxiv.org/abs/2504.13837)
- Qwen2.5-Coder — [model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
