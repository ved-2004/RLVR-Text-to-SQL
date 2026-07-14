# RLVR for Text-to-SQL

Post-training a small code LLM with reinforcement learning from verifiable rewards (GRPO + execution feedback), and studying **whether RL extends a model's capability or merely sharpens its confidence.**

## Results

Spider dev set (1,034 examples), Qwen2.5-Coder-1.5B-Instruct.

**Execution accuracy** (official Spider evaluator):

| | easy | medium | hard | extra | **all** |
|---|---|---|---|---|---|
| Base | 0.665 | 0.258 | 0.253 | 0.199 | **0.345** |
| + GRPO | 0.810 | 0.359 | 0.299 | 0.217 | **0.434** |
| Δ | +14.5 | +10.1 | +4.6 | +1.8 | **+8.9** |

**Capability vs. confidence** (pass@1 = one greedy attempt, pass@8 = best of 8 samples):

| | pass@1 | pass@8 |
|---|---|---|
| Base | 0.450 | 0.660 |
| + GRPO | 0.525 | 0.670 |
| Δ | **+7.5** | **+1.0** |

**False-positive rate** (test-suite evaluator, GRPO model — same evaluator, only the databases change):

| | accuracy |
|---|---|
| Single database | 0.601 |
| Across test suite | 0.519 |
| **Gap** | **8.2 pts** |

## Findings

**RL sharpened the model rather than expanding it.** pass@1 rose 7.5 points while pass@8 moved 1.0 — the set of problems the model can solve *at all* is essentially unchanged. It learned to pick the right query more often on the first try, not to solve new problems.

The clearest evidence: the model scores **0.000 on set operations (INTERSECT / UNION / EXCEPT) both before and after training.** GRPO improved execution accuracy by 9 points without teaching it a single new SQL construct. This independently reproduces the effect reported by [Yue et al.](https://arxiv.org/abs/2504.13837) on a different task.

**Execution accuracy overstates correctness.** A wrong query can return the right rows on one database by coincidence (`WHERE age > 34` vs `>= 34`). The [test-suite metric](https://arxiv.org/abs/2010.02840) runs each query against many perturbed databases and only counts it correct if it matches on all of them. The 8.2-point gap is the false-positive rate — roughly 1 in 7 "correct" answers wasn't. This gap is where reward hacking would show up, and it's the primary instrument for the next phase.

## Setup

- **Model:** Qwen2.5-Coder-1.5B-Instruct (general code model, deliberately *not* SQL-specialized, so "did RL help?" stays interpretable)
- **Algorithm:** GRPO (TRL `GRPOTrainer`), LoRA r=16 on all linear layers, 8 rollouts per prompt
- **Reward:** binary execution — run the generated SQL, compare the result set to gold, 1 or 0. No shaping.
- **Data:** Spider train split (500 examples), evaluated on the full dev set (1,034)
- **Eval:** official Spider evaluator (execution accuracy) + test-suite evaluator (false-positive audit)

## Caveats

1.5B model, 500 training examples, single seed, single epoch. This is a smoke test — the numbers demonstrate the pipeline works and the effect is real, not that this is a tuned result.

Note that the two Spider evaluators use different comparison logic and produce different absolute numbers on identical predictions. Any gap must be measured with a *single* evaluator, varying only the databases.

## Repo

```
notebooks/01_baseline.ipynb    # evaluate the untrained model → 34.5%
notebooks/02_grpo.ipynb        # GRPO training + eval → 43.4%
results/                       # raw evaluator output
```

## Next

- Scale to the full 7K Spider train split
- Move to Qwen2.5-Coder-7B
- Controlled comparison: **GRPO vs DAPO vs PPO**, same task, same reward — does algorithm choice or reward design matter more?
- Reward-shaping study: does dense/partial credit help, or does it just widen the false-positive gap?

## References

- Spider — [Yu et al., 2018](https://arxiv.org/abs/1809.08887) · [dataset](https://yale-lily.github.io/spider)
- Test-suite accuracy — [Zhong, Yu & Klein, 2020](https://arxiv.org/abs/2010.02840) · [code](https://github.com/taoyds/test-suite-sql-eval)
- GRPO — [DeepSeekMath, Shao et al., 2024](https://arxiv.org/abs/2402.03300)
- pass@k analysis — [Yue et al., 2025](https://arxiv.org/abs/2504.13837)
