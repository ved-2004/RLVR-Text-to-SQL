# RLVR for Text-to-SQL

Post-training Qwen2.5-Coder with GRPO and an execution-verified reward, and measuring what reinforcement learning actually changes inside the model.

## Results

Spider dev, 1,034 questions. All current results come from Modal runs with identical prompts and greedy decoding for single predictions. pass@8 uses sampled completions.

| metric | 1.5B base | 1.5B GRPO | 7B base | 7B GRPO |
|---|---|---|---|---|
| EX (official) | 0.521 | 0.580 | 0.587 | 0.596 |
| test-suite, single DB | 0.591 | 0.718 | 0.768 | 0.794 |
| test-suite, suite DBs | 0.510 | 0.632 | 0.695 | 0.730 |
| false-positive gap (pts) | 8.1 | 8.6 | 7.3 | 6.4 |
| pass@1 | 0.566 | 0.689 | 0.729 | 0.755 |
| pass@8 | 0.753 | 0.786 | 0.797 | 0.816 |
| IUEN (set operations) | 0.013 | 0.105 | 0.184 | 0.171 |
| ECE (token level) | 0.049 | 0.062 | 0.074 | 0.086 |

The false-positive gap compares single-DB and suite-DB accuracy using the same test-suite evaluator.

## Findings

1. **Gain tracks headroom.** Before training, the pass@1 to pass@8 gap was 18.7 points at 1.5B and 6.8 at 7B. The same recipe gave +5.9 EX points at 1.5B and +0.9 at 7B. The untrained 7B beats the trained 1.5B on EX.
2. **Token level.** Teacher-forcing gold SQL through base and trained models gives mean per-token KL of 0.043 at 1.5B versus 0.012 at 7B (3.65x). Training fixed 155 dev questions at 1.5B versus 35 at 7B. At 1.5B, the change is broad, with entropy down about 0.11 nats across all SQL roles. At 7B, the small change concentrates at aggregate and FROM tokens.
3. **Style convergence.** At 1.5B, aliases (T1, T2) per query fall from 2.63 to 0.36, qualified `table.column` references rise from 0.46 to 1.41, and JOINs fall from 0.51 to 0.35. The 7B base already writes this way (0.43 aliases, 1.62 qualified references) and barely changed. Nothing in the reward mentions style.
4. **Execution accuracy overstates correctness.** Single-DB accuracy exceeds suite-DB accuracy by 6 to 9 points at every scale, using the same evaluator. Under a binary reward, the gap did not widen in a way that suggests reward hacking (7B: 7.3 to 6.4 points).

## Caveats

One seed, one epoch, LoRA r=16, single-turn. pass@k uses a looser set-based check, so compare pass@k only to pass@k. ECE uses argmax-equals-gold-token, which the style shift contaminates. Treat it as suggestive.

## Setup

Qwen2.5-Coder-1.5B/7B-Instruct, TRL `GRPOTrainer`, LoRA r=16 and alpha=32 on all linear layers, 8 rollouts per prompt, lr 1e-5, beta 0.04, binary execution reward, and the full 7,000-example Spider train split. Training runs on Modal A100 with W&B logging. Evaluation generation uses vLLM.

## Reproduce

Run from the repository root with the Modal CLI installed and authenticated. Configure a Modal secret named `wandb` containing `WANDB_API_KEY`, following the training script's docstring. Keep the dataset archives in the parent workspace directory.

```sh
modal volume create spider-data
modal volume put spider-data ../spider_data.zip /spider_data.zip
modal volume put spider-data ../test_suite_database.zip /test_suite_database.zip
modal run modal/spider_eval_modal.py::setup_data
```

Select a scale and train. Set `scale=1.5` for the smaller model.

```sh
scale=7
base="Qwen/Qwen2.5-Coder-${scale}B-Instruct"
run="${scale}b-grpo-binary-7k"
modal run --detach modal/spider_grpo_modal.py::train --model-id "$base" --run-name "$run"
```

After training finishes, merge and evaluate both models. Training checkpoints are saved to the Modal volume and resumed when the same run name is reused.

```sh
modal run modal/spider_grpo_modal.py::merge --model-id "$base" --run-name "$run"
modal run modal/spider_eval_modal.py::evaluate --model-id "$base"
modal run modal/spider_eval_modal.py::evaluate --model-id "/data/merged/$run"
```

Run the token-role analysis (E1) and preference, calibration, and style analysis (E2) using the predictions generated on that volume.

```sh
modal run modal/e1_token_analysis.py::analyze \
  --base-id "$base" --rl-path "/data/merged/$run" --tag "${scale}b" \
  --pred-base "/data/pred_${base##*/}.txt" --pred-rl "/data/pred_$run.txt"
modal run modal/e2_preference_calibration.py::analyze \
  --base-id "$base" --rl-path "/data/merged/$run" --tag "${scale}b" \
  --pred-base "/data/pred_${base##*/}.txt" --pred-rl "/data/pred_$run.txt"
```

## Files

- `modal/`: evaluation, checkpointed GRPO training, E1 token analysis, and E2 preference/calibration scripts.
- `results/modal/`: evaluation summaries and predictions by model, shared `gold.txt`, and E1/E2 summaries, copied from the local `modal_pull/` export.
- `notebooks/legacy_colab/` and `results/legacy_colab/`: original notebooks and their outputs.

The old numbers in [notebooks/legacy_colab](notebooks/legacy_colab/) used evaluation prompts truncated at 1,536 tokens and are superseded by [results/modal](results/modal/).

## Next

1. Layer-wise LoRA ablation to locate which layers and attention heads carry the gain.
2. Test whether the style shift is causal.
3. Test training-free recovery via execution self-consistency.
4. More seeds, then DAPO.

## References

- Spider: [Yu et al., 2018, arXiv:1809.08887](https://arxiv.org/abs/1809.08887).
- Test-suite accuracy: [Zhong, Yu & Klein, 2020, arXiv:2010.02840](https://arxiv.org/abs/2010.02840).
- GRPO: [Shao et al., 2024, arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
- [Yue et al., 2025, arXiv:2504.13837](https://arxiv.org/abs/2504.13837).
