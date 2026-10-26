# RLVR for Text-to-SQL

Post-training Qwen2.5-Coder with GRPO and an execution-verified reward, and measuring what reinforcement learning actually changes inside the model.

## Results

All numbers live in [report/main.pdf](report/main.pdf), generated from `results/modal/`. Do not copy them into this README.

1. The same GRPO recipe helps the 1.5B model much more than the 7B, in line with how much pass@1-to-pass@8 headroom each base model had.
2. At 1.5B, RL also changes how the model writes SQL (it drops `T1`/`T2` aliases for `table.column`), although the reward never mentions style.
3. The gap between single-database and test-suite accuracy stays roughly constant through training, so there is no clear sign of reward hacking.

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

The report sources (LaTeX and the script that turns `results/modal/` into its numbers) are kept outside this repository. Only the built PDF is committed, at `report/main.pdf`.

## Files

- `modal/`: evaluation, checkpointed GRPO training, E1 token analysis, E2 preference/calibration, E3 style and sampling, E4 alias neurons, E5 copy/remove of GRPO's LoRA delta, and `reeval_official.py` (CPU rescoring of stored predictions). `run_experiments.sh` launches them by name (`bash modal/run_experiments.sh e5`).
- `report/`: the built `main.pdf` only.
- `results/modal/`: evaluation summaries and predictions by model, shared `gold.txt`, and E1/E2 summaries, copied from the local `modal_pull/` export.
- `notebooks/legacy_colab/` and `results/legacy_colab/`: original notebooks and their outputs.

The old numbers in [notebooks/legacy_colab](notebooks/legacy_colab/) used evaluation prompts truncated at 1,536 tokens and are superseded by [report/main.pdf](report/main.pdf).

## Next

E3 (style and sampling) and E4 (alias neurons) results are in the report. E4 steering needs smaller doses.

1. Layer-wise LoRA ablation to locate which layers and attention heads carry the gain.
2. Test whether the style shift is causal.
3. Test training-free recovery via execution self-consistency.
4. More seeds, then DAPO.

## References

- Spider: [Yu et al., 2018, arXiv:1809.08887](https://arxiv.org/abs/1809.08887).
- Test-suite accuracy: [Zhong, Yu & Klein, 2020, arXiv:2010.02840](https://arxiv.org/abs/2010.02840).
- GRPO: [Shao et al., 2024, arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
- [Yue et al., 2025, arXiv:2504.13837](https://arxiv.org/abs/2504.13837).
