# Experiment 1: sample-level RULI export

`export_ruli_scores.py` reads existing shadow outputs and saved target-model
checkpoints. It performs inference only; it does not train, unlearn, or modify the
official RULI implementation.

Run it from the repository root:

```bash
python experiments/experiment_1/export_ruli_scores.py \
  --shadow-path ../Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path ../Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed \
  --original-checkpoint ../Ruli/path/to/original-checkpoint \
  --unlearned-checkpoint ../Ruli/path/to/unlearned-checkpoint \
  --device cuda:0
```

All RULI paths are inputs and may point to a sibling checkout, a mounted RunPod
volume, or downloaded artifacts. Nothing is imported from or written into the RULI
source tree.

The default output is `results/ruli_scores.csv` in this directory. The default ID
slices reproduce the official text evaluation: sorted IDs 200--399 are UNLEARN and
400--599 are OUT. Override them with `--unlearn-start`, `--unlearn-count`,
`--out-start`, and `--out-count` when a run used different slices.

The export contains both evaluation groups so its metrics can be checked against the
official attack. For graph correlations on the forget set, filter to
`split == "unlearn"`.

## Score definitions

- `privacy_score`: target loss under the unlearned model, scored against
  `unlearn_unlearned` (positive) and `out_unlearned` (negative) per-sample KDEs.
- `efficacy_score`: the UNLEARN target loss under the unlearned model, or the OUT
  target loss under the original model, scored against `unlearn_unlearned`
  (positive) and `out_original` (negative) per-sample KDEs.
- `out_shadow_mean`: mean of `out_unlearned`, the privacy OUT reference requested
  for the initial CSV schema.
- `efficacy_out_shadow_mean`: mean of `out_original`, included because the official
  efficacy attack uses a different OUT reference.
- `loss_change`: `unlearned_loss - original_loss`.
- `split`: `unlearn` or `out`; both attack labels are 1 for UNLEARN and 0 for OUT.

The script prints AUC, accuracy, TPR@1%FPR, and TPR@5%FPR reconstructed from the
exported scores. These are an immediate check that the CSV matches the saved run.
Per-sample KDE needs at least two non-singular observations in each condition. For
incomplete smoke-test shadow files, `--kde-error nan` exports available values and
writes unavailable scores as NaN instead of stopping.
