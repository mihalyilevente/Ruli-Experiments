# Model generalization: first matched shadow pair

This compact export contains completed shadow 0 for GPT-2 and Pythia-160M, using the same frozen assignment map. It is a fresh matched comparison, not an exact reproduction of the historical shadow run.

- `training_summary.json`: validated completion, runtime, epoch counts, and prediction exceptions.
- `*_shadow_0_per_sample.csv`: 660 sample rows per model, with roles and original/unlearned losses.
- `<model>/shadow_0/`: original losses and all four stage completion records.
- `logs/` and `provenance/`: full training logs, assignment map, preflights, environment, and launch notes.
- `download_checksums.json`: SHA-256 hashes verified against the pod for every downloaded file.

Only one of nine shadows per model has been trained. Final RULI scores and HIGH/LOW/PLACEBO contrasts are not available yet; they require the remaining shadows and target runs. Model checkpoints and optimizer scratch remain on the existing RunPod volume and are not included in this compact export.
