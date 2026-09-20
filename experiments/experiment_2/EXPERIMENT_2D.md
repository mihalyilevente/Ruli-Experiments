# Experiment 2D: localize the LOW-versus-PLACEBO effect

Experiment 2D restores Experiment 2B's final two-epoch retain fine-tuning to
the three-stage pipeline measured here. Experiment 2C ended immediately after
NPO. No 2A/2B/2C code, results, manifest, or upstream RULI source is changed.
The new files are `run_experiment_2d.py`, `evaluate_experiment_2d.py`, this guide,
and `tests/test_experiment_2d.py`.

## Training and capture boundaries

Each seed independently reloads the same persisted **pretrained GPT-2** initial
state for HIGH, LOW, and PLACEBO, just as in 2B/2C. This is not a new random
initialization. Full state-dict SHA-256 hashes must agree for three preflight
reloads and all three actual branch starts. Python, NumPy, PyTorch, and CUDA RNGs
are reset before each branch; the existing helper sets both Trainer `seed` and
`data_seed`. WikiText selection always uses frozen artifact seed 42.

```text
initial_shared
  -> initial SFT (5 epochs, upstream early stopping/best-model selection)
  -> prefix (1 epoch)
  -> save PRE-NPO -> measure PRE-NPO
  -> NPO (15 epochs)
  -> save POST-NPO -> measure POST-NPO
  -> retain FT (2 epochs, upstream best-model selection)
  -> save POST-FT -> measure POST-FT
```

The runner invokes the unchanged upstream `train_sft`, `train_prefix`, and
`unlearn_model` helpers. The 2B dataset builder supplies condition target-IN
plus the same UNLEARN and 15,000 WikiText rows for initial SFT and prefix;
condition target-IN plus that background for NPO's retain side and final FT;
and the identical 200 UNLEARN rows for NPO's forget side.

HIGH retains the ordered original 200 target-IN IDs; LOW uses `(HIGH - U) + R`;
PLACEBO uses `(HIGH - P) + R`. S/U/P/R remain 28/32/32/32. The 121 negative
controls, thresholds, GPT-2, optimizer settings, target partition, and fixed
nine-shadow artifact are all frozen. No shadow training occurs.

`_run_branch()` calls the measurement function synchronously after each save.
Final FT is unreachable if POST-NPO measurement raises or fails validation.
The measurement hashes model parameters before/after inference and restores
Python/NumPy/PyTorch/CUDA RNG state and every module's training mode. Thus the
new instrumentation does not consume subsequent training randomness.

Final FT requests exactly 2 epochs. An observer callback records the actual
`on_train_end` Trainer state in `completed_trainer_state.json`. Before saving
POST-FT or deleting scratch, the runner requires both configured and completed
epochs to equal 2. This works even when upstream checkpoint pruning deletes the
last checkpoint in favor of an earlier best model. Metadata retains the full
completion state, its hash, global steps, and the upstream best checkpoint.
The observer does not modify optimizer settings, training control, or existing
callbacks. **POST-FT is the model returned after the two-epoch
run, including upstream `load_best_model_at_end=True`**, exactly as in 2B; it
may use the weights selected at an earlier epoch. Initial SFT likewise retains
upstream early stopping, so 5 is its configured maximum, as in 2B/2C.

## Outputs and alignment

All outputs live under `results/experiment_2d/seed_<SEED>/`:

```text
initial_shared/
HIGH_pre_npo/       HIGH_post_npo/       HIGH_post_ft/
LOW_pre_npo/        LOW_post_npo/        LOW_post_ft/
PLACEBO_pre_npo/    PLACEBO_post_npo/    PLACEBO_post_ft/
measurements/<CONDITION>_<STAGE>.json
run_metadata.json
trainer_work/       # only during training or retained after a failure
evaluation/
  per_sample_three_stage.csv
  primary_contrast.csv
  evaluation_summary.json
  post_ft_ruli_per_sample.csv
```

All nine measurements use the same immutable token tuples, indexed by explicit
frozen sample IDs, in the exact manifest UNLEARN order. The validated target
loader checks storage hashes, fingerprint, identifier fields, and supported
sample text hashes. An ordered-ID/token hash binds all measurements. Immediate
boundary records store the actual 200 losses and checkpoint hash; run metadata
stores each record's file hash. Evaluation rejects missing, extra, reordered,
or nonfinite measurements, checks every loaded checkpoint's parameter hash
(ten including initial_shared), and verifies reloaded losses against immediate
measurements (`rel_tol=1e-5`, `abs_tol=1e-5`, allowing device numerical rounding).
Artifact, parameter, source, ID, and token hashes require exact equality.

`per_sample_three_stage.csv` has 200 rows, each with seed, sample ID, cohort
flags, all requested HIGH/LOW/PLACEBO losses and deltas, LOW/HIGH-versus-PLACEBO
DiDs, final loss gaps, and secondary per-stage privacy scores/log odds.
`primary_contrast.csv` contains the same fields for the 28 S samples in frozen
S order. The summary reports condition losses/deltas and contrast means,
medians, positive counts, and fractions strictly above zero for supported S,
121 negative controls, and all UNLEARN. Ties do not count as positive.
No five-seed statistical analysis is performed.

## Estimands and interpretation

Loss is the validated RULI mean next-token cross entropy over the **final 7
valid prediction positions**. The existing behavioral check and actual
`MIAEvaluator._batch_inference` path are reused, not reimplemented.

For each sample, condition, and seed:

```text
delta_npo   = post_npo_loss - pre_loss
delta_ft    = post_ft_loss  - post_npo_loss
delta_total = post_ft_loss  - pre_loss

NPO_DiD   = LOW_delta_npo   - PLACEBO_delta_npo
FT_DiD    = LOW_delta_ft    - PLACEBO_delta_ft
TOTAL_DiD = LOW_delta_total - PLACEBO_delta_total
final_gap = LOW_post_ft_loss - PLACEBO_post_ft_loss
```

- Positive NPO_DiD means a larger immediate NPO loss increase in LOW (2C's question).
- Positive FT_DiD means LOW's FT change was more positive. When FT decreases
  loss, this means a larger decrease / stronger relearning in PLACEBO. Inspect
  absolute deltas before concluding either condition actually relearned.
- Positive TOTAL_DiD means a larger net loss increase in LOW from PRE-NPO to
  POST-FT. It equals NPO_DiD + FT_DiD, up to floating-point rounding.
- Positive final_gap means higher final loss, hence stronger remaining
  suppression in LOW. Unlike TOTAL_DiD, this includes any pre-existing gap.

HIGH-minus-PLACEBO repeats these calculations as a robustness reference;
LOW-minus-PLACEBO remains primary. The same positive-fraction summaries for
HIGH are descriptive, not a replacement hypothesis.

## Secondary RULI connection to 2B

PRE-NPO KDE references use `unlearn_original` and `out_original`; POST-NPO and
POST-FT use `unlearn_unlearned` and `out_unlearned`. These scales differ, and the
unlearned shadows include final FT. No cross-stage KDE subtraction is used as
the primary mechanism result.

For POST-FT, the evaluator calls the **same `_evaluate_condition` privacy path
used by 2B**, with the same frozen 200 UNLEARN + 200 OUT IDs, fixed shadows,
Scott bandwidth, and bounded RULI formula. Its per-sample scores must reproduce
upstream `MIAEvaluator.evaluate_with_kde` AUC, ACC, TPR@1%FPR, and TPR@5%FPR.
`post_ft_ruli_per_sample.csv` saves these 1,200 secondary rows across three
conditions. The summary includes aggregate metrics and cohort log-odds
contrasts to connect to 2B. Efficacy is not requested here; its fields remain
empty. Raw-loss DiD remains primary even if secondary KDE contrasts differ.

## Seed 42 commands

Run from the artifact host's existing RULI environment (preserve its CUDA
PyTorch). As in 2B/2C, the frozen target needs `datasets==5.0.1` and
`pyarrow==21.0.0`. These commands assume the existing `/workspace` layout.
The local filename `smoke_700` does not establish artifact identity: exact
frozen storage hashes must pass; substitutes are rejected.

Full CPU validation, temporary shared initial save/reloads, **no training**:

```bash
cd /workspace/Ruli-Experiments
source /workspace/Ruli/.venv/bin/activate
python experiments/experiment_2/run_experiment_2d.py \
  --seed 42 --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --device cpu --validate-only
```

Training seed 42, including all nine synchronous boundary measurements:

```bash
python experiments/experiment_2/run_experiment_2d.py \
  --seed 42 --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --output-root /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2d/seed_42 \
  --device cuda:0
```

Evaluation after training:

```bash
python experiments/experiment_2/evaluate_experiment_2d.py \
  --seed 42 --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --experiment-output /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2d/seed_42 \
  --device cuda:0
```

The evaluator accepts `--device cpu`. Its `--validate-only` checks metadata,
frozen artifacts, boundary records, loss behavior, KDE construction, and ten
checkpoint structures without inference; loaded parameter hashes and boundary
loss reproduction require full evaluation. Runner `--validate-manifest-only`
is a lightweight alternative that does not establish full artifact readiness.

Nonempty training/evaluation destinations are rejected. Seeds 42--46 are
accepted individually, with no multi-seed launcher. No training or long GPU
evaluation is launched automatically by installation or tests.

There are **no algorithm, optimizer, seeding, loss, or final privacy-scoring
changes relative to 2B**. Additions are checkpoint/measurement instrumentation
and stage-localized analysis. Relative to 2C, final retain FT is restored.
CPU synthetic regressions cover stage barriers, actual upstream KDE aggregation,
RNG/mode preservation, exact FT epoch evidence, alignment, nonfinite rejection,
metadata tampering, validation-only behavior, and overwrite protection.
