# Experiment 2

The frozen protocol and intervention manifest define Experiment 2 before any
training begins. Do not regenerate or edit `results/intervention_manifest.json`
while running Experiment 2A or Experiment 2B.

## Experiment 2A: training

`run_experiment_2a.py` imports the current training helpers from the sibling RULI
checkout. It reproduces the official initial GPT-2 SFT, prefix training, and NPO
stages once. It then saves one shared post-NPO checkpoint and independently loads
that checkpoint for the HIGH, LOW, and PLACEBO two-epoch final retain-SFT stages.
The preregistered model seeds are 42, 43, 44, 45, and 46. The model seed is
explicitly passed to Python, NumPy, PyTorch, CUDA, and both `seed` and `data_seed`
in every upstream Hugging Face `TrainingArguments` instance.

The WikiText background remains frozen independently: it is always reconstructed
with selection seed 42 and checked row-by-row against the immutable manifest.
Changing the model seed does not change the target dataset, intervention sets,
background membership, target evaluation partition, or shadow artifact.

The runner verifies the manifest's internal hash and protocol checks, the exact
shadow and target artifacts, the official evaluation partition, all condition
set relations, and every frozen WikiText background row before training. A full
preflight that does not allocate or train GPT-2 is available with
`--validate-only`.

From RunPod, with the repositories at `/workspace/Ruli` and
`/workspace/Ruli-Experiments` and the environment activated, run:

```bash
cd /workspace/Ruli-Experiments
RULI_ROOT=/workspace/Ruli
SHADOW_PATH=$RULI_ROOT/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth
TARGET_PATH=$RULI_ROOT/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700
python experiments/experiment_2/run_experiment_2a.py \
  --seed 43 \
  --ruli-root "$RULI_ROOT" \
  --shadow-path "$SHADOW_PATH" \
  --target-data-path "$TARGET_PATH" \
  --device cuda:0
```

Outputs are written under `results/experiment_2a/seed_<SEED>/` and ignored by Git.
The runner refuses to overwrite an existing checkpoint or metadata file.

This phase generates training checkpoints only. It does not run KDE/RULI
evaluation, calculate privacy or efficacy, train shadow models, or implement
Experiment 2B.

## Experiment 2A: evaluation

`evaluate_experiment_2a.py` validates the frozen manifest, exact 9-shadow
artifact, official target dataset, all four seed-specific checkpoint directories,
the training-run seed and frozen hyperparameters, and
the upstream RULI text-loss behavior. Under RULI's thirds assignment, nine total
shadow models yield three observations per sample in each IN, OUT, and UNLEARN
condition distribution. The evaluator evaluates HIGH, LOW, and PLACEBO and
writes identifier-aligned per-sample privacy scores for all 200 UNLEARN and 200
OUT samples. The primary output is the preregistered paired contrast
`privacy_log_odds_LOW - privacy_log_odds_PLACEBO` for the 28 supported samples.

Validate the complete input layout without loading model weights:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/evaluate_experiment_2a.py \
  --seed 43 \
  --ruli-root /workspace/Ruli \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --experiment-output /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2a/seed_43 \
  --device cuda:0 \
  --validate-only
```

Run one seed's evaluation explicitly:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/evaluate_experiment_2a.py \
  --seed 43 \
  --ruli-root /workspace/Ruli \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --experiment-output /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2a/seed_43 \
  --device cuda:0
```

The reference efficacy evaluator scores UNLEARN losses from the final model and
OUT losses from the original pre-unlearning model. The training runner did not
save that original model, so the default evaluation exports exact efficacy
scores for UNLEARN and intentionally leaves OUT efficacy fields blank. If an
independently preserved, provenance-matched original checkpoint exists, pass it
with `--original-checkpoint` to reproduce OUT efficacy and aggregate efficacy
metrics. The shared `post_npo_pre_final_ft` checkpoint is never used as a
substitute.

Outputs are written under the seed directory's `evaluation/` subdirectory:

- `per_sample_scores.csv`
- `primary_contrast.csv`
- `evaluation_summary.json`

The evaluator refuses to overwrite these outputs and does not train models,
retrain shadows, change thresholds, select post-hoc cohorts, or run other seeds.

## Sequential seeds 43--46

`run_remaining_seeds.py` runs training followed by evaluation for each remaining
seed in order and stops on the first failure. It writes one append-only log per
seed under `results/experiment_2a/logs/`. A seed with all training and evaluation
markers is skipped; partial training or evaluation output is rejected so nothing
is silently rerun or overwritten. Seed 42 is not in the orchestration list.

Run it in the foreground:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/run_remaining_seeds.py \
  --ruli-root /workspace/Ruli \
  --device cuda:0
```

Or launch the same command in a `nohup`-friendly way:

```bash
cd /workspace/Ruli-Experiments
nohup python experiments/experiment_2/run_remaining_seeds.py \
  --ruli-root /workspace/Ruli \
  --device cuda:0 \
  > experiments/experiment_2/results/experiment_2a/remaining_seeds.nohup.log \
  2>&1 &
```

## Experiment 2B: full-history intervention

`run_experiment_2b.py` branches before initial SFT. For each seed it persists
one `initial_shared` GPT-2 state and independently reloads that state for HIGH,
LOW, and PLACEBO. It resets Python, NumPy, PyTorch CPU, and all CUDA RNGs before
each condition and passes the seed to both Hugging Face `seed` and `data_seed`.
It hashes the complete parameter state on three preflight reloads before any
training and again on the actual condition branches before SFT.

Each branch runs the complete upstream RULI pipeline. The frozen condition target
set is injected as follows:

- initial SFT: condition target-IN + common UNLEARN + common WikiText;
- prefix training: condition target-IN + common UNLEARN + common WikiText;
- NPO forget data: common UNLEARN;
- NPO retain data: condition target-IN + common WikiText;
- final retain FT: condition target-IN + common WikiText.

The runner calls the existing RULI helpers without changing their losses,
optimizers, learning rates, batch sizes, or epoch counts. It saves each
condition's post-prefix, pre-NPO checkpoint because that is the correct original
model for reference efficacy evaluation. It does not require condition hashes to
match after SFT, prefix training, NPO, or final FT.

Validate seed 42's manifest, exact target and shadow artifacts, fixed target
partition, condition datasets, 15,000-row background, and initial-state identity
without training or persistent output:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/run_experiment_2b.py \
  --seed 42 \
  --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --device cuda:0 \
  --validate-only
```

Run the complete seed-42 training pipeline explicitly:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/run_experiment_2b.py \
  --seed 42 \
  --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --output-root /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2b/seed_42 \
  --device cuda:0
```

Outputs are isolated under `results/experiment_2b/seed_<SEED>/`: one shared
initial checkpoint, three condition-specific pre-NPO checkpoints, three final
checkpoints, trainer work directories, and `run_metadata.json`. Existing
Experiment 2A outputs are never read as training state or overwritten.

## Experiment 2B: evaluation

`evaluate_experiment_2b.py` reuses the exact fixed-9-shadow validation,
last-seven-token loss, KDE definitions, bounded privacy formula, target
partitions, supported cohort, negative-control cohort, and paired
LOW-minus-PLACEBO logic from the validated Experiment 2A evaluator. For efficacy,
each final model is paired with its own saved pre-NPO model; a post-NPO model is
never substituted.

Evaluate seed 42:

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/evaluate_experiment_2b.py \
  --seed 42 \
  --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --experiment-output /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2b/seed_42 \
  --device cuda:0
```

The primary outcome remains
`privacy_log_odds_LOW - privacy_log_odds_PLACEBO` on the same 28 supported
samples, with preregistered direction `LOW < PLACEBO`. The same 121-sample
negative-control cohort is reported. The evaluator writes the same three output
formats below the seed's `evaluation/` directory and performs no cross-seed
statistics or outcome interpretation.

## Sequential Experiment 2B seeds 42--46

`run_experiment_2b_seeds.py` runs training followed by evaluation for seeds 42,
43, 44, 45, and 46 in order. It skips complete seeds, rejects partial outputs,
logs each seed separately, and stops at the first failure. It is never launched
automatically.

```bash
cd /workspace/Ruli-Experiments
python experiments/experiment_2/run_experiment_2b_seeds.py \
  --ruli-root /workspace/Ruli \
  --manifest /workspace/Ruli-Experiments/experiments/experiment_2/results/intervention_manifest.json \
  --shadow-path /workspace/Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700 \
  --output-base /workspace/Ruli-Experiments/experiments/experiment_2/results/experiment_2b \
  --device cuda:0
```
