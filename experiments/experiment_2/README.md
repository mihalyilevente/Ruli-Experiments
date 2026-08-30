# Experiment 2

The frozen protocol and intervention manifest define Experiment 2 before any
training begins. Do not regenerate or edit
`results/intervention_manifest.json` while running Experiment 2A.

## Experiment 2A: seed-42 training

`run_experiment_2a.py` imports the current training helpers from the sibling RULI
checkout. It reproduces the official initial GPT-2 SFT, prefix training, and NPO
stages once. It then saves one shared post-NPO checkpoint and independently loads
that checkpoint for the HIGH, LOW, and PLACEBO two-epoch final retain-SFT stages.

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
  --seed 42 \
  --ruli-root "$RULI_ROOT" \
  --shadow-path "$SHADOW_PATH" \
  --target-data-path "$TARGET_PATH" \
  --device cuda:0
```

Outputs are written under `results/experiment_2a/seed_42/` and ignored by Git.
The runner refuses to overwrite an existing checkpoint or metadata file.

This phase generates training checkpoints only. It does not run KDE/RULI
evaluation, calculate privacy or efficacy, train shadow models, loop over five
seeds, or implement Experiment 2B.
