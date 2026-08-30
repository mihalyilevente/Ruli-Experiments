# RULI dependency experiments

This repository contains research code for studying whether semantic dependencies
between training samples make NPO/DPO-style machine unlearning less effective. The
official RULI repository remains independent and unchanged; it supplies trained
checkpoints, target datasets, and shadow-model outputs to these experiments.

## Repository boundary

Keep both repositories as siblings:

```text
Research/
├── Ruli/                  # upstream implementation and generated artifacts
└── Ruli-Experiments/      # this repository
```

RULI is not installed or copied into this repository. Experiment scripts accept
paths to its saved artifacts. This avoids relying on RULI's working-directory-based
internal imports and keeps upstream changes separate from research changes.

Experiment 1 includes a small observability-only patch under `patches/`. It adds an
opt-in sample export to the official inference run without changing training,
unlearning, dataset selection, KDE calculations, or attack decisions.

## Recommended RunPod/Linux setup

With the repositories checked out as `/workspace/Ruli` and
`/workspace/Ruli-Experiments`, run:

```bash
cd /workspace/Ruli-Experiments
bash scripts/setup_ruli_env.sh
source /workspace/Ruli/.venv/bin/activate
```

The setup script creates `/workspace/Ruli/.venv` when necessary, reuses an
existing PyTorch installation (including the RunPod base image's CUDA build), and
installs the tested RULI/Experiment 1 dependencies. It preserves
`transformers==4.39.1`, `accelerate==0.28.0`, and
`huggingface-hub==0.25.0`. For Python 3.12 it uses NumPy 1.26.4 instead of the
upstream Python-incompatible `numpy==1.23.5` pin. Set `RULI_ROOT` or
`RULI_VENV_DIR` to override the default locations.

The script finishes with `pip check`, an import check, and a version report that
includes Python, torch, CUDA availability, Transformers, Accelerate, Datasets,
and scikit-learn.

## Minimal local Windows setup

Create a dedicated environment from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

GPU-heavy RULI training and unlearning can still run from the sibling `Ruli`
repository on RunPod. Copy or reference its output paths when running the analysis
scripts here.

## Export the Experiment 1 retained corpus

The retained-corpus export is CPU-only and performs no inference or training. From
this repository, point it at the shadow file from the official run:

```bash
python experiments/experiment_1/export_retain_corpus.py \
  --shadow-path ../Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path /workspace/Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed_smoke_700
```

The reference configuration writes 15,200 aligned JSONL rows (200 target IN rows
followed by 15,000 shuffled WikiText attack rows) plus provenance metadata and
artifact hashes. See the [Experiment 1 instructions](experiments/experiment_1/README.md)
for the exact construction and all path options.

## Experiments

- [Experiment 1](experiments/experiment_1/README.md): export the exact retained
  corpus and per-sample RULI scores, build aligned UNLEARN/RETAIN embeddings, and
  measure and statistically analyze source-specific retained semantic support.
- [Experiment 2](experiments/experiment_2/README.md): run the frozen semantic
  support intervention from one shared post-NPO checkpoint.

Generated data, checkpoints, and results are excluded from version control.
