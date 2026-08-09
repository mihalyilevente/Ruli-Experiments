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
└── Ruli-experiments/      # this repository
```

RULI is not installed or copied into this repository. Experiment scripts accept
paths to its saved artifacts. This avoids relying on RULI's working-directory-based
internal imports and keeps upstream changes separate from research changes.

## Setup

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

## Experiments

- [Experiment 1](experiments/experiment_1/README.md): export per-sample RULI scores
  and losses as the input to embedding and semantic-graph analysis.

Generated data, checkpoints, and results are excluded from version control.
