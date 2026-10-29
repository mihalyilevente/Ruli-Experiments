# Model generalization: Pythia-160M

This is a separate, fail-closed implementation of the Experiment 2B full-history
intervention. It imports the existing protocol and scoring helpers without editing
Experiments 2A–2D or the sibling RULI checkout. Full training is opt-in and runs one
seed or one shadow per invocation. Nothing automatically launches five seeds.

## Approved protocol: fresh matched comparison (2026-10-29)

The user approved `min(7, seq_len - 1)`, an explicit log of the 25 Pythia
exceptions, seven predictions for every primary/control sample, and one new
saved assignment map shared by GPT-2 and Pythia shadows. This is a **fresh matched
comparison, not an exact reproduction of the old shadow run**.

- Frozen configuration: [comparison.json](protocol/matched_models_v1/comparison.json).
- Shared map: [assignment_map.json](protocol/matched_models_v1/assignment_map.json).
- Map payload SHA-256:
  `e5cfe50ce1ac8a040d13c440d2ad75c0a0048a4b521e931be69e05b5d6ce29c8`.
- Dedicated assignment RNG: `random.Random(42)`, ascending target IDs 0-659,
  shuffle shadow IDs 0-8, three IN / three OUT / three UNLEARN per sample.
  The saved map is authoritative and cannot be overwritten by the generator.
  Both model paths check its exact hash; a different valid assignment is rejected.
- Both models use the same raw target, 15,000 background, and validation
  memberships. They have separate tokenized data and shadow/checkpoint directories.

Pythia exceptions are exactly IDs
`52, 59, 62, 74, 75, 117, 138, 175, 209, 239, 314, 315, 319, 350, 365, 368,
371, 402, 457, 488, 500, 518, 521, 525, 653`. Each has seven input tokens and six
predictions. All 28 primary and all 121 negative-control samples must still have
seven predictions, for both models. A changed exception set, a shorter example,
invalid mask, or a short protected-cohort sample fails validation. GPT-2 has no
exceptions. No text is added, dropped, or reselected.

Each data bundle contains `target_prediction_counts.json` for all 660 targets.
The same counts and policy are validated during shadow generation and assembly,
and exported with target evaluation rows. Full 200 UNLEARN and 200 OUT metric
denominators are retained; Pythia's descriptive metrics include 16 six-prediction
examples. Primary and negative-control cohorts are unchanged.

The previous strict failed audit and invalid data bundle are preserved under
the old paths. New artifacts use `matched_models_v1`. The
[historical proposal](PROTOCOL_REVISION_DRAFT.md) explains the tradeoffs.
The implementation does not claim to have recovered the old random assignment.

Earlier real Pythia/GPT-2 four-stage smoke runs and a synthetic Pythia shadow run
passed. They are compatibility checks, not research results. The changes here
leave upstream training/loss code and Experiments 2A-2D untouched. No full shadow
or target training is launched automatically. Full research training additionally
requires a successful model-specific preflight report.

Validation of this revision: 70 tests passed; both data bundles and all nine
shared assignment partitions passed the no-training comparison audit. Both
model-specific GPU preflights and research shadow-0 runs subsequently completed
on 29 October 2026. Pythia took about 27 minutes and GPT-2 about 26 minutes.
The Pythia seed-42 shared initial checkpoint and plan are prepared; target
training and shadows 1-8 for both models remain pending.

The compact [shadow-0 findings report](reports/shadow_0/FINDINGS_REPORT.md)
includes the original losses, per-sample CSVs, completion records, logs,
environment details, and verified download checksums. Both NPO runs requested
15 epochs and completed 90 optimizer steps, reported as 13.846 epochs; the
report distinguishes this actual budget from the configured epoch count.
These are interim shadow results, not final RULI scores or intervention contrasts.
Large datasets, model checkpoints, and machine-specific launch scripts remain
local or on the existing RunPod volume and are excluded from Git.

## Architecture audit and implementation

| Existing assumption / behavior | Treatment |
|---|---|
| 2B `MODEL_NAME="gpt2"`; 2C/2D metadata and evaluators require GPT-2 | New runner has `--model-name`; old runners/evaluators remain frozen. |
| GPT-2 data/shadow/checkpoint paths in `utils.load_data`, `target_data.py`, `attack_main.py`, `mia_inference.py`, and Experiment 2 | New data bundles and checkpoints use separate model-specific directories. The new path never calls the GPT-2 cache loader. |
| `load_data` reuses the `gpt2` cache even with another `model_name` | Bypassed; reconstruct raw source membership, then tokenize with the selected tokenizer. |
| Raw text discarded by `target_data.py`; tokenization truncates at 128 | Recover exact source strings through full GPT-2 token-sequence matches. Ambiguous or missing matches fail. Decoded text alone is never raw-source proof. |
| Manifest `text_sha256` hashes GPT-2-decoded text | Verify that hash separately; store full raw-source hashes and new tokenized hashes under different fields. |
| EOS used as PAD | Inspect each model's own special IDs, use its EOS as PAD, preserve right padding and existing collators. No vocabulary resizing or copied GPT-2 IDs. |
| GPT-2 class/module/vocabulary assumptions | The inspected active text paths already use Auto classes. No `GPT2LMHeadModel`, hardcoded vocabulary dimension, `transformer.wte`, or block-name dependency was found there. GPT-2 labels in diagnostic printouts are not a training dependency. |
| Prefix implementation | Existing `train_prefix` calls `PrefixUnlearn(loss_type='gdr')` and updates the causal LM. It is not soft-prefix tuning and accesses no architecture-specific embeddings. Reused unchanged, including padding/label behavior. |
| NPO implementation | Existing full-logit formula `-logsigmoid(0.1*(reference_logits-current_logits)).mean()*2/0.1` is retained. This code's objective is not silently replaced with another NPO formulation. Forward/backward checked against the actual implementation. |
| Last-seven loss | Reuse `MIAEvaluator._batch_inference`; validate the exact approved short-sample exceptions and protect both primary/control cohorts. |
| Shadow inference | Reuse `LanguageMIA.perform_lm_inference`, including its current model-mode behavior. The exact new frozen assignment map is required for both models. |
| Evaluator provenance tied to GPT-2 | New identity checks precede the unchanged 2B KDE scoring and contrast functions. GPT-2 shadows, incomplete shadows, smoke artifacts, and changed checkpoints are rejected. |

The current Pythia release is pinned to
`50f5173d932e8e61f858120bcb800b97af589f46`, resolved from `main` and successfully
loaded locally. The GPT-2 compatibility path is pinned to
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. Model and tokenizer use the same pinned
commit. The old `pythia-160m-v0` name is not supported.

Primary upstream source: [EleutherAI Pythia-160M model card](https://huggingface.co/EleutherAI/pythia-160m).

## Raw identity and comparability

`prepare_data.py` first checks every frozen target storage file against the
manifest. It matches each target token sequence against GPT-2 tokenizations of
the original candidate window: nonblank strings from the first 2,000 WikiText-103
validation rows. Multiple occurrences of the *same string* are recorded; multiple
distinct full strings with the same truncated token sequence fail as ambiguous.

Background `source_index` values index the first 50,000 training rows after the
original GPT-2 nonempty-token filter. The script looks up those exact indices in
the frozen order and verifies every decoded-text hash. It never reselects 15,000
rows. The first 2,500 validation rows retain their original nonempty membership.
The downloaded raw source snapshot is pinned in metadata.

Every raw record stores its logical ID, source index/indices, full text, raw-text
SHA-256, GPT-2 token hash, and GPT-2-decoded text hash. Each split stores an ordered
raw-membership hash, selected-model tokenized hash, and dataset fingerprint.
S/U/P/R and the first-200/next-200/next-200 partition stay frozen.

Research stages remain initial SFT 5 epochs, prefix 1, NPO 15 (beta 0.1), final
retain FT 2. The existing optimizer, batch sizes, accumulation, early stopping,
and best-checkpoint selection remain unchanged. All condition branches reload one
shared initial checkpoint and reset Python, NumPy, PyTorch, and CUDA RNGs to the
requested seed (42–46). Metadata captures effective Trainer arguments and actual
train-end epoch counts; final FT must complete exactly two epochs.

The new evaluator reports all per-sample losses, bounded RULI scores, privacy and
efficacy AUC/ACC, LOW−PLACEBO, HIGH−PLACEBO, and LOW−HIGH on the same 28 supported
samples, with a 121-sample negative-control summary. It adds no statistical tests.

## Deviations and limits

- Tokenizer, vocabulary, architecture, pretrained weights/training corpus, and
  token boundaries necessarily differ. Keeping the 128-token cap means the
  visible text endpoint can differ despite identical complete source strings.
- Smoke tests use synthetic texts, one optimizer step per stage, batch size one,
  and accumulation one. Their metadata explicitly excludes them from research.
- Local smoke uses Python 3.11 / PyTorch 2.5.1+cu118. The preserved cloud freeze
  records Python 3.12.3 / PyTorch 2.8.0+cu128; both use Transformers 4.39.1.
  Reproduce the archived environment for a research run. No environment was
  changed automatically. Source files match the cloud copies after newline
  normalization.
- Shadow training uses explicit recorded Trainer/Python/NumPy/Torch seeds, while
  the old assignment RNG was unrecorded. The new map is deliberately labeled as a
  fresh matched comparison; no claim of exact old-run reproduction is made.
- Full-size batches/long sequences have not been capacity-tested. An OOM must
  be reported; research hyperparameters are not silently reduced.

## Commands

Run from `Ruli-experiments` with the sibling RULI environment activated. Locally,
use `../Ruli/.venv/Scripts/python.exe` in place of `python`. All generated outputs
are protected against overwrite. The frozen shared map is already present; do
not regenerate it when switching models or target seeds.

Prepare each model's data on a fresh checkout (already done locally):

```bash
python experiments/model_generalization/prepare_data.py --model-name gpt2
python experiments/model_generalization/prepare_data.py --model-name EleutherAI/pythia-160m
```

Defaults use the retrieved reference target at
`artifacts/reference_gpt2/selective_dataset_prefixed_smoke_700`; use
`--target-data-path` for another location containing the same frozen files.
Outputs go to `artifacts/matched_models_v1/<model>/data/` and never overwrite the
old strict audit. The raw WikiText revision is pinned, rather than re-resolved
from `main`.

Validate both bundles, cohorts, and all nine shared partitions without training:

```bash
python experiments/model_generalization/validate_matched_comparison.py --report experiments/model_generalization/validation/matched_models_v1/matched_comparison.json
python experiments/model_generalization/train_shadows.py --validate-only --model-name gpt2 --shadow-index 0
python experiments/model_generalization/train_shadows.py --validate-only --model-name EleutherAI/pythia-160m --shadow-index 0
```

Choose a new `--report` path to repeat the already-saved matched comparison audit.
Both shadow commands default to the same `protocol/matched_models_v1/assignment_map.json`.
`--assignments` may point to an exact copy, never a different map.

Full preflight (runs bounded compatibility optimizer steps in disposable models,
not research training; run before launching costly jobs):

```bash
python experiments/model_generalization/run_model_generalization.py --validate-only --model-name gpt2 --device cuda:0
python experiments/model_generalization/run_model_generalization.py --validate-only --model-name EleutherAI/pythia-160m --device cuda:0
```

Each saves its own `validation/matched_models_v1/<model>_preflight.json`.
Research training requires that report for the exact model, data, protocol, and
RULI source hashes. Independent architecture smoke commands remain available:

```bash
python experiments/model_generalization/run_model_generalization.py --smoke-test --device cuda:0
python experiments/model_generalization/train_shadows.py --smoke-test --device cuda:0
```

Future full shadow generation requires explicit execution. This Bash block runs
**18 shadows total**, nine for each model, all using the same saved assignments:

```bash
for model in gpt2 EleutherAI/pythia-160m; do
  for i in 0 1 2 3 4 5 6 7 8; do
    python experiments/model_generalization/train_shadows.py --train --model-name "$model" --shadow-index "$i" --device cuda:0 || exit 1
  done
  python experiments/model_generalization/train_shadows.py --assemble --model-name "$model" || exit 1
done
```

Per-model outputs are isolated under
`artifacts/matched_models_v1/gpt2/shadows/` and
`artifacts/matched_models_v1/pythia-160m/shadows/`. Assembly checks every completed
shadow's exact map hash, protocol label, prediction counts, and data identity,
then requires three finite observations per sample/role and usable KDEs. It saves
model-specific `.pth` files and metadata sidecars. Old GPT-2 shadows are rejected
by the matched pipeline and are never overwritten.

Prepare Pythia seed 42 without training:

```bash
python experiments/model_generalization/run_model_generalization.py --prepare-seed-only --seed 42
```

Future seed-42 training and evaluation (after separate shadow generation):

```bash
python experiments/model_generalization/run_model_generalization.py --train --seed 42 --device cuda:0 --shadow-path experiments/model_generalization/artifacts/matched_models_v1/pythia-160m/shadows/shadow_9_attack_random_npo_pythia160m.pth
python experiments/model_generalization/evaluate_model_generalization.py --seed 42 --device cuda:0 --shadow-path experiments/model_generalization/artifacts/matched_models_v1/pythia-160m/shadows/shadow_9_attack_random_npo_pythia160m.pth
```

Target outputs use `results/matched_models_v1/pythia-160m/seed_42/`. No automatic
five-seed launch exists. The evaluator also accepts `--validate-only`. Re-scoring
historical GPT-2 target checkpoints will require explicit provenance validation
and an import path; do not relabel old run metadata or retrain targets silently.

Tests:

```bash
python -B -m unittest discover -s tests -q
```

## Files and artifacts to retain

New source files: `common.py`, `prepare_data.py`, `training.py`,
`run_model_generalization.py`, `train_shadows.py`, and
`evaluate_model_generalization.py`, `freeze_matched_assignments.py`, and
`validate_matched_comparison.py`. Shared protocol/map files are under
`protocol/matched_models_v1/`. Configuration also includes `model_revisions.json` and
`.gitignore`. Tests: `tests/test_model_generalization.py`. Small audit reports live
in `validation/`; large data/checkpoints are ignored by Git.

Before terminating a future cloud session, retain:

- Original reference target/background caches, raw source snapshot/provenance,
  frozen intervention manifest, original shadow file, and assignment evidence.
- New data bundle, including all raw JSON, tokenized datasets, and metadata.
- Both models' separate shadow `.pth` files, their `.metadata.json`, the shared
  assignment map/configuration, and all eighteen `shadow_<i>/losses.json` records. These are far smaller than model checkpoints.
- Per seed: `initial_shared`, each condition's `pre_npo` and `final`, `plan.json`,
  `run_metadata.json`, completion records, and evaluation CSV/JSON.
- Code revision, `model_revisions.json`, environment freeze, validation reports,
  and logs. Keep Trainer scratch if restart/debugging is needed; it includes
  optimizer states and is much larger than final weights.

Pythia has approximately 162 million parameters: an FP32 weight checkpoint is
about 0.60 GiB, seven target checkpoints about 4.2 GiB per seed, and eighteen
shadow pre/final checkpoints about 10.9 GiB. Trainer scratch adds several times
that amount. The code does not silently delete training checkpoints.
