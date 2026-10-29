# GPT-2 and Pythia-160M: first matched shadow-pair findings

Report based on the runs completed on 29 October 2026. This is an interim report for shadow 0, training seed 42, for each model. It is a **fresh matched comparison, not an exact reproduction of the historical shadow run**.

Both models completed the four-stage training pipeline and produced finite original and final losses for all 660 targets. In this pair, loss increased on the UNLEARN group for both models, with a larger within-model increase for Pythia. This establishes that the implemented pipeline can run on both architectures and provides a first descriptive observation. It does not yet establish generalization of the final RULI result, stronger privacy protection, or a reliable model ranking.

The models used identical raw memberships: 660 targets, 15,000 background examples, and 1,617 validation examples. Their tokenizations differ. The same deterministic assignment map gave shadow 0 exactly 240 IN, 222 OUT, and 198 UNLEARN targets. IN targets participate in training and are retained; UNLEARN targets participate in initial training and are subsequently selected for unlearning; OUT targets are excluded from that shadow's target training membership. These shadow-specific roles should not be confused with the fixed target-experiment partitions.

| Training record | GPT-2 | Pythia-160M |
|---|---:|---:|
| Completed shadows | 1 of 9 | 1 of 9 |
| Recorded elapsed time | 26m 11s | 27m 20s |
| Initial SFT | 5 epochs; 4,825 steps | 5 epochs; 4,825 steps |
| Prefix stage | 1 epoch; 3,860 steps | 1 epoch; 3,860 steps |
| NPO configuration | 15 requested epochs | 15 requested epochs |
| NPO actual completion | 13.846 reported epochs; 90 steps | 13.846 reported epochs; 90 steps |
| Final retain fine-tuning | 2 epochs; 1,906 steps | 2 epochs; 1,906 steps |

The NPO distinction matters: these runs must not be described as completing exactly 15 data passes. The matching 90-step budget is consistent with the existing Trainer's gradient-accumulation accounting: 198 forget examples at batch size 16 give 13 batches; accumulation of two produces a planned `floor(13/2) × 15 = 90` optimizer steps. The stage records confirm equal effective optimizer-step budgets in this pair. The current validation checks the requested NPO configuration and requires exactly two completed epochs for final retain fine-tuning; its successful result does not imply exactly 15 completed NPO epochs. This behavior should be explicitly retained or revised before extending the comparison; changing it would constitute a protocol change.

The loss measurements compare the checkpoint after the prefix stage with the checkpoint after **both NPO and final retain fine-tuning**. They therefore do not isolate the effect of NPO. Loss is mean next-token cross-entropy over the approved prediction window. A positive change means the final model assigns less probability, on average, to those observed token continuations.

| Model | Shadow role | Samples | Original mean loss | Final mean loss | Mean change | Median change | Samples with increased loss |
|---|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | IN | 240 | 2.0045 | 1.9580 | -0.0465 | -0.0403 | 83 / 240 (34.6%) |
| GPT-2 | OUT | 222 | 2.3208 | 2.3366 | +0.0157 | +0.0126 | 132 / 222 (59.5%) |
| GPT-2 | UNLEARN | 198 | 1.9625 | 2.0299 | +0.0675 | +0.0496 | 149 / 198 (75.3%) |
| Pythia-160M | IN | 240 | 1.4157 | 1.4781 | +0.0624 | +0.0583 | 144 / 240 (60.0%) |
| Pythia-160M | OUT | 222 | 3.1019 | 3.1031 | +0.0012 | +0.0460 | 119 / 222 (53.6%) |
| Pythia-160M | UNLEARN | 198 | 1.4724 | 2.0158 | +0.5434 | +0.4479 | 187 / 198 (94.4%) |

GPT-2's retained IN group improves slightly on this loss measure, while its UNLEARN group becomes harder to predict. Pythia's UNLEARN group shows a larger shift, while its IN group also becomes slightly harder to predict. Pythia's near-zero mean OUT change coexists with a positive median, so it should not be interpreted as every OUT sample remaining unchanged. These are descriptive patterns from one trained shadow per model. Differences in vocabulary, token boundaries, pretrained weights, and architecture prevent interpreting raw cross-model loss differences as a calibrated privacy or utility ranking. No significance tests or confidence claims are made, and the 660 samples are not 660 independent training replications.

The approved loss rule is `min(7, seq_len - 1)`. All GPT-2 targets have seven predictions. Exactly 25 Pythia targets have six predictions: 11 IN, 10 OUT, and 4 UNLEARN in this shadow assignment. All 28 primary and 121 negative-control samples retain seven predictions for both models. The full exception IDs and per-sample counts are preserved in the summary and CSVs.

As a descriptive sensitivity check, excluding the 25 six-prediction Pythia rows gives mean loss changes of +0.0635 for IN (229 samples), +0.0049 for OUT (212), and +0.5457 for UNLEARN (194). The qualitative Pythia pattern therefore persists in its seven-prediction subset. This check does not replace the approved full-denominator analysis, does not remove exceptions from the stored results, and is not a matched-subset cross-model test.

The recorded environment was Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.39.1, Accelerate 0.28.0, and Datasets 5.0.1 on one RTX PRO 4000 Blackwell GPU. Both models used the existing environment with NumPy 1.26.4 and SciPy 1.14.1; these differ from the archived package list's NumPy 2.1.2 and SciPy 1.18.1. This difference is documented rather than treated as exact historical environment reproduction. The model-specific GPU preflights passed. A separate capacity probe did not complete because a launcher filename collided with Python's `queue` module; the launcher was renamed and its supervisor replaced without restarting the research job. The subsequently completed research stages demonstrate capacity for these actual runs, but no dedicated worst-case peak-memory measurement is available.

The local export contains original loss JSON files, four completion records per model, logs, model-specific preflights, environment details, and the shared assignment map. All 21 downloaded files were checked against their recorded SHA-256 hashes again for this report. The CSV values were checked against the original loss JSON files, all 660 sample IDs and role memberships matched across models, and all 2,640 exported scalar losses were finite. Checkpoints and optimizer scratch remain on the RunPod volume and are not part of this compact local export.

The remaining experimental work is to complete shadows 1–8 for both models, assemble and validate the three observations per sample per role, then run the target interventions and final evaluation. Until that work is complete, there are no new RULI scores, attack AUC/ACC results, or HIGH/LOW/PLACEBO contrasts comparable to the earlier 2B/2C/2D evaluation folders. The observations here support proceeding with a documented matched experiment, while leaving the central model-generalization claim open.

Evidence: [training summary](training_summary.json), [recomputed descriptive statistics](descriptive_statistics.json), [GPT-2 per-sample data](gpt2_shadow_0_per_sample.csv), [Pythia per-sample data](pythia-160m_shadow_0_per_sample.csv), [environment](provenance/environment.json), [launch note](provenance/launch_note.json), and [download checksums](download_checksums.json). Run `python analyze_findings.py` in this directory to reproduce the descriptive statistics and local consistency checks.
