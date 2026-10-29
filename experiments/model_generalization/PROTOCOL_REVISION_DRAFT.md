# Proposed protocol revision: Pythia model generalization

**Historical draft. Created 2026-10-26; reviewed 2026-10-29.**

The user subsequently approved both amendments and matching assignments for
GPT-2/Pythia. The active implementation is documented in [README.md](README.md)
and `protocol/matched_models_v1/comparison.json`. The proposal below is retained
as history; its approval requests and former blockers are superseded. Full
research training has not been authorized or launched by that implementation.

This document proposes two explicit exceptions to the original request. Existing
validation blocks remain active. It authorizes no training, assignment generation,
data replacement, or changes to Experiments 2A-2D. The existing invalid data bundle
and audit reports remain evidence, not validated research inputs.

## Recommendation

Preserve every raw example, logical sample ID, semantic intervention, and training
stage. Permit the existing RULI short-sequence loss behavior for the exact 25
identified Pythia samples, and generate one new, deterministic nine-shadow
assignment map with honest provenance.

For a comparison with matched shadow assignments, use that same new map to
produce separate GPT-2 and Pythia shadow artifacts, then re-evaluate the existing
GPT-2 target checkpoints if their provenance passes validation. Keep the original
GPT-2 results unchanged. A cheaper Pythia-only run is a defensible exploratory
follow-up, but should not be described as an exact replay of the original shadow
assignments or an isolated causal test of architecture.

Neither route makes architecture the sole changed factor: pretrained weights,
pretraining history, tokenizer, vocabulary, and token boundaries differ too.
Use the claim **model/tokenizer generalization under a documented revised
protocol**, rather than pure architecture isolation.

## Evidence and affected samples

The retrieved reference target files and GPT-2 nine-shadow artifact match the
frozen manifest. Full raw strings were reconstructed for all 660 target IDs and
the same 15,000 background examples. Background and validation token membership
also match the retrieved GPT-2 caches. See [data audit](validation/data_audit.json)
and [reference receipt](validation/reference_receipt.json).

Exactly 25 target strings have seven Pythia tokens, hence six next-token
predictions without an added beginning-of-sequence token:

| Frozen partition | Count | Logical IDs |
|---|---:|---|
| IN | 8 | 52, 59, 62, 74, 75, 117, 138, 175 |
| UNLEARN | 9 | 209, 239, 314, 315, 319, 350, 365, 368, 371 |
| OUT | 7 | 402, 457, 488, 500, 518, 521, 525 |
| Reserve | 1 | 653 |

Thus 24 of the 600 partitioned examples are affected, including 16 of the 400
UNLEARN/OUT examples used in target privacy metrics. Reserve ID 653 remains part
of the 660-example shadow universe even though it is not a selected replacement.

None of these IDs belongs to S, U, P, R, or the 121 negative controls. All 28
primary samples and all 121 negative controls retain seven valid predictions.
This does **not** prove the exception is irrelevant to their results: short IN
and UNLEARN examples still participate in training, and shadow calibration can
affect every evaluated sample.

The local and retrieved RULI training sources match after newline normalization.
Both `MIAEvaluator._batch_inference` and
`LanguageMIA.perform_lm_inference` already set
`ngram_window = min(7, seq_len - 1)`. The original GPT-2 target construction ensured
at least eight tokens, so this shorter window was not needed for its targets.
The new strict eligibility check, rather than lack of evaluator support, blocks
these Pythia examples.

## Amendment A: explicit short-sequence exception

**Proposed selection: retain the current tokenizer and upstream loss code.**

For a model input of length L, evaluate the mean next-token cross-entropy over
the final `k = min(7, L - 1)` prediction positions. Apply this identical rule to
target and shadow inference, before and after unlearning. No BOS, EOS, or padding
is inserted as text content to manufacture an extra prediction.

For this pinned Pythia dataset, allow exactly the 25 IDs listed above to have
L=7 and k=6. Every other target ID must have L>=8 and k=7. Refuse a changed
exception set, a shorter sequence, or any mismatch between the declared and
observed per-ID counts. The GPT-2 reference path continues to require k=7 for
every target. This is a versioned protocol exception, not a general bypass flag.

The last-seven loss implementation and its seven-position cap stay unchanged.
The original requirement that *every* evaluation sample supply exactly seven
predictions is relaxed and must be disclosed. The existing prefix and NPO
objectives, including their current collators and padding behavior, are unchanged;
this proposal makes no new claim that the NPO objective itself is last-seven CE.

Record `input_token_count`, `prediction_token_count`, `short_sequence_exception`,
and the loss-policy version for each target/shadow sample. Export these fields
with raw losses and carry them into evaluation metadata. Keep the full 200
UNLEARN and 200 OUT denominators. Label their AUC/ACC as descriptive metrics that
include 16 six-prediction examples; do not silently report only eligible rows.
Primary and negative-control summaries retain their frozen 28 and 121 samples.
No new statistical tests or outcome-dependent selection are proposed.

Alternatives considered:

| Option | Consequence | Recommendation |
|---|---|---|
| Existing capped window with explicit exceptions | Leaves input content and all membership unchanged; 25 shadow-universe samples use six predictions | Preferred minimal revision |
| Prepend Pythia's BOS uniformly | Supplies an extra conditioning token but changes every input, positions, and potentially the 128-token truncation boundary; requires a separately specified budget policy | Valid alternative only with separate approval and validation |
| Add BOS only to short examples | Introduces a length-dependent input transformation | Not preferred |
| Drop, replace, or exclude short examples | Changes training membership or evaluation denominators and potentially the intervention | Not proposed |

## Amendment B: new documented shadow assignments

The original runner calls Python `random.shuffle` before Trainer construction
without seeding that RNG. Its `--seed` controls background selection. The saved
artifact contains six dictionaries of losses, with three observations per role,
but no shadow IDs. No original assignment map was found in the inspected project
or freeze archive. A newly generated map cannot be represented as recovered
original membership. Later Trainer RNG snapshots are not evidence of the
pre-assignment RNG state, especially after Trainer reseeding.

**Proposed selection: a new assignment map, frozen before any new research run.**

1. Use the unchanged ordered target universe, IDs 0-659.
2. Create a dedicated `random.Random(42)` instance. Keep its assignment seed
   independent of training RNGs and target-run seeds 42-46.
3. For each target ID in ascending order, shuffle shadow IDs 0-8 once. Assign
   the first three to IN, the next three to OUT, and the last three to UNLEARN,
   matching the original assignment algorithm.
4. Save the complete map before training. Store each role's IDs in sorted order
   for canonical serialization, hash the map, and reuse it for all target seeds.
   Persist the Python version and generation algorithm, so the saved map rather
   than a future re-execution is authoritative.
5. If a revised GPT-2 calibration is produced, use this same map for both models.
   Train nine shadows per model and require exactly three finite observations
   per sample and role. Do not impose a new requirement that each shadow have
   exactly 220 examples in each role; the original shuffle does not guarantee it.

The fixed target partition (200 IN / 200 UNLEARN / 200 OUT) remains unchanged.
It is distinct from each shadow's randomized training membership across all 660
targets. The 15,000-example background selection also remains unchanged.

Metadata must explicitly say `assignment_origin: new_deterministic_revision`,
`original_assignment_reproduced: false`, and identify this protocol revision.
Record the old shadow hash as a historical reference only. Never attach new
assignments to the old GPT-2 loss arrays or claim that they describe those arrays.
Save Pythia shadows separately, using the Pythia tokenizer and model. No GPT-2
shadow losses are used to score Pythia checkpoints.

## Choice of comparison scope

| Scope | Work required after approval | Supported interpretation |
|---|---|---|
| Pythia-only exploratory follow-up | Nine new Pythia shadows; initially one target seed, 42 | Tests the same directional semantic-neighbour hypothesis in Pythia under the revised protocol; the historical GPT-2 calibration remains different |
| Matched shadow assignments (recommended for a stronger comparison) | Nine new shadows for each model; one new Pythia target seed initially; validate and re-score existing GPT-2 target checkpoints with the new GPT-2 shadows | Reduces assignment differences between models, while retaining model/tokenization and software-history limitations |

Reusing saved GPT-2 target checkpoints avoids retraining targets only if their
manifest, data identities, seed, stage history, and parameter hashes can be
verified. Re-evaluation outputs belong to the new revision directory. If required
checkpoints or provenance are missing, stop and report the additional work; do
not silently schedule a GPT-2 target rerun. The old results remain historical
evidence. If environment equality cannot be established, disclose that remaining
limitation rather than claiming matching assignments resolve it.

The primary contrast remains `privacy_log_odds_LOW - privacy_log_odds_PLACEBO`
on S, with negative values supporting the hypothesis. Keep HIGH-PLACEBO,
LOW-HIGH, the 121 negative controls, bounded RULI scores, descriptive privacy
AUC/ACC, and per-sample raw losses. Treat numerical effect-size comparisons across
models cautiously: each model has its own loss distributions and KDE calibration.

## Frozen requirements

No change to WikiText source strings, logical IDs, the ordered background,
S=28/U=32/P=32/R=32, semantic thresholds, HIGH/LOW/PLACEBO membership, NPO formula
or beta, optimizer settings, training-stage order, target seeds 42-46, or final
two-epoch retain FT. No semantic reselection or additional shadows per model.
The original 2B early-stopping and best-checkpoint behavior remains intact.

Retain pinned model/tokenizer revisions and the current raw-source revision.
The existing 128-token cap remains unchanged under the preferred short-window
option. Even with identical raw strings, different tokenizers can expose
different text endpoints at that cap; that limitation remains disclosed.

## Implementation and validation after approval

Use a separate revision identifier, proposed `pythia_2b_revision_1`, and separate
artifact/result directories. The original semantic manifest stays immutable;
an amendment manifest references its hash and adds the approved loss policy,
allowlisted IDs, assignment-map hash, comparison scope, and revision status.
The draft itself is not a machine-readable approval token.

Implementation would change data eligibility checks and metadata, assignment
generation/validation, and provenance checks in the new experiment only.
It would not change the upstream loss functions or Experiments 2A-2D.

Before any research training:

1. Verify all original raw identities, IDs, and S/U/P/R memberships again.
2. Verify all 660 expected prediction counts, including exactly the 25 exceptions;
   verify seven predictions for every primary and negative-control sample.
3. Prove target and shadow inference select identical final positions for
   seven-token and longer inputs, with no padding counted as content.
4. Confirm every condition still has 200 target-IN rows and identical background,
   UNLEARN, and OUT membership.
5. Validate all 660 assignment rows, disjoint roles covering nine shadows, three
   observations per role, and byte-identical maps across models if selected.
6. Require an amendment-aware validation report. The old strict failure report
   cannot be relabeled as passed. Re-run bounded compatibility checks only where
   changed preprocessing/provenance paths need them.
7. Prepare a reviewable seed-42 plan and artifact-retention estimate. Do not
   launch full shadows, target training, or all five seeds automatically.

## Decisions requested

The recommended package is Amendment A plus Amendment B, with matched
GPT-2/Pythia shadow assignments. This requires 18 new shadow training runs in
total (nine per model), compared with nine for the exploratory Pythia-only
option. These counts describe future work, not jobs launched or authorized.

- Approve or reject Amendment A: six predictions only for the 25 audited short
  Pythia samples, using the unchanged capped-window implementation.
- Approve or reject Amendment B: a new frozen assignment map with explicit
  non-reproduction provenance.
- Choose Pythia-only exploratory scope or matched GPT-2/Pythia shadow assignments.

Approval to implement a revision is separate from authorization to launch costly
training. Preparing this draft changed documentation only; current strict
validation and all existing artifacts remain unchanged.
