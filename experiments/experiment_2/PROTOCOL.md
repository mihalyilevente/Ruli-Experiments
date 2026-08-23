# Experiment 2 Protocol

Protocol version: 1.0
Date frozen: 2026-08-23

## 1. Motivation

Experiment 1 tested whether semantic support in the retained training
corpus predicts residual information after NPO unlearning as measured by
RULI.

The primary Experiment 1 observation was source-specific.

Among the 149 non-heading UNLEARN samples:

- semantic support from the 15,000 WikiText attack/retain examples showed
  essentially no association with RULI privacy leakage;
- semantic support from the 200 retained target-IN examples showed a
  positive exploratory association around cosine similarity >= 0.75;
- target-IN neighbor count >= 0.75:
    ordinary Spearman rho = 0.2365, p = 0.00369;
    partial Spearman controlling GPT-2 token count:
    rho = 0.2073, p = 0.01145;
- the effect was stable under word-count and character-length adjustment;
- removing the largest-neighborhood sample (sample 346) gave
  rho = 0.2183, p = 0.00769;
- binary presence of >=1 target-IN neighbor at >=0.75 gave
  Mann-Whitney p = 0.00381;
- however, the source-specific findings did not survive the predefined
  BH/FDR correction.

Therefore Experiment 1 establishes an exploratory association, not a
causal effect.

Experiment 2 tests the corresponding causal hypothesis by deliberately
changing semantic support while holding other aspects of training as
constant as practical.

---

## 2. Frozen Experiment 1 inputs

Experiment 2 must not modify Experiment 1 results.

Ruli-Experiments reference commit:

0482c2fe4fb4271c6cb9d8973b254db853c28250

Git tag:

experiment1-final-2026-08-23

Frozen semantic encoder:

sentence-transformers/all-mpnet-base-v2

Revision:

e8c3b32edf5434bc2275fc9bab85f82640a19130

Frozen retained-corpus SHA-256:

69a753cb427bcc4997bd0f4ceddba01d9bfa9b31a6736cc6b3bea1be16e305ee

Frozen UNLEARN embedding SHA-256:

66b647a9335871cd6373a3f805fdcd81bbf09fe7f4a367a85d0ef44634065b69

Frozen RETAIN embedding SHA-256:

55f9afa6aba700063559107cd38e640ecfe6aa811b0416c18fb9044617893e9b

Frozen Experiment 1 semantic-support CSV SHA-256:

de83299449607bd20220cee059784a172e4647f831c870d7799300bbc8f4c334

Frozen shadow artifact:

shadow_9_attack_random_npo_gpt2.pth

No Experiment 1 threshold, cohort definition, embedding model, or
analysis decision may be changed based on Experiment 2 outcomes.

---

## 3. Research question

Does removing strongly semantically related retained target examples
cause forgotten samples to become less distinguishishable from samples
that were never trained on?

More specifically:

Does removing target-IN examples with cosine similarity >= 0.75 to the
supported UNLEARN samples reduce fixed-shadow RULI privacy evidence
after NPO?

---

## 4. Primary hypothesis

Let S be the set of non-heading Experiment 1 UNLEARN samples with at
least one target-IN neighbor having cosine similarity >= 0.75.

Experiment 1 gives:

|S| = 28.

The primary hypothesis is:

privacy_log_odds_LOW < privacy_log_odds_PLACEBO

for samples in S.

LOW removes semantic target-IN neighbors.

PLACEBO removes an equal number of semantically unrelated target-IN
examples.

A lower RULI privacy log-likelihood ratio means the unlearned sample is
more OUT-like and therefore less distinguishable as previously trained.

---

## 5. Fixed similarity threshold

The intervention threshold is fixed before Experiment 2 training:

cosine similarity >= 0.75

This threshold was selected from the Experiment 1 analysis and must not
be changed after Experiment 2 results are observed.

Thresholds 0.70 and 0.80 may not be substituted as alternative primary
tests.

---

## 6. Target cohort

Primary target cohort S:

1. sample is one of the official 200 UNLEARN target samples;
2. sample is not a WikiText heading;
3. target_in_neighbor_count_ge_0_75 > 0 in the frozen Experiment 1
   support analysis.

Expected size:

28 samples.

A secondary negative-control cohort consists of non-heading UNLEARN
samples with zero target-IN neighbors >= 0.75.

Expected size:

121 samples.

The target cohorts must be determined before any Experiment 2 model is
trained.

---

## 7. Semantic support set U

For every sample s in S, calculate cosine similarity against the
original 200 target-IN embeddings.

Define:

U = union of target-IN examples for which similarity(s, target-IN) >=
0.75 for at least one s in S.

The exact frozen Experiment 1 MPNet embeddings must be used.

U must be generated programmatically and stored in the intervention
manifest before training.

No manual addition or removal of members of U is permitted.

---

## 8. Experimental conditions

Three retain conditions are defined.

### HIGH

The original Experiment 1 retain configuration.

HIGH target-IN set:

original 200 target-IN samples.

### LOW

Remove every target-IN example in U.

Replace them one-for-one with unrelated reserve target examples R.

LOW target-IN set:

(original target-IN - U) + R

### PLACEBO

Remove |U| target-IN examples that are not semantic neighbors of S.

Replace them with exactly the same reserve examples R used by LOW.

PLACEBO target-IN set:

(original target-IN - P) + R

where:

|P| = |U|.

The important primary causal comparison is:

LOW versus PLACEBO.

This controls for the effect of replacing |U| training samples without
specifically removing semantic neighbors.

HIGH remains the original-baseline reference condition.

---

## 9. Placebo set P

P must be selected deterministically from the original target-IN
examples.

Eligible placebo examples must:

- not belong to U;
- not be WikiText headings;
- have maximum cosine similarity < 0.70 to every primary target in S;
- not be exact-text duplicates of primary targets;
- be matched as closely as possible to U on GPT-2 token length.

Matching must be deterministic.

Ties must be resolved by ascending sample ID.

The final P set must be written to the frozen intervention manifest.

---

## 10. Replacement set R

Reserve examples should be selected from target examples outside the
official first 600 evaluation IDs whenever sufficient eligible examples
exist.

The exact ordered target IDs from the shadow artifact must be used
rather than assuming IDs are necessarily contiguous.

Eligible replacements must:

- not be IN, UNLEARN, or OUT evaluation samples;
- not be WikiText headings;
- not exactly duplicate a primary target;
- have maximum cosine similarity < 0.70 to every target in S;
- be approximately token-length matched to the removed support examples.

The same R set must be used in LOW and PLACEBO.

If fewer than |U| eligible reserve examples exist, the manifest builder
must stop with an error.

The replacement rule must not be relaxed automatically.

Any protocol revision required because of insufficient reserve examples
must occur before training and must increment the protocol version.

---

## 11. Unchanged background corpus

The WikiText attack/retain dataset remains exactly the same as in the
reference experiment:

attack_size = 15000
seed = 42 for selecting the reference attack corpus.

No WikiText example may be removed or replaced as part of the semantic
intervention.

This is deliberate because Experiment 1 found no corresponding
WikiText-support association.

---

## 12. Experiment 2A: primary causal experiment

Experiment 2A isolates the effect of semantic support during the final
retain fine-tuning stage.

For each training seed:

1. Train the initial GPT-2 model using the original Experiment 1
   training configuration.
2. Perform NPO using the original Experiment 1 configuration.
3. Save one common post-NPO, pre-FT checkpoint.
4. Clone that exact checkpoint into three branches:
      HIGH
      LOW
      PLACEBO
5. Run the final 2-epoch retain SFT separately on the three
   condition-specific retain datasets.
6. Evaluate all three resulting models identically.

Within a seed, HIGH, LOW, and PLACEBO therefore begin the intervention
from exactly the same model parameters.

This isolates whether semantic target support during post-NPO retain
fine-tuning restores or reinforces information associated with the
forgotten targets.

---

## 13. Experiment 2B: secondary causal experiment

Experiment 2B is performed only after the Experiment 2A protocol and
implementation are frozen.

Experiment 2B tests the broader training-history effect.

The HIGH, LOW, and PLACEBO target-IN configurations are used during:

- initial SFT;
- prefix training;
- the retain set supplied during unlearning;
- final retain FT.

The UNLEARN set remains identical across conditions.

The 15,000-example WikiText attack corpus remains identical across
conditions.

Experiment 2B asks whether semantic support throughout training history,
rather than specifically during post-unlearning retain fine-tuning,
affects resistance to NPO unlearning.

Experiment 2A is primary.
Experiment 2B is secondary.

---

## 14. Training hyperparameters

Unless a documented implementation incompatibility is discovered before
training, retain the reference configuration:

model:
gpt2

initial SFT:
5 epochs

prefix training:
1 epoch

unlearning:
NPO

NPO epochs:
15

post-unlearning retain SFT:
2 epochs

attack corpus:
15000 WikiText examples

The same optimizer configuration and learning rates used by the
reference RULI implementation must be retained.

No hyperparameter tuning based on Experiment 2 outcomes is permitted.

---

## 15. Replication seeds

Predeclared model seeds:

42
43
44
45
46

Within a seed, all conditions must use:

- identical initialization;
- identical common pre-intervention checkpoint where applicable;
- identical random seed reset policy;
- identical training hyperparameters;
- identical validation data.

Condition order must not determine RNG initialization.

Each condition must begin its branch with the same explicitly reset RNG
state.

---

## 16. RULI evaluation

The existing 9-shadow artifact is retained as a fixed scoring reference
for the primary Experiment 2 analysis.

These measurements must therefore be described as:

fixed-shadow RULI measurements

rather than a new exact replication of the complete RULI procedure.

Shadow models are not retrained during the primary experiment.

If Experiment 2 produces a convincing causal effect, matched-shadow
retraining may be performed later as a confirmatory experiment under a
separate protocol.

---

## 17. Primary outcome

The primary outcome is the per-sample privacy KDE log-likelihood ratio.

For observed target loss L:

privacy_log_odds =
    log p(L | UNLEARN-unlearned shadow distribution)
    -
    log p(L | OUT-unlearned shadow distribution)

The same gaussian_kde models and bandwidth behavior as the reference
RULI implementation must be used.

For numerical stability, use KDE logpdf rather than computing very small
densities and then taking their logarithms.

Interpretation:

larger privacy_log_odds
    = more UNLEARN-like
    = stronger residual membership evidence

smaller privacy_log_odds
    = more OUT-like
    = better privacy forgetting

The bounded reference RULI score must also be exported for compatibility.

---

## 18. Secondary outcomes

Secondary outcomes include:

- bounded privacy RULI score;
- efficacy KDE log-likelihood ratio;
- bounded efficacy RULI score;
- post-NPO pre-FT last-7-token loss;
- post-FT last-7-token loss;
- validation loss;
- validation perplexity;
- aggregate RULI AUC/ACC/TPR metrics.

Secondary outcomes do not replace the primary privacy-log-odds
hypothesis.

---

## 19. Primary estimand

For every supported target s and seed r define:

D(s,r) =
    privacy_log_odds_LOW(s,r)
    -
    privacy_log_odds_PLACEBO(s,r)

The causal hypothesis predicts:

D < 0.

The main reported estimand is the average paired LOW-minus-PLACEBO
difference across the supported target cohort and replication seeds.

Report:

- mean paired effect;
- median paired effect;
- effect for each individual seed;
- proportion of supported targets with D < 0;
- 95% uncertainty interval;
- full per-target paired results.

Observations from multiple seeds for the same target must not be treated
as independent samples.

A hierarchical/bootstrap procedure must respect both target and seed
structure.

---

## 20. Robustness/reference contrast

HIGH is not the primary causal control.

It is included to establish whether PLACEBO behaves similarly to the
original training configuration.

The expected pattern under the hypothesis is:

HIGH approximately PLACEBO > LOW

for privacy distinguishability.

A LOW < HIGH result without LOW < PLACEBO is insufficient evidence that
semantic-neighbor removal itself caused the effect.

---

## 21. Negative-control cohort

The 121 non-heading UNLEARN samples having zero target-IN neighbors at
the 0.75 threshold form the negative-control cohort.

They are evaluated using the same trained models.

The expected effect of LOW versus PLACEBO should be substantially weaker
for these samples than for the supported cohort.

This negative-control analysis is secondary.

---

## 22. Intervention manifest

Before any Experiment 2 model training, create and freeze:

experiments/experiment_2/results/intervention_manifest.json

It must contain at least:

- protocol version;
- Experiment 1 reference commit;
- Experiment 1 artifact hashes;
- embedding model name;
- embedding model revision;
- threshold;
- S sample IDs;
- U sample IDs;
- P sample IDs;
- R sample IDs;
- pairwise similarities establishing U;
- maximum S similarity for every P and R example;
- GPT-2 token lengths;
- heading flags;
- exact-text hashes;
- original dataset indices;
- HIGH membership;
- LOW membership;
- PLACEBO membership;
- deterministic matching information;
- hashes of every input artifact;
- hash of the completed manifest.

The manifest must validate all experimental invariants and fail loudly if
any invariant is violated.

---

## 23. Pre-training feasibility gate

No model training may begin until the manifest verifies:

1. |S| = 28;
2. no headings occur in S;
3. every U member has similarity >= 0.75 to at least one S member;
4. no P member has maximum S similarity >= 0.70;
5. no R member has maximum S similarity >= 0.70;
6. |P| = |U|;
7. |R| = |U|;
8. HIGH, LOW, and PLACEBO contain equal numbers of target-IN examples;
9. LOW and PLACEBO use exactly the same R examples;
10. the 15,000 WikiText attack corpus is unchanged;
11. all frozen Experiment 1 artifact hashes match expected values.

Any failure stops execution.

---

## 24. Implementation order

Implementation must proceed in this order:

Phase 1:
protocol only.

Phase 2:
build_intervention_manifest.py only.

Phase 3:
run the manifest generator and manually inspect the resulting cohort and
replacement selections.

Phase 4:
freeze and commit the intervention manifest.

Phase 5:
only then implement Experiment 2A training.

No Experiment 2 training code should be written before Phases 1-4 are
reviewed.

---

## 25. Interpretation rule

Evidence supporting the semantic-support mechanism requires a systematic
paired reduction in privacy distinguishability for LOW relative to
PLACEBO.

A result where HIGH and PLACEBO differ substantially but LOW and
PLACEBO do not is interpreted as generic dataset-perturbation sensitivity,
not evidence for semantic-neighbor causality.

A null LOW-versus-PLACEBO result is evidence against the specific causal
hypothesis tested here, even if Experiment 1's observational correlation
remains present.

Experiment 2 results must not be used to retroactively redefine the
Experiment 1 semantic threshold or target cohort.
