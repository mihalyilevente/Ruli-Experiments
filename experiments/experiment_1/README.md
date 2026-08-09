# Experiment 1: semantic structure and sample-level RULI

`export_ruli_scores.py` reads existing shadow outputs and saved target-model
checkpoints. It performs inference only; it does not train, unlearn, or modify the
official RULI implementation.

Run it from the repository root:

```bash
python experiments/experiment_1/export_ruli_scores.py \
  --shadow-path ../Ruli/core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target-data-path ../Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed \
  --original-checkpoint ../Ruli/path/to/original-checkpoint \
  --unlearned-checkpoint ../Ruli/path/to/unlearned-checkpoint \
  --device cuda:0
```

All RULI paths are inputs and may point to a sibling checkout, a mounted RunPod
volume, or downloaded artifacts. Nothing is imported from or written into the RULI
source tree.

The default output is `results/ruli_scores.csv` in this directory. The default ID
slices reproduce the official text evaluation: sorted IDs 200--399 are UNLEARN and
400--599 are OUT. Override them with `--unlearn-start`, `--unlearn-count`,
`--out-start`, and `--out-count` when a run used different slices.

The export contains both evaluation groups so its metrics can be checked against the
official attack. For graph correlations on the forget set, filter to
`split == "unlearn"`.

## Score definitions

- `privacy_score`: target loss under the unlearned model, scored against
  `unlearn_unlearned` (positive) and `out_unlearned` (negative) per-sample KDEs.
- `efficacy_score`: the UNLEARN target loss under the unlearned model, or the OUT
  target loss under the original model, scored against `unlearn_unlearned`
  (positive) and `out_original` (negative) per-sample KDEs.
- `out_shadow_mean`: mean of `out_unlearned`, the privacy OUT reference requested
  for the initial CSV schema.
- `efficacy_out_shadow_mean`: mean of `out_original`, included because the official
  efficacy attack uses a different OUT reference.
- `loss_change`: `unlearned_loss - original_loss`.
- `split`: `unlearn` or `out`; both attack labels are 1 for UNLEARN and 0 for OUT.

The script prints AUC, accuracy, TPR@1%FPR, and TPR@5%FPR reconstructed from the
exported scores. These are an immediate check that the CSV matches the saved run.
Per-sample KDE needs at least two non-singular observations in each condition. For
incomplete smoke-test shadow files, `--kde-error nan` exports available values and
writes unavailable scores as NaN instead of stopping.

## Build semantic embeddings

After exporting `results/ruli_scores.csv`, encode every evaluated sample with:

```bash
python experiments/experiment_1/build_embeddings.py \
  --device cuda:0
```

The default model is `sentence-transformers/all-mpnet-base-v2`. Embeddings are
L2-normalized by default so their dot product is cosine similarity. Use `--model`
to select another Sentence Transformers model, and use `--revision` to pin a model
commit for a fully reproducible run.

To embed only the forget samples used for the eventual correlation analysis:

```bash
python experiments/experiment_1/build_embeddings.py \
  --split unlearn \
  --device cuda:0
```

The output `results/embeddings.npz` contains:

- `embeddings`: the float32 embedding matrix;
- `sample_ids`: target-dataset IDs aligned with matrix rows;
- `source_rows`: zero-based rows in `ruli_scores.csv`;
- `splits`: `unlearn` or `out` for each vector;
- `text_sha256`: hashes that detect accidental text/alignment changes.

`results/embeddings.metadata.json` records the model, requested revision, device,
normalization, input/output hashes, dimensions, and library versions.

Important limitation: this embeds only samples present in `ruli_scores.csv`. The
current export contains UNLEARN and OUT evaluation samples, not the full retained
training corpus. A graph built from these vectors measures connectivity within the
evaluation set; it does not yet measure semantic support supplied by all retained
training examples.

## Build the semantic-similarity graph

Construct the exact undirected threshold graph from the aligned embeddings:

```bash
python experiments/experiment_1/build_similarity_graph.py \
  --threshold 0.75
```

Each node is one sample. An edge is included when cosine similarity is greater than
or equal to the threshold. Self-edges are excluded, and each undirected pair is
written once. The exact mode calculates similarities in blocks, avoiding a dense
all-pairs matrix in memory, although its runtime remains quadratic.

For a larger corpus, restrict candidate edges to each node's nearest neighbors:

```bash
python experiments/experiment_1/build_similarity_graph.py \
  --threshold 0.75 \
  --top-k 25
```

Top-k mode takes the union of the qualifying neighbor lists, producing an
undirected graph in which a pair is retained if either endpoint selected the other.
This is not identical to the exact threshold graph unless `k` is sufficiently large.

The graph stage checks the NPZ, embedding metadata, score-CSV hash, sample IDs,
splits, source rows, and text hashes before constructing edges. It writes:

- `similarity_graph.graphml`: the attributed NetworkX graph;
- `similarity_graph.nodes.csv`: aligned RULI fields and source text;
- `similarity_graph.edges.csv`: edge endpoints and cosine similarity;
- `similarity_graph.metadata.json`: construction settings, hashes, counts, density,
  isolate count, similarity summary, and library versions.

The default threshold of `0.75` is only a starting value, not a scientifically
privileged cutoff. Experiment 1 should report sensitivity across multiple thresholds
or use a justified graph-construction rule before interpreting degree or community
effects.

## Analyze graph structure

Compute node properties, Louvain communities, bridge indicators, and correlations:

```bash
python experiments/experiment_1/analyze_graph.py
```

Correlations use UNLEARN nodes by default because those are the samples relevant to
the initial difficulty-of-unlearning hypothesis. Use `--analysis-split all` to
include OUT nodes or `--analysis-split out` for the OUT subset.

For every node, the analysis includes degree, weighted degree, normalized semantic
density, mean neighbor similarity, clustering coefficients, component/community
size, internal/external degree, participation coefficient, within-community degree
z-score, core number, degree/betweenness/closeness centrality, PageRank, bridge-edge
count, and split composition of its neighbors. Similarity is converted to shortest-
path distance as `max(1 - similarity, 1e-9)` for path-based centrality.

The output directory `results/graph_analysis/` contains:

- `node_metrics.csv`: graph properties joined with the RULI node attributes;
- `correlations.csv`: Spearman and Pearson correlations for each graph-feature and
  RULI target pair, including sample size, p-value, status, and Benjamini-Hochberg
  q-value;
- `community_summary.csv`: community composition, density, connectivity, similarity,
  and mean RULI outcomes;
- `analysis_summary.json`: settings, provenance hashes, graph summaries, modularity,
  and output hashes.

Exact betweenness is the default. For a large graph, use
`--betweenness-samples 200` to request a seeded approximation. Community detection
uses cosine edge weights by default; `--unweighted-communities` uses topology only.

These results are correlational. Threshold sensitivity, multiple testing, original
loss, text length, and other confounders must be considered before treating a graph
association as evidence that semantic dependency causes unlearning difficulty.
