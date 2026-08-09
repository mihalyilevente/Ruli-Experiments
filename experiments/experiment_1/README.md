# Experiment 1: semantic structure and sample-level RULI

The score export is captured directly inside the official RULI evaluation. It does
not reload checkpoints, repeat model inference, or recalculate a KDE likelihood
ratio.

## Capture the official per-sample values

Apply the observability-only patch to a clean or compatible RULI checkout:

```bash
git -C ../Ruli apply ../Ruli-experiments/patches/ruli_sample_capture.patch
```

The patch adds one optional argument to `text/mia_inference.py` and copies values
already present in the privacy and efficacy `evaluate_with_kde` loops. It does not
change training, unlearning, dataset selection, inference losses, KDE construction,
likelihood ratios, labels, thresholds, or returned metrics.

Run the same official 9-shadow command that produced the reference metrics, adding
only the export path:

```bash
cd ../Ruli/text
python mia_inference.py \
  --shadow_path ../core/attack/attack_inferences/WikiText103/shadow_9_attack_random_npo_gpt2.pth \
  --target_data_path ./data/WikiText-103-local/gpt2/selective_dataset_prefixed \
  --unlearn_method npo \
  --sft_epochs 5 \
  --unlearn_epochs 15 \
  --prefix_epochs 1 \
  --sample_export_path ../../Ruli-experiments/experiments/experiment_1/results/official_ruli_samples.pth
```

All ordinary arguments must remain identical to the reference run. The new file
stores each evaluated sample ID, observed loss, label, KDE likelihood ratio, and the
positive/negative shadow observations passed to that exact KDE.

## Convert the direct capture to CSV

From the experiment repository, decode the corresponding text and verify metrics:

```bash
python experiments/experiment_1/export_ruli_scores.py \
  --capture-path experiments/experiment_1/results/official_ruli_samples.pth \
  --target-data-path ../Ruli/text/data/WikiText-103-local/gpt2/selective_dataset_prefixed \
  --tokenizer gpt2
```

The default verification requires the exported rows to reproduce the official
9-shadow AUCs at the reported precision: privacy `0.8531` and efficacy `0.8589`.
It also reconstructs all official metrics from the captured labels and likelihood
ratios and requires them to match the metrics returned during the same run. Use
`--no-verify-expected-aucs` only for a deliberately non-reference smoke run.

The output `results/ruli_scores.csv` includes both UNLEARN and OUT groups. The score
columns are the exact captured KDE likelihood ratios. Privacy uses
`unlearn_unlearned` versus `out_unlearned`; efficacy uses `unlearn_unlearned` versus
`out_original`. Each distribution is stored as a JSON array in the CSV, and
`results/ruli_scores.metrics.json` records the verification and artifact hashes.

The official execution does not compute original-model losses for the UNLEARN
samples. Consequently this direct export intentionally does not contain
`original_loss` or `loss_change`; adding them would require extra inference and would
violate the direct-capture requirement.

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

These results are correlational. Threshold sensitivity, multiple testing, observed
loss, text length, and other confounders must be considered before treating a graph
association as evidence that semantic dependency causes unlearning difficulty.
