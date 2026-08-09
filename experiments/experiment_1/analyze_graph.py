"""Analyze semantic graph structure against per-sample RULI outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.stats import pearsonr, spearmanr


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TARGETS = (
    "privacy_score",
    "efficacy_score",
    "original_loss",
    "unlearned_loss",
    "loss_change",
)
GRAPH_FEATURES = (
    "degree",
    "weighted_degree",
    "semantic_density",
    "mean_neighbor_similarity",
    "degree_centrality",
    "local_clustering_coefficient",
    "weighted_clustering_coefficient",
    "betweenness_centrality",
    "closeness_centrality",
    "pagerank",
    "core_number",
    "component_size",
    "community_size",
    "internal_degree",
    "external_degree",
    "participation_coefficient",
    "within_community_degree_z",
    "bridge_edge_count",
    "is_bridge_endpoint",
    "same_split_neighbor_fraction",
)
NODE_OUTPUT_FIELDS = (
    "node_id",
    "sample_id",
    "embedding_index",
    "source_row",
    "split",
    "component_id",
    "component_size",
    "community_id",
    "community_size",
    "degree",
    "weighted_degree",
    "semantic_density",
    "mean_neighbor_similarity",
    "degree_centrality",
    "local_clustering_coefficient",
    "weighted_clustering_coefficient",
    "betweenness_centrality",
    "closeness_centrality",
    "pagerank",
    "core_number",
    "internal_degree",
    "external_degree",
    "participation_coefficient",
    "within_community_degree_z",
    "bridge_edge_count",
    "is_bridge_endpoint",
    "unlearn_neighbor_count",
    "out_neighbor_count",
    "same_split_neighbor_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute node/community properties and correlate them with RULI "
            "privacy, efficacy, and loss outcomes."
        )
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=RESULTS_DIR / "similarity_graph.graphml",
    )
    parser.add_argument(
        "--graph-metadata",
        type=Path,
        default=None,
        help=(
            "Graph metadata JSON. By default, infer "
            "<graph stem>.metadata.json when it exists."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "graph_analysis",
    )
    parser.add_argument(
        "--analysis-split",
        choices=("unlearn", "out", "all"),
        default="unlearn",
        help="Node population used for correlations (default: unlearn).",
    )
    parser.add_argument(
        "--target",
        action="append",
        dest="targets",
        help=(
            "Node attribute to correlate with graph features. Repeat for multiple "
            "targets; defaults to the RULI scores and losses."
        ),
    )
    parser.add_argument(
        "--community-resolution",
        type=float,
        default=1.0,
        help="Louvain resolution parameter (default: 1.0).",
    )
    parser.add_argument(
        "--unweighted-communities",
        action="store_true",
        help="Ignore cosine weights during Louvain community detection.",
    )
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        default=None,
        help=(
            "Approximate betweenness using this many pivot nodes. By default, "
            "calculate exact betweenness."
        ),
    )
    parser.add_argument("--minimum-correlation-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_graph_metadata(
    graph_path: Path, requested_path: Path | None
) -> tuple[Path | None, dict[str, Any] | None]:
    inferred_path = graph_path.with_suffix(".metadata.json")
    metadata_path = requested_path or inferred_path
    if not metadata_path.is_file():
        if requested_path is not None:
            raise FileNotFoundError(f"Graph metadata does not exist: {metadata_path}")
        print(
            f"[WARNING] No graph metadata found at {metadata_path}; "
            "provenance checks will be limited."
        )
        return None, None
    return metadata_path, _load_json(metadata_path)


def _as_finite_float(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {description}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric {description}: {value!r}")
    return number


def _load_graph(
    path: Path, metadata: Mapping[str, Any] | None
) -> nx.Graph:
    if not path.is_file():
        raise FileNotFoundError(f"GraphML file does not exist: {path}")
    if metadata is not None:
        expected_hash = (metadata.get("output_hashes") or {}).get("graphml_sha256")
        if expected_hash and expected_hash != _sha256_file(path):
            raise ValueError(
                "GraphML hash does not match its metadata. The graph may have "
                "changed after construction."
            )

    loaded = nx.read_graphml(path)
    if loaded.is_directed():
        raise ValueError("Expected an undirected similarity graph.")
    if loaded.is_multigraph():
        raise ValueError("Expected a simple graph, not a multigraph.")
    graph = nx.Graph(loaded)
    if nx.number_of_selfloops(graph):
        raise ValueError("Similarity graph contains self-edges.")

    sample_ids: set[int] = set()
    for node, attributes in graph.nodes(data=True):
        if "sample_id" not in attributes or "split" not in attributes:
            raise ValueError(
                f"Node {node!r} lacks required sample_id or split attributes."
            )
        sample_id = int(attributes["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in graph: {sample_id}")
        sample_ids.add(sample_id)
        split = str(attributes["split"]).strip().lower()
        if split not in {"unlearn", "out"}:
            raise ValueError(f"Node {node!r} has unsupported split {split!r}.")
        attributes["sample_id"] = sample_id
        attributes["split"] = split

    for source, target, attributes in graph.edges(data=True):
        raw_similarity = attributes.get(
            "cosine_similarity", attributes.get("weight")
        )
        similarity = _as_finite_float(
            raw_similarity, f"similarity on edge ({source}, {target})"
        )
        if not 0.0 <= similarity <= 1.0:
            raise ValueError(
                "Weighted graph analysis requires cosine similarities between "
                f"0 and 1; edge ({source}, {target}) has {similarity}."
            )
        attributes["similarity"] = similarity
        attributes["distance"] = max(1.0 - similarity, 1e-9)
    return graph


def _sorted_groups(groups: Sequence[set[str]], graph: nx.Graph) -> list[set[str]]:
    def group_key(group: set[str]) -> tuple[int, int]:
        minimum_sample_id = min(graph.nodes[node]["sample_id"] for node in group)
        return (-len(group), minimum_sample_id)

    return sorted((set(group) for group in groups), key=group_key)


def _assign_groups(groups: Sequence[set[str]]) -> tuple[dict[str, int], dict[int, int]]:
    membership: dict[str, int] = {}
    sizes: dict[int, int] = {}
    for group_id, group in enumerate(groups):
        sizes[group_id] = len(group)
        for node in group:
            membership[node] = group_id
    return membership, sizes


def _detect_communities(
    graph: nx.Graph,
    resolution: float,
    seed: int,
    weighted: bool,
) -> tuple[list[set[str]], float | None]:
    if resolution <= 0:
        raise ValueError("--community-resolution must be greater than zero.")
    if graph.number_of_nodes() == 0:
        return [], None
    if graph.number_of_edges() == 0:
        return [{node} for node in graph.nodes], None

    weight = "similarity" if weighted else None
    communities = nx.community.louvain_communities(
        graph,
        weight=weight,
        resolution=resolution,
        seed=seed,
    )
    sorted_communities = _sorted_groups(list(communities), graph)
    modularity = nx.community.modularity(
        graph,
        sorted_communities,
        weight=weight,
        resolution=resolution,
    )
    return sorted_communities, float(modularity)


def _compute_centralities(
    graph: nx.Graph,
    betweenness_samples: int | None,
    seed: int,
) -> dict[str, dict[str, float]]:
    node_count = graph.number_of_nodes()
    if betweenness_samples is not None and betweenness_samples <= 0:
        raise ValueError("--betweenness-samples must be greater than zero.")
    if node_count == 0:
        return {
            "degree_centrality": {},
            "betweenness_centrality": {},
            "closeness_centrality": {},
            "pagerank": {},
        }

    sample_count = betweenness_samples
    if sample_count is not None and sample_count >= node_count:
        sample_count = None
    print(
        "[INFO] Computing "
        f"{'exact' if sample_count is None else f'k={sample_count} approximate'} "
        "betweenness centrality..."
    )
    betweenness = nx.betweenness_centrality(
        graph,
        k=sample_count,
        normalized=True,
        weight="distance",
        seed=seed,
    )
    closeness = nx.closeness_centrality(graph, distance="distance")
    pagerank = nx.pagerank(
        graph,
        alpha=0.85,
        weight="similarity",
        max_iter=1_000,
        tol=1e-10,
    )
    return {
        "degree_centrality": nx.degree_centrality(graph),
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,
        "pagerank": pagerank,
    }


def _participation_coefficient(
    neighbor_community_counts: Counter[int], degree: int
) -> float:
    if degree == 0:
        return 0.0
    squared_fractions = sum(
        (count / degree) ** 2
        for count in neighbor_community_counts.values()
    )
    return float(1.0 - squared_fractions)


def _node_metric_rows(
    graph: nx.Graph,
    communities: Sequence[set[str]],
    betweenness_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    components = _sorted_groups(list(nx.connected_components(graph)), graph)
    component_id, component_sizes = _assign_groups(components)
    community_id, community_sizes = _assign_groups(communities)

    degrees = dict(graph.degree())
    weighted_degrees = dict(graph.degree(weight="similarity"))
    clustering = nx.clustering(graph)
    weighted_clustering = nx.clustering(graph, weight="similarity")
    centralities = _compute_centralities(graph, betweenness_samples, seed)
    core_numbers = (
        nx.core_number(graph)
        if graph.number_of_nodes()
        else {}
    )
    bridge_pairs = {frozenset(edge) for edge in nx.bridges(graph)}
    node_count = graph.number_of_nodes()

    internal_degrees: dict[str, int] = {}
    preliminary_rows: list[dict[str, Any]] = []
    for node, attributes in graph.nodes(data=True):
        neighbors = list(graph.neighbors(node))
        similarities = [
            graph.edges[node, neighbor]["similarity"] for neighbor in neighbors
        ]
        neighbor_community_counts = Counter(
            community_id[neighbor] for neighbor in neighbors
        )
        internal_degree = neighbor_community_counts.get(community_id[node], 0)
        internal_degrees[node] = internal_degree
        split_counts = Counter(graph.nodes[neighbor]["split"] for neighbor in neighbors)
        bridge_edge_count = sum(
            1 for neighbor in neighbors if frozenset((node, neighbor)) in bridge_pairs
        )
        degree = degrees[node]
        row: dict[str, Any] = {
            "node_id": node,
            "sample_id": int(attributes["sample_id"]),
            "embedding_index": int(attributes.get("embedding_index", -1)),
            "source_row": int(attributes.get("source_row", -1)),
            "split": attributes["split"],
            "component_id": component_id[node],
            "component_size": component_sizes[component_id[node]],
            "community_id": community_id[node],
            "community_size": community_sizes[community_id[node]],
            "degree": degree,
            "weighted_degree": float(weighted_degrees[node]),
            "semantic_density": (
                float(weighted_degrees[node]) / (node_count - 1)
                if node_count > 1
                else 0.0
            ),
            "mean_neighbor_similarity": (
                float(np.mean(similarities)) if similarities else 0.0
            ),
            "degree_centrality": float(centralities["degree_centrality"][node]),
            "local_clustering_coefficient": float(clustering[node]),
            "weighted_clustering_coefficient": float(weighted_clustering[node]),
            "betweenness_centrality": float(
                centralities["betweenness_centrality"][node]
            ),
            "closeness_centrality": float(centralities["closeness_centrality"][node]),
            "pagerank": float(centralities["pagerank"][node]),
            "core_number": int(core_numbers[node]),
            "internal_degree": internal_degree,
            "external_degree": degree - internal_degree,
            "participation_coefficient": _participation_coefficient(
                neighbor_community_counts, degree
            ),
            "bridge_edge_count": bridge_edge_count,
            "is_bridge_endpoint": int(bridge_edge_count > 0),
            "unlearn_neighbor_count": split_counts.get("unlearn", 0),
            "out_neighbor_count": split_counts.get("out", 0),
            "same_split_neighbor_fraction": (
                split_counts.get(attributes["split"], 0) / degree if degree else 0.0
            ),
        }
        for key, value in attributes.items():
            if key not in row:
                row[key] = value
        preliminary_rows.append(row)

    community_internal_values: dict[int, np.ndarray] = {}
    for current_community_id in range(len(communities)):
        values = [
            internal_degrees[node]
            for node in communities[current_community_id]
        ]
        community_internal_values[current_community_id] = np.asarray(
            values, dtype=np.float64
        )
    for row in preliminary_rows:
        values = community_internal_values[row["community_id"]]
        standard_deviation = float(np.std(values))
        row["within_community_degree_z"] = (
            (row["internal_degree"] - float(np.mean(values))) / standard_deviation
            if standard_deviation > 0
            else 0.0
        )

    return sorted(preliminary_rows, key=lambda row: row["embedding_index"])


def _finite_value(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _benjamini_hochberg(rows: list[dict[str, Any]]) -> None:
    valid_indices = [
        index
        for index, row in enumerate(rows)
        if row["status"] == "ok" and row["p_value"] is not None
    ]
    if not valid_indices:
        return
    ordered = sorted(valid_indices, key=lambda index: rows[index]["p_value"])
    test_count = len(ordered)
    adjusted = [0.0] * test_count
    running_minimum = 1.0
    for reverse_index in range(test_count - 1, -1, -1):
        row_index = ordered[reverse_index]
        rank = reverse_index + 1
        candidate = rows[row_index]["p_value"] * test_count / rank
        running_minimum = min(running_minimum, candidate)
        adjusted[reverse_index] = min(1.0, running_minimum)
    for ordered_index, row_index in enumerate(ordered):
        rows[row_index]["q_value_bh"] = adjusted[ordered_index]


def _correlation_rows(
    node_rows: Sequence[Mapping[str, Any]],
    analysis_split: str,
    targets: Sequence[str],
    minimum_n: int,
) -> list[dict[str, Any]]:
    if minimum_n < 3:
        raise ValueError("--minimum-correlation-n must be at least 3.")
    population = [
        row
        for row in node_rows
        if analysis_split == "all" or row["split"] == analysis_split
    ]
    if not population:
        raise ValueError(f"No nodes match --analysis-split={analysis_split!r}.")
    available_fields = set().union(*(row.keys() for row in population))
    missing_targets = [target for target in targets if target not in available_fields]
    if missing_targets:
        raise ValueError(
            "Requested correlation targets are absent from the selected nodes: "
            f"{', '.join(missing_targets)}"
        )

    results: list[dict[str, Any]] = []
    for target in targets:
        for feature in GRAPH_FEATURES:
            pairs = [
                (feature_value, target_value)
                for row in population
                if (feature_value := _finite_value(row, feature)) is not None
                and (target_value := _finite_value(row, target)) is not None
            ]
            for method in ("spearman", "pearson"):
                result: dict[str, Any] = {
                    "analysis_split": analysis_split,
                    "target": target,
                    "feature": feature,
                    "method": method,
                    "n": len(pairs),
                    "coefficient": None,
                    "p_value": None,
                    "q_value_bh": None,
                    "status": "ok",
                }
                if len(pairs) < minimum_n:
                    result["status"] = "insufficient_n"
                else:
                    feature_values = np.asarray([pair[0] for pair in pairs])
                    target_values = np.asarray([pair[1] for pair in pairs])
                    if np.ptp(feature_values) == 0:
                        result["status"] = "constant_feature"
                    elif np.ptp(target_values) == 0:
                        result["status"] = "constant_target"
                    else:
                        statistic = (
                            spearmanr(feature_values, target_values)
                            if method == "spearman"
                            else pearsonr(feature_values, target_values)
                        )
                        result["coefficient"] = float(statistic.statistic)
                        result["p_value"] = float(statistic.pvalue)
                results.append(result)
    _benjamini_hochberg(results)
    return results


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [value for row in rows if (value := _finite_value(row, field)) is not None]
    return float(np.mean(values)) if values else None


def _community_rows(
    graph: nx.Graph,
    communities: Sequence[set[str]],
    node_rows_by_id: Mapping[str, Mapping[str, Any]],
    targets: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for community_id, community in enumerate(communities):
        subgraph = graph.subgraph(community)
        internal_similarities = [
            attributes["similarity"]
            for _, _, attributes in subgraph.edges(data=True)
        ]
        external_edges = sum(
            1
            for source, target in graph.edges()
            if (source in community) != (target in community)
        )
        node_rows = [node_rows_by_id[node] for node in community]
        row: dict[str, Any] = {
            "community_id": community_id,
            "size": len(community),
            "unlearn_count": sum(row["split"] == "unlearn" for row in node_rows),
            "out_count": sum(row["split"] == "out" for row in node_rows),
            "internal_edge_count": subgraph.number_of_edges(),
            "external_edge_count": external_edges,
            "internal_density": nx.density(subgraph) if len(community) > 1 else 0.0,
            "mean_internal_similarity": (
                float(np.mean(internal_similarities))
                if internal_similarities
                else None
            ),
        }
        for target in targets:
            row[f"mean_{target}"] = _mean_field(node_rows, target)
        rows.append(row)
    return rows


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if array.size == 0:
        return {
            "minimum": None,
            "mean": None,
            "median": None,
            "maximum": None,
            "standard_deviation": None,
        }
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array)),
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace, targets: Sequence[str]) -> None:
    if args.community_resolution <= 0:
        raise ValueError("--community-resolution must be greater than zero.")
    if args.betweenness_samples is not None and args.betweenness_samples <= 0:
        raise ValueError("--betweenness-samples must be greater than zero.")
    if args.minimum_correlation_n < 3:
        raise ValueError("--minimum-correlation-n must be at least 3.")
    if not targets:
        raise ValueError("At least one correlation target is required.")
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate --target values are not allowed.")
    if args.output_dir.resolve() in {
        args.graph.resolve(),
        (args.graph_metadata or args.graph.with_suffix(".metadata.json")).resolve(),
    }:
        raise ValueError("--output-dir must not overwrite an input file.")


def main() -> None:
    args = parse_args()
    targets = tuple(args.targets or DEFAULT_TARGETS)
    _validate_args(args, targets)
    graph_metadata_path, graph_metadata = _load_graph_metadata(
        args.graph, args.graph_metadata
    )
    graph = _load_graph(args.graph, graph_metadata)
    print(
        f"[INFO] Loaded graph with {graph.number_of_nodes():,} nodes and "
        f"{graph.number_of_edges():,} edges."
    )

    communities, modularity = _detect_communities(
        graph,
        args.community_resolution,
        args.seed,
        weighted=not args.unweighted_communities,
    )
    print(f"[INFO] Detected {len(communities):,} communities.")
    node_rows = _node_metric_rows(
        graph, communities, args.betweenness_samples, args.seed
    )
    correlation_rows = _correlation_rows(
        node_rows,
        args.analysis_split,
        targets,
        args.minimum_correlation_n,
    )
    node_rows_by_id = {str(row["node_id"]): row for row in node_rows}
    community_rows = _community_rows(
        graph, communities, node_rows_by_id, targets
    )

    node_output = args.output_dir / "node_metrics.csv"
    correlation_output = args.output_dir / "correlations.csv"
    community_output = args.output_dir / "community_summary.csv"
    summary_output = args.output_dir / "analysis_summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    extra_node_fields = sorted(
        set().union(*(row.keys() for row in node_rows)).difference(NODE_OUTPUT_FIELDS)
    )
    _write_csv(node_output, node_rows, [*NODE_OUTPUT_FIELDS, *extra_node_fields])
    correlation_fields = (
        "analysis_split",
        "target",
        "feature",
        "method",
        "n",
        "coefficient",
        "p_value",
        "q_value_bh",
        "status",
    )
    _write_csv(correlation_output, correlation_rows, correlation_fields)
    community_fields = (
        "community_id",
        "size",
        "unlearn_count",
        "out_count",
        "internal_edge_count",
        "external_edge_count",
        "internal_density",
        "mean_internal_similarity",
        *(f"mean_{target}" for target in targets),
    )
    _write_csv(community_output, community_rows, community_fields)

    components = list(nx.connected_components(graph))
    successful_tests = [row for row in correlation_rows if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_is_correlational": True,
        "analysis_split": args.analysis_split,
        "targets": list(targets),
        "graph_features": list(GRAPH_FEATURES),
        "minimum_correlation_n": args.minimum_correlation_n,
        "multiple_testing_correction": (
            "Benjamini-Hochberg across all successful feature-target-method tests"
        ),
        "community_algorithm": "Louvain",
        "community_resolution": args.community_resolution,
        "community_weight": (
            None if args.unweighted_communities else "cosine similarity"
        ),
        "community_seed": args.seed,
        "community_count": len(communities),
        "modularity": modularity,
        "betweenness_mode": (
            "exact"
            if args.betweenness_samples is None
            or args.betweenness_samples >= graph.number_of_nodes()
            else "approximate"
        ),
        "betweenness_samples": args.betweenness_samples,
        "shortest_path_distance": "max(1 - cosine_similarity, 1e-9)",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "density": nx.density(graph),
        "component_count": len(components),
        "largest_component_size": max(
            (len(component) for component in components), default=0
        ),
        "isolated_node_count": nx.number_of_isolates(graph),
        "degree_summary": _numeric_summary(
            [float(row["degree"]) for row in node_rows]
        ),
        "weighted_degree_summary": _numeric_summary(
            [float(row["weighted_degree"]) for row in node_rows]
        ),
        "successful_correlation_test_count": len(successful_tests),
        "graphml": str(args.graph.resolve()),
        "graphml_sha256": _sha256_file(args.graph),
        "graph_metadata": (
            str(graph_metadata_path.resolve())
            if graph_metadata_path is not None
            else None
        ),
        "graph_metadata_sha256": (
            _sha256_file(graph_metadata_path)
            if graph_metadata_path is not None
            else None
        ),
        "outputs": {
            "node_metrics_csv": str(node_output.resolve()),
            "correlations_csv": str(correlation_output.resolve()),
            "community_summary_csv": str(community_output.resolve()),
        },
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ("networkx", "numpy", "scipy")
        },
    }
    summary["output_hashes"] = {
        "node_metrics_csv_sha256": _sha256_file(node_output),
        "correlations_csv_sha256": _sha256_file(correlation_output),
        "community_summary_csv_sha256": _sha256_file(community_output),
    }
    with summary_output.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"[INFO] Wrote node metrics to {node_output.resolve()}")
    print(f"[INFO] Wrote correlations to {correlation_output.resolve()}")
    print(f"[INFO] Wrote community summary to {community_output.resolve()}")
    print(f"[INFO] Wrote analysis metadata to {summary_output.resolve()}")


if __name__ == "__main__":
    main()
