"""Build a semantic-similarity graph from aligned sample embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_THRESHOLD = 0.75
GRAPH_NUMERIC_FIELDS = (
    "original_loss",
    "unlearned_loss",
    "out_shadow_mean",
    "unlearn_shadow_mean",
    "privacy_score",
    "efficacy_score",
    "privacy_label",
    "efficacy_label",
    "loss_change",
    "efficacy_out_shadow_mean",
    "out_shadow_count",
    "unlearn_shadow_count",
    "efficacy_out_shadow_count",
)
INTEGER_GRAPH_FIELDS = {
    "privacy_label",
    "efficacy_label",
    "out_shadow_count",
    "unlearn_shadow_count",
    "efficacy_out_shadow_count",
}


@dataclass(frozen=True)
class EmbeddingArtifact:
    embeddings: np.ndarray
    sample_ids: np.ndarray
    source_rows: np.ndarray
    splits: np.ndarray
    text_sha256: np.ndarray
    renormalized: bool


@dataclass(frozen=True)
class SimilarityEdge:
    source_index: int
    target_index: int
    similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an undirected cosine-similarity graph from embeddings.npz "
            "and verify its alignment with ruli_scores.csv."
        )
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=RESULTS_DIR / "embeddings.npz",
    )
    parser.add_argument(
        "--embedding-metadata",
        type=Path,
        default=None,
        help=(
            "Embedding metadata JSON. By default, infer "
            "<embeddings stem>.metadata.json when it exists."
        ),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=RESULTS_DIR / "ruli_scores.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "similarity_graph.graphml",
    )
    parser.add_argument(
        "--nodes-output",
        type=Path,
        default=None,
        help="Node CSV path (defaults to <output stem>.nodes.csv).",
    )
    parser.add_argument(
        "--edges-output",
        type=Path,
        default=None,
        help="Edge CSV path (defaults to <output stem>.edges.csv).",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Graph metadata JSON path (defaults to <output stem>.metadata.json).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine similarity for an edge (default: 0.75).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Optional maximum nearest neighbors considered per node. Without "
            "this option, calculate the exact threshold graph."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="Rows per similarity block in exact-threshold mode.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for top-k nearest-neighbor mode.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=5_000_000,
        help=(
            "Stop before creating more than this many edges (default: 5M). "
            "Use 0 to disable the guard."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as input_file:
            value = json.load(input_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_embedding_metadata(
    embeddings_path: Path, requested_path: Path | None
) -> tuple[Path | None, dict[str, Any] | None]:
    inferred_path = embeddings_path.with_suffix(".metadata.json")
    metadata_path = requested_path or inferred_path
    if not metadata_path.is_file():
        if requested_path is not None:
            raise FileNotFoundError(
                f"Embedding metadata does not exist: {metadata_path}"
            )
        print(
            f"[WARNING] No embedding metadata found at {metadata_path}; "
            "provenance checks will be limited."
        )
        return None, None
    return metadata_path, _load_json(metadata_path)


def _load_embeddings(
    path: Path, metadata: Mapping[str, Any] | None
) -> EmbeddingArtifact:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding artifact does not exist: {path}")
    if metadata is not None:
        expected_hash = metadata.get("output_npz_sha256")
        if expected_hash and expected_hash != _sha256_file(path):
            raise ValueError(
                "Embedding NPZ hash does not match its metadata. The artifact "
                "may have changed after embedding generation."
            )

    required_arrays = {
        "embeddings",
        "sample_ids",
        "source_rows",
        "splits",
        "text_sha256",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = required_arrays.difference(archive.files)
        if missing:
            raise ValueError(
                f"Embedding NPZ is missing arrays: {', '.join(sorted(missing))}"
            )
        embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.int64)
        source_rows = np.asarray(archive["source_rows"], dtype=np.int64)
        splits = np.asarray(archive["splits"], dtype=np.str_)
        text_sha256 = np.asarray(archive["text_sha256"], dtype=np.str_)

    if embeddings.ndim != 2 or embeddings.shape[1] == 0:
        raise ValueError(f"Invalid embedding matrix shape: {embeddings.shape}")
    row_count = embeddings.shape[0]
    aligned_arrays = {
        "sample_ids": sample_ids,
        "source_rows": source_rows,
        "splits": splits,
        "text_sha256": text_sha256,
    }
    for name, values in aligned_arrays.items():
        if values.ndim != 1 or len(values) != row_count:
            raise ValueError(
                f"Array {name!r} is not aligned with {row_count} embedding rows."
            )
    if len(np.unique(sample_ids)) != row_count:
        raise ValueError("Embedding artifact contains duplicate sample IDs.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    if not np.isin(splits, ["unlearn", "out"]).all():
        raise ValueError("Embedding artifact contains an unsupported split.")

    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Embedding artifact contains a zero-length vector.")
    is_normalized = np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)
    metadata_claims_normalized = (
        metadata is not None and metadata.get("normalized") is True
    )
    if metadata_claims_normalized and not is_normalized:
        raise ValueError(
            "Embedding metadata says vectors are normalized, but their norms disagree."
        )
    if not is_normalized:
        embeddings = embeddings / norms[:, None]

    if metadata is not None:
        expected_count = metadata.get("record_count")
        expected_dimension = metadata.get("embedding_dimension")
        if expected_count is not None and int(expected_count) != row_count:
            raise ValueError("Embedding row count does not match its metadata.")
        dimension_disagrees = (
            expected_dimension is not None
            and int(expected_dimension) != embeddings.shape[1]
        )
        if dimension_disagrees:
            raise ValueError("Embedding dimension does not match its metadata.")

    return EmbeddingArtifact(
        embeddings=embeddings,
        sample_ids=sample_ids,
        source_rows=source_rows,
        splits=splits,
        text_sha256=text_sha256,
        renormalized=not is_normalized,
    )


def _load_and_align_score_rows(
    path: Path,
    artifact: EmbeddingArtifact,
    embedding_metadata: Mapping[str, Any] | None,
) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"RULI score CSV does not exist: {path}")
    score_hash = _sha256_file(path)
    if embedding_metadata is not None:
        expected_hash = embedding_metadata.get("input_csv_sha256")
        if expected_hash and expected_hash != score_hash:
            raise ValueError(
                "Score CSV hash does not match the CSV used to build embeddings."
            )

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        required = {"sample_id", "text", "split"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"Score CSV is missing columns: {', '.join(sorted(missing))}"
            )
        reserved = {"embedding_index", "source_row", "text_sha256"}
        collisions = reserved.intersection(fieldnames)
        if collisions:
            raise ValueError(
                "Score CSV uses graph-reserved columns: "
                f"{', '.join(sorted(collisions))}"
            )
        all_rows = list(reader)

    aligned_rows: list[dict[str, str]] = []
    for embedding_index, source_row in enumerate(artifact.source_rows.tolist()):
        if source_row < 0 or source_row >= len(all_rows):
            raise ValueError(
                f"Embedding {embedding_index} references invalid source row "
                f"{source_row}."
            )
        row = all_rows[source_row]
        try:
            csv_sample_id = int(row["sample_id"])
        except (TypeError, ValueError) as exc:
            invalid_id = row["sample_id"]
            raise ValueError(
                f"Score CSV row {source_row} has invalid sample_id {invalid_id!r}."
            ) from exc
        if csv_sample_id != int(artifact.sample_ids[embedding_index]):
            raise ValueError(
                f"Sample-ID mismatch at embedding {embedding_index}: NPZ has "
                f"{artifact.sample_ids[embedding_index]}, CSV has {csv_sample_id}."
            )
        csv_split = (row["split"] or "").strip().lower()
        if csv_split != str(artifact.splits[embedding_index]):
            raise ValueError(
                f"Split mismatch for sample {csv_sample_id}: NPZ has "
                f"{artifact.splits[embedding_index]!r}, CSV has {csv_split!r}."
            )
        csv_text_hash = _text_hash(row["text"] or "")
        if csv_text_hash != str(artifact.text_sha256[embedding_index]):
            raise ValueError(
                f"Text hash mismatch for sample {csv_sample_id}; the score CSV "
                "text differs from the embedded text."
            )
        aligned_rows.append(row)
    return aligned_rows, fieldnames


def _check_edge_limit(edge_count: int, max_edges: int) -> None:
    if max_edges > 0 and edge_count > max_edges:
        raise RuntimeError(
            f"Graph exceeded --max-edges={max_edges:,}. Increase the similarity "
            "threshold or use --top-k to bound graph size."
        )


def _build_exact_threshold_edges(
    embeddings: np.ndarray,
    threshold: float,
    block_size: int,
    max_edges: int,
) -> list[SimilarityEdge]:
    if block_size <= 0:
        raise ValueError("--block-size must be greater than zero.")
    node_count = embeddings.shape[0]
    edges: list[SimilarityEdge] = []
    total_blocks = math.ceil(node_count / block_size)

    for block_number, start in enumerate(
        range(0, node_count, block_size), start=1
    ):
        stop = min(start + block_size, node_count)
        similarities = embeddings[start:stop] @ embeddings.T
        for local_index, source_index in enumerate(range(start, stop)):
            first_target = source_index + 1
            if first_target >= node_count:
                continue
            row = similarities[local_index, first_target:]
            target_indices = np.flatnonzero(row >= threshold) + first_target
            _check_edge_limit(len(edges) + len(target_indices), max_edges)
            for target_index in target_indices.tolist():
                edges.append(
                    SimilarityEdge(
                        source_index=source_index,
                        target_index=target_index,
                        similarity=float(
                            np.clip(similarities[local_index, target_index], -1.0, 1.0)
                        ),
                    )
                )
        print(
            f"[INFO] Similarity block {block_number}/{total_blocks}; "
            f"edges so far: {len(edges):,}",
            flush=True,
        )
    return edges


def _build_top_k_edges(
    embeddings: np.ndarray,
    threshold: float,
    top_k: int,
    n_jobs: int,
    max_edges: int,
) -> list[SimilarityEdge]:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    node_count = embeddings.shape[0]
    if node_count < 2:
        return []

    from sklearn.neighbors import NearestNeighbors

    neighbor_count = min(top_k + 1, node_count)
    search = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="brute",
        metric="cosine",
        n_jobs=n_jobs,
    )
    distances, neighbor_indices = search.fit(embeddings).kneighbors(embeddings)
    edge_by_pair: dict[tuple[int, int], float] = {}

    for source_index in range(node_count):
        accepted = 0
        for distance, target_index_raw in zip(
            distances[source_index], neighbor_indices[source_index]
        ):
            target_index = int(target_index_raw)
            if source_index == target_index:
                continue
            similarity = float(np.clip(1.0 - distance, -1.0, 1.0))
            if similarity < threshold:
                continue
            pair = tuple(sorted((source_index, target_index)))
            edge_by_pair[pair] = max(similarity, edge_by_pair.get(pair, -1.0))
            accepted += 1
            if accepted == top_k:
                break
        _check_edge_limit(len(edge_by_pair), max_edges)

    return [
        SimilarityEdge(source, target, similarity)
        for (source, target), similarity in sorted(edge_by_pair.items())
    ]


def _graph_numeric_attributes(row: Mapping[str, str]) -> dict[str, int | float]:
    attributes: dict[str, int | float] = {}
    for field in GRAPH_NUMERIC_FIELDS:
        raw_value = row.get(field)
        if raw_value is None or not raw_value.strip():
            continue
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value for {field}: {raw_value!r}"
            ) from exc
        if not math.isfinite(value):
            continue
        if field in INTEGER_GRAPH_FIELDS:
            if not value.is_integer():
                raise ValueError(
                    f"Expected an integer value for {field}: {raw_value!r}"
                )
            attributes[field] = int(value)
        else:
            attributes[field] = value
    return attributes


def _build_graph(
    artifact: EmbeddingArtifact,
    score_rows: Sequence[Mapping[str, str]],
    edges: Sequence[SimilarityEdge],
    threshold: float,
    top_k: int | None,
) -> nx.Graph:
    graph = nx.Graph()
    graph.graph.update(
        {
            "similarity_metric": "cosine",
            "similarity_threshold": float(threshold),
            "construction_mode": (
                "top_k_union" if top_k is not None else "exact_threshold"
            ),
            "top_k": int(top_k) if top_k is not None else -1,
        }
    )
    for embedding_index, row in enumerate(score_rows):
        sample_id = int(artifact.sample_ids[embedding_index])
        attributes: dict[str, Any] = {
            "embedding_index": embedding_index,
            "sample_id": sample_id,
            "source_row": int(artifact.source_rows[embedding_index]),
            "split": str(artifact.splits[embedding_index]),
            "text_sha256": str(artifact.text_sha256[embedding_index]),
        }
        attributes.update(_graph_numeric_attributes(row))
        graph.add_node(str(sample_id), **attributes)

    for edge in edges:
        source_id = str(int(artifact.sample_ids[edge.source_index]))
        target_id = str(int(artifact.sample_ids[edge.target_index]))
        graph.add_edge(
            source_id,
            target_id,
            cosine_similarity=float(edge.similarity),
            weight=float(edge.similarity),
        )
    return graph


def _write_node_csv(
    path: Path,
    artifact: EmbeddingArtifact,
    score_rows: Sequence[Mapping[str, str]],
    score_fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["embedding_index", "source_row", "text_sha256", *score_fieldnames]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for embedding_index, row in enumerate(score_rows):
            writer.writerow(
                {
                    "embedding_index": embedding_index,
                    "source_row": int(artifact.source_rows[embedding_index]),
                    "text_sha256": str(artifact.text_sha256[embedding_index]),
                    **row,
                }
            )


def _write_edge_csv(
    path: Path,
    artifact: EmbeddingArtifact,
    edges: Sequence[SimilarityEdge],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source_sample_id",
        "target_sample_id",
        "source_embedding_index",
        "target_embedding_index",
        "cosine_similarity",
    )
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "source_sample_id": int(artifact.sample_ids[edge.source_index]),
                    "target_sample_id": int(artifact.sample_ids[edge.target_index]),
                    "source_embedding_index": edge.source_index,
                    "target_embedding_index": edge.target_index,
                    "cosine_similarity": edge.similarity,
                }
            )


def _similarity_summary(edges: Sequence[SimilarityEdge]) -> dict[str, float | None]:
    if not edges:
        return {"minimum": None, "mean": None, "median": None, "maximum": None}
    values = np.asarray([edge.similarity for edge in edges], dtype=np.float64)
    return {
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
    }


def _validate_args(args: argparse.Namespace, output_paths: Sequence[Path]) -> None:
    if not -1.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between -1 and 1.")
    if args.top_k is not None and args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")
    if args.block_size <= 0:
        raise ValueError("--block-size must be greater than zero.")
    if args.max_edges < 0:
        raise ValueError("--max-edges must be zero or greater.")
    if args.n_jobs == 0:
        raise ValueError("--n-jobs cannot be zero.")
    if args.output.suffix.lower() != ".graphml":
        raise ValueError("--output must have a .graphml suffix.")

    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("All graph output paths must be different.")
    embedding_metadata_input = (
        args.embedding_metadata or args.embeddings.with_suffix(".metadata.json")
    )
    resolved_inputs = {
        args.embeddings.resolve(),
        args.scores.resolve(),
        embedding_metadata_input.resolve(),
    }
    if any(path in resolved_inputs for path in resolved_outputs):
        raise ValueError("Graph output paths must not overwrite input artifacts.")


def main() -> None:
    args = parse_args()
    nodes_output = args.nodes_output or args.output.with_suffix(".nodes.csv")
    edges_output = args.edges_output or args.output.with_suffix(".edges.csv")
    metadata_output = args.metadata_output or args.output.with_suffix(".metadata.json")
    output_paths = [args.output, nodes_output, edges_output, metadata_output]
    _validate_args(args, output_paths)

    embedding_metadata_path, embedding_metadata = _load_embedding_metadata(
        args.embeddings, args.embedding_metadata
    )
    artifact = _load_embeddings(args.embeddings, embedding_metadata)
    score_rows, score_fieldnames = _load_and_align_score_rows(
        args.scores, artifact, embedding_metadata
    )
    print(
        f"[INFO] Loaded {artifact.embeddings.shape[0]:,} aligned embeddings "
        f"with dimension {artifact.embeddings.shape[1]:,}."
    )

    if args.top_k is None:
        print(
            f"[INFO] Building exact graph at cosine similarity >= {args.threshold:.4f}."
        )
        edges = _build_exact_threshold_edges(
            artifact.embeddings,
            args.threshold,
            args.block_size,
            args.max_edges,
        )
        construction_mode = "exact_threshold"
    else:
        print(
            f"[INFO] Building top-{args.top_k} union graph at cosine similarity "
            f">= {args.threshold:.4f}."
        )
        edges = _build_top_k_edges(
            artifact.embeddings,
            args.threshold,
            args.top_k,
            args.n_jobs,
            args.max_edges,
        )
        construction_mode = "top_k_union"

    graph = _build_graph(
        artifact, score_rows, edges, args.threshold, args.top_k
    )
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(
        graph,
        args.output,
        encoding="utf-8",
        prettyprint=True,
        infer_numeric_types=True,
        named_key_ids=True,
    )
    _write_node_csv(nodes_output, artifact, score_rows, score_fieldnames)
    _write_edge_csv(edges_output, artifact, edges)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    possible_edges = node_count * (node_count - 1) / 2
    isolated_count = sum(1 for _, degree in graph.degree() if degree == 0)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "construction_mode": construction_mode,
        "similarity_metric": "cosine",
        "similarity_threshold": args.threshold,
        "top_k": args.top_k,
        "block_size": args.block_size if args.top_k is None else None,
        "node_count": node_count,
        "edge_count": edge_count,
        "density": edge_count / possible_edges if possible_edges else 0.0,
        "isolated_node_count": isolated_count,
        "similarity_summary": _similarity_summary(edges),
        "embeddings_npz": str(args.embeddings.resolve()),
        "embeddings_npz_sha256": _sha256_file(args.embeddings),
        "embedding_metadata": (
            str(embedding_metadata_path.resolve())
            if embedding_metadata_path is not None
            else None
        ),
        "embedding_metadata_sha256": (
            _sha256_file(embedding_metadata_path)
            if embedding_metadata_path is not None
            else None
        ),
        "score_csv": str(args.scores.resolve()),
        "score_csv_sha256": _sha256_file(args.scores),
        "embeddings_renormalized_for_cosine": artifact.renormalized,
        "outputs": {
            "graphml": str(args.output.resolve()),
            "nodes_csv": str(nodes_output.resolve()),
            "edges_csv": str(edges_output.resolve()),
        },
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ("networkx", "numpy", "scikit-learn")
        },
    }
    metadata["output_hashes"] = {
        "graphml_sha256": _sha256_file(args.output),
        "nodes_csv_sha256": _sha256_file(nodes_output),
        "edges_csv_sha256": _sha256_file(edges_output),
    }
    with metadata_output.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")

    print(
        f"[INFO] Wrote graph with {node_count:,} nodes and {edge_count:,} edges "
        f"({isolated_count:,} isolates) to {args.output.resolve()}"
    )
    print(f"[INFO] Wrote node table to {nodes_output.resolve()}")
    print(f"[INFO] Wrote edge table to {edges_output.resolve()}")
    print(f"[INFO] Wrote provenance metadata to {metadata_output.resolve()}")


if __name__ == "__main__":
    main()
