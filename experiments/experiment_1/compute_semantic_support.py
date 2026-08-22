"""Measure direct retained-corpus semantic support for each UNLEARN sample."""

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

import numpy as np


EXPECTED_UNLEARN_ROWS = 200
EXPECTED_RETAIN_ROWS = 15_200
REFERENCE_RETAIN_SHA256 = (
    "69a753cb427bcc4997bd0f4ceddba01d9bfa9b31a6736cc6b3bea1be16e305ee"
)
EXPECTED_RETAIN_SOURCES = {"target_in": 200, "wikitext_attack": 15_000}
TOP_KS = (5, 10, 25)
THRESHOLDS = (0.70, 0.75, 0.80)
TOP_NEIGHBORS = 10
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute all 200 x 15,200 normalized cosine similarities and join "
            "per-UNLEARN semantic-support measurements to ruli_scores.csv."
        )
    )
    parser.add_argument(
        "--ruli-scores",
        type=Path,
        default=RESULTS_DIR / "ruli_scores.csv",
    )
    parser.add_argument(
        "--retained-corpus",
        type=Path,
        default=RESULTS_DIR / "retained_corpus.jsonl",
    )
    parser.add_argument(
        "--unlearn-embeddings",
        type=Path,
        default=RESULTS_DIR / "unlearn_embeddings.npz",
    )
    parser.add_argument(
        "--unlearn-metadata",
        type=Path,
        default=None,
        help="Defaults to <unlearn embeddings stem>.metadata.json.",
    )
    parser.add_argument(
        "--retain-embeddings",
        type=Path,
        default=RESULTS_DIR / "retain_embeddings.npz",
    )
    parser.add_argument(
        "--retain-metadata",
        type=Path,
        default=None,
        help="Defaults to <retain embeddings stem>.metadata.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "unlearn_semantic_support.csv",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to <output stem>.metadata.json.",
    )
    parser.add_argument(
        "--verify-reference-retained-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the retained JSONL SHA-256 from the exact reference export "
            "(default: enabled)."
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


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must contain a JSON object.")
    return dict(value)


def _finite_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {description}: {value!r}")
    return result


def _load_score_rows(
    path: Path,
) -> tuple[list[str], dict[int, dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"RULI score CSV does not exist: {path}")
    required = {
        "sample_id",
        "text",
        "split",
        "privacy_score",
        "efficacy_score",
        "privacy_observed_loss",
        "efficacy_observed_loss",
    }
    unlearn_rows: dict[int, dict[str, str]] = {}
    all_ids: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("RULI score CSV contains duplicate column names.")
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"RULI score CSV is missing: {', '.join(sorted(missing))}"
            )
        for source_row, row in enumerate(reader):
            try:
                sample_id = int(row["sample_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Score CSV row {source_row} has invalid sample_id "
                    f"{row['sample_id']!r}."
                ) from exc
            if sample_id in all_ids:
                raise ValueError(f"Duplicate sample_id in RULI score CSV: {sample_id}")
            all_ids.add(sample_id)
            split = (row["split"] or "").strip().lower()
            if split not in {"unlearn", "out"}:
                raise ValueError(
                    f"Score CSV row {source_row} has unsupported split {split!r}."
                )
            if split != "unlearn":
                continue
            if not row["text"].strip():
                raise ValueError(f"UNLEARN sample {sample_id} has empty text.")
            for field in (
                "privacy_score",
                "efficacy_score",
                "privacy_observed_loss",
                "efficacy_observed_loss",
            ):
                _finite_float(row[field], f"sample {sample_id} {field}")
            unlearn_rows[sample_id] = dict(row)
    if len(unlearn_rows) != EXPECTED_UNLEARN_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_UNLEARN_ROWS} UNLEARN score rows; "
            f"found {len(unlearn_rows)}."
        )
    return fieldnames, unlearn_rows


def _load_retain_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retained corpus does not exist: {path}")
    required = {"row_id", "retained_index", "source", "source_index", "text"}
    rows: dict[str, dict[str, Any]] = {}
    retained_indices: set[int] = set()
    sources: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as input_file:
        for source_row, line in enumerate(input_file):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at retained-corpus line {source_row + 1}."
                ) from exc
            if not isinstance(raw, Mapping):
                raise ValueError(
                    f"Retained-corpus line {source_row + 1} is not an object."
                )
            missing = required.difference(raw)
            if missing:
                raise ValueError(
                    f"Retained-corpus line {source_row + 1} is missing: "
                    f"{', '.join(sorted(missing))}"
                )
            row_id = str(raw["row_id"])
            retained_index = int(raw["retained_index"])
            source = str(raw["source"])
            if not row_id or row_id in rows:
                raise ValueError(f"Empty or duplicate retained row_id: {row_id!r}")
            if retained_index in retained_indices:
                raise ValueError(f"Duplicate retained_index: {retained_index}")
            text = raw["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Retained row {row_id} has invalid text.")
            row = dict(raw)
            row["retained_index"] = retained_index
            row["source_index"] = int(raw["source_index"])
            rows[row_id] = row
            retained_indices.add(retained_index)
            sources[source] += 1
    if len(rows) != EXPECTED_RETAIN_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_RETAIN_ROWS} retained rows; found {len(rows)}."
        )
    if retained_indices != set(range(EXPECTED_RETAIN_ROWS)):
        raise ValueError("retained_index values must be exactly 0..15199.")
    if dict(sources) != EXPECTED_RETAIN_SOURCES:
        raise ValueError(
            "Unexpected retained source counts: "
            f"expected {EXPECTED_RETAIN_SOURCES}, found {dict(sources)}."
        )
    return rows


def _load_npz(path: Path, required_arrays: set[str]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding NPZ does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = required_arrays.difference(archive.files)
        if missing:
            raise ValueError(
                f"Embedding NPZ {path} is missing: {', '.join(sorted(missing))}"
            )
        return {name: np.asarray(archive[name]) for name in required_arrays}


def _validate_embedding_matrix(
    embeddings: np.ndarray, expected_rows: int, description: str
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{description} embeddings must have {expected_rows} rows; "
            f"found shape {embeddings.shape}."
        )
    if embeddings.shape[1] == 0 or not np.isfinite(embeddings).all():
        raise ValueError(f"{description} embeddings are empty or non-finite.")
    if embeddings.dtype.kind != "f":
        raise ValueError(f"{description} embeddings do not have a floating dtype.")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{description} embeddings are not L2-normalized.")


def _validate_metadata(
    metadata: Mapping[str, Any],
    role: str,
    expected_rows: int,
    input_path: Path,
    embedding_path: Path,
) -> None:
    required = {
        "schema_version",
        "artifact_role",
        "record_count",
        "embedding_dimension",
        "embedding_dtype",
        "model_name",
        "model_revision_requested",
        "model_revision_resolved",
        "library_versions",
        "normalized",
        "input_artifact",
        "output_npz_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(
            f"{role} embedding metadata is missing: {', '.join(sorted(missing))}"
        )
    if metadata.get("schema_version") != 2:
        raise ValueError(f"Unsupported {role} embedding metadata schema.")
    if metadata.get("artifact_role") != role:
        raise ValueError(f"Embedding metadata role is not {role!r}.")
    if int(metadata.get("record_count", -1)) != expected_rows:
        raise ValueError(f"Embedding metadata has the wrong {role} row count.")
    if metadata.get("normalized") is not True:
        raise ValueError(f"{role} embedding metadata does not require normalization.")
    input_artifact = metadata.get("input_artifact")
    if not isinstance(input_artifact, Mapping):
        raise ValueError(f"{role} metadata has no input_artifact object.")
    if input_artifact.get("sha256") != _sha256_file(input_path):
        raise ValueError(
            f"{role} source artifact hash differs from embedding metadata."
        )
    if metadata.get("output_npz_sha256") != _sha256_file(embedding_path):
        raise ValueError(f"{role} embedding NPZ hash differs from its metadata.")


def _aligned_unlearn(
    arrays: Mapping[str, np.ndarray], score_rows: Mapping[int, Mapping[str, str]]
) -> tuple[np.ndarray, list[int]]:
    embeddings = arrays["embeddings"]
    _validate_embedding_matrix(embeddings, EXPECTED_UNLEARN_ROWS, "UNLEARN")
    sample_ids = arrays["sample_ids"]
    text_hashes = arrays["text_sha256"]
    if sample_ids.shape != (EXPECTED_UNLEARN_ROWS,) or text_hashes.shape != (
        EXPECTED_UNLEARN_ROWS,
    ):
        raise ValueError("UNLEARN alignment arrays have incorrect shapes.")
    ids = [int(value) for value in sample_ids]
    if len(set(ids)) != EXPECTED_UNLEARN_ROWS:
        raise ValueError("UNLEARN embedding sample_ids are not unique.")
    if set(ids) != set(score_rows):
        raise ValueError("UNLEARN embedding sample_ids do not match the score CSV.")
    for embedding_row, sample_id in enumerate(ids):
        expected_hash = _text_hash(score_rows[sample_id]["text"])
        if str(text_hashes[embedding_row]) != expected_hash:
            raise ValueError(
                f"UNLEARN text hash mismatch for explicit sample_id {sample_id}."
            )
    return embeddings, ids


def _aligned_retain(
    arrays: Mapping[str, np.ndarray], retain_rows: Mapping[str, Mapping[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    embeddings = arrays["embeddings"]
    _validate_embedding_matrix(embeddings, EXPECTED_RETAIN_ROWS, "RETAIN")
    one_dimensional = (
        "row_ids",
        "retained_indices",
        "sources",
        "source_indices",
        "text_sha256",
    )
    for name in one_dimensional:
        if arrays[name].shape != (EXPECTED_RETAIN_ROWS,):
            raise ValueError(f"RETAIN alignment array {name!r} has the wrong shape.")
    row_ids = [str(value) for value in arrays["row_ids"]]
    retained_indices = [int(value) for value in arrays["retained_indices"]]
    if len(set(row_ids)) != EXPECTED_RETAIN_ROWS:
        raise ValueError("RETAIN embedding row_ids are not unique.")
    if len(set(retained_indices)) != EXPECTED_RETAIN_ROWS:
        raise ValueError("RETAIN embedding retained_indices are not unique.")
    if set(row_ids) != set(retain_rows):
        raise ValueError("RETAIN embedding row_ids do not match retained_corpus.jsonl.")

    aligned_rows: list[dict[str, Any]] = []
    for embedding_row, row_id in enumerate(row_ids):
        source_row = retain_rows[row_id]
        expected = {
            "retained_index": int(source_row["retained_index"]),
            "source": str(source_row["source"]),
            "source_index": int(source_row["source_index"]),
            "text_sha256": _text_hash(str(source_row["text"])),
        }
        actual = {
            "retained_index": retained_indices[embedding_row],
            "source": str(arrays["sources"][embedding_row]),
            "source_index": int(arrays["source_indices"][embedding_row]),
            "text_sha256": str(arrays["text_sha256"][embedding_row]),
        }
        if actual != expected:
            raise ValueError(
                f"RETAIN provenance mismatch for explicit row_id {row_id}: "
                f"expected {expected}, found {actual}."
            )
        aligned_rows.append({"row_id": row_id, **expected})
    return embeddings, aligned_rows


def _same_model(unlearn: Mapping[str, Any], retain: Mapping[str, Any]) -> None:
    fields = (
        "model_name",
        "model_revision_requested",
        "model_revision_resolved",
        "embedding_dimension",
        "embedding_dtype",
        "normalized",
        "library_versions",
    )
    differences = [field for field in fields if unlearn.get(field) != retain.get(field)]
    if differences:
        raise ValueError(
            "UNLEARN and RETAIN embeddings were not generated with the same "
            f"configuration; differing fields: {', '.join(differences)}."
        )


def _support_rows(
    similarity: np.ndarray,
    sample_ids: Sequence[int],
    score_rows: Mapping[int, Mapping[str, str]],
    retain_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    retain_indices = np.asarray(
        [int(row["retained_index"]) for row in retain_rows], dtype=np.int64
    )
    for unlearn_index, sample_id in enumerate(sample_ids):
        values = similarity[unlearn_index]
        # Explicit retained_index breaks exact-similarity ties deterministically.
        top_25 = np.lexsort((retain_indices, -values))[: max(TOP_KS)]
        row = dict(score_rows[sample_id])
        row["unlearn_text_sha256"] = _text_hash(row["text"])
        row["maximum_retained_similarity"] = float(values[top_25[0]])
        for top_k in TOP_KS:
            row[f"mean_top_{top_k}_similarity"] = float(
                np.mean(values[top_25[:top_k]], dtype=np.float64)
            )
        for threshold in THRESHOLDS:
            suffix = f"{threshold:.2f}".replace(".", "_")
            mask = values >= threshold
            row[f"retained_neighbor_count_ge_{suffix}"] = int(np.count_nonzero(mask))
            row[f"retained_similarity_sum_ge_{suffix}"] = float(
                np.sum(values[mask], dtype=np.float64)
            )
        for rank, retained_embedding_row in enumerate(
            top_25[:TOP_NEIGHBORS], start=1
        ):
            retained = retain_rows[int(retained_embedding_row)]
            prefix = f"top_{rank}_retained"
            row[f"{prefix}_row_id"] = retained["row_id"]
            row[f"{prefix}_retained_index"] = retained["retained_index"]
            row[f"{prefix}_source"] = retained["source"]
            row[f"{prefix}_source_index"] = retained["source_index"]
            row[f"{prefix}_text_sha256"] = retained["text_sha256"]
            row[f"{prefix}_similarity"] = float(values[retained_embedding_row])
        output_rows.append(row)
    return sorted(output_rows, key=lambda row: int(row["sample_id"]))


def _new_fields() -> list[str]:
    fields = [
        "unlearn_text_sha256",
        "maximum_retained_similarity",
        *(f"mean_top_{top_k}_similarity" for top_k in TOP_KS),
    ]
    for threshold in THRESHOLDS:
        suffix = f"{threshold:.2f}".replace(".", "_")
        fields.extend(
            (
                f"retained_neighbor_count_ge_{suffix}",
                f"retained_similarity_sum_ge_{suffix}",
            )
        )
    for rank in range(1, TOP_NEIGHBORS + 1):
        prefix = f"top_{rank}_retained"
        fields.extend(
            (
                f"{prefix}_row_id",
                f"{prefix}_retained_index",
                f"{prefix}_source",
                f"{prefix}_source_index",
                f"{prefix}_text_sha256",
                f"{prefix}_similarity",
            )
        )
    return fields


def _write_csv(
    path: Path,
    source_fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    new_fields = _new_fields()
    collisions = set(source_fields).intersection(new_fields)
    if collisions:
        raise ValueError(
            "Output semantic-support columns already exist in the score CSV: "
            f"{', '.join(sorted(collisions))}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=[*source_fields, *new_fields])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    unlearn_metadata_path = (
        args.unlearn_metadata
        or args.unlearn_embeddings.with_suffix(".metadata.json")
    )
    retain_metadata_path = args.retain_metadata or args.retain_embeddings.with_suffix(
        ".metadata.json"
    )
    output_metadata_path = args.metadata_output or args.output.with_suffix(
        ".metadata.json"
    )
    paths = (
        args.ruli_scores,
        args.retained_corpus,
        args.unlearn_embeddings,
        unlearn_metadata_path,
        args.retain_embeddings,
        retain_metadata_path,
        args.output,
        output_metadata_path,
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("All input and output artifact paths must be distinct.")

    score_fields, score_rows = _load_score_rows(args.ruli_scores)
    retained_corpus_hash = _sha256_file(args.retained_corpus)
    if (
        args.verify_reference_retained_hash
        and retained_corpus_hash != REFERENCE_RETAIN_SHA256
    ):
        raise ValueError(
            "Retained corpus does not match the reference SHA-256: "
            f"expected {REFERENCE_RETAIN_SHA256}, found {retained_corpus_hash}."
        )
    retain_source_rows = _load_retain_rows(args.retained_corpus)
    unlearn_metadata = _load_json(unlearn_metadata_path, "UNLEARN metadata")
    retain_metadata = _load_json(retain_metadata_path, "RETAIN metadata")
    _validate_metadata(
        unlearn_metadata,
        "unlearn",
        EXPECTED_UNLEARN_ROWS,
        args.ruli_scores,
        args.unlearn_embeddings,
    )
    _validate_metadata(
        retain_metadata,
        "retain",
        EXPECTED_RETAIN_ROWS,
        args.retained_corpus,
        args.retain_embeddings,
    )
    _same_model(unlearn_metadata, retain_metadata)

    unlearn_arrays = _load_npz(
        args.unlearn_embeddings, {"embeddings", "sample_ids", "text_sha256"}
    )
    retain_arrays = _load_npz(
        args.retain_embeddings,
        {
            "embeddings",
            "row_ids",
            "retained_indices",
            "sources",
            "source_indices",
            "text_sha256",
        },
    )
    unlearn_embeddings, sample_ids = _aligned_unlearn(unlearn_arrays, score_rows)
    retain_embeddings, aligned_retain_rows = _aligned_retain(
        retain_arrays, retain_source_rows
    )
    if unlearn_embeddings.shape[1] != retain_embeddings.shape[1]:
        raise ValueError("UNLEARN and RETAIN embedding dimensions differ.")
    for role, embeddings, metadata in (
        ("UNLEARN", unlearn_embeddings, unlearn_metadata),
        ("RETAIN", retain_embeddings, retain_metadata),
    ):
        if embeddings.shape[1] != int(metadata["embedding_dimension"]):
            raise ValueError(f"{role} NPZ dimension differs from its metadata.")
        if str(embeddings.dtype) != metadata["embedding_dtype"]:
            raise ValueError(f"{role} NPZ dtype differs from its metadata.")

    print(
        "[VERIFY] Validated 200 UNLEARN and 15,200 RETAIN embeddings, IDs, "
        "text hashes, normalization, model configuration, and source hashes"
    )
    similarity = unlearn_embeddings @ retain_embeddings.T
    if similarity.shape != (EXPECTED_UNLEARN_ROWS, EXPECTED_RETAIN_ROWS):
        raise ValueError(f"Unexpected similarity shape: {similarity.shape}.")
    if not np.isfinite(similarity).all():
        raise ValueError("Similarity matrix contains NaN or infinite values.")
    rows = _support_rows(
        similarity, sample_ids, score_rows, aligned_retain_rows
    )
    _write_csv(args.output, score_fields, rows)

    input_artifacts = {}
    for name, path in (
        ("ruli_scores", args.ruli_scores),
        ("retained_corpus", args.retained_corpus),
        ("unlearn_embeddings", args.unlearn_embeddings),
        ("unlearn_embedding_metadata", unlearn_metadata_path),
        ("retain_embeddings", args.retain_embeddings),
        ("retain_embedding_metadata", retain_metadata_path),
    ):
        input_artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
    metadata = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "normalized cosine similarity via "
            "unlearn_embeddings @ retain_embeddings.T"
        ),
        "unlearn_count": EXPECTED_UNLEARN_ROWS,
        "retain_count": EXPECTED_RETAIN_ROWS,
        "similarity_count": EXPECTED_UNLEARN_ROWS * EXPECTED_RETAIN_ROWS,
        "embedding_dimension": int(unlearn_embeddings.shape[1]),
        "model_name": unlearn_metadata["model_name"],
        "model_revision_requested": unlearn_metadata.get("model_revision_requested"),
        "model_revision_resolved": unlearn_metadata.get("model_revision_resolved"),
        "normalized": True,
        "reference_retained_sha256": REFERENCE_RETAIN_SHA256,
        "reference_retained_hash_check_enabled": (
            args.verify_reference_retained_hash
        ),
        "top_k_means": list(TOP_KS),
        "thresholds": list(THRESHOLDS),
        "top_neighbors_preserved": TOP_NEIGHBORS,
        "tie_breaker": "ascending explicit retained_index",
        "alignment": (
            "sample_id and row_id joins with per-row text/provenance validation"
        ),
        "library_versions": {"numpy": importlib.metadata.version("numpy")},
        "input_artifacts": input_artifacts,
        "output_csv": str(args.output.resolve()),
        "output_csv_sha256": _sha256_file(args.output),
    }
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metadata_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_file:
        json.dump(metadata, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(f"[INFO] Wrote {len(rows)} aligned rows to {args.output.resolve()}")
    print(f"[VERIFY] Output SHA-256: {metadata['output_csv_sha256']}")
    print(f"[INFO] Wrote provenance metadata to {output_metadata_path.resolve()}")


if __name__ == "__main__":
    main()
