"""Build reproducible, identifier-aligned Experiment 1 embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
REFERENCE_RETAIN_SHA256 = (
    "69a753cb427bcc4997bd0f4ceddba01d9bfa9b31a6736cc6b3bea1be16e305ee"
)
EXPECTED_UNLEARN_ROWS = 200
EXPECTED_RETAIN_ROWS = 15_200
EXPECTED_RETAIN_SOURCES = {"target_in": 200, "wikitext_attack": 15_000}
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class UnlearnRecord:
    """One explicitly identified UNLEARN score row."""

    source_row: int
    sample_id: int
    text: str


@dataclass(frozen=True)
class RetainRecord:
    """One explicitly identified retained-corpus row."""

    source_row: int
    row_id: str
    retained_index: int
    source: str
    source_index: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode exactly the 200 UNLEARN rows or all 15,200 retained rows "
            "and save identifiers alongside the normalized embeddings."
        )
    )
    parser.add_argument(
        "--corpus",
        choices=("unlearn", "retain"),
        default="unlearn",
        help="Corpus to encode (default: unlearn).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input artifact. Defaults to results/ruli_scores.csv for unlearn "
            "and results/retained_corpus.jsonl for retain."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NPZ. Defaults to results/unlearn_embeddings.npz or "
            "results/retain_embeddings.npz."
        ),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata JSON path (defaults to <output stem>.metadata.json).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence Transformers model name or local path.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/commit for reproducible Hugging Face loading.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Encoding device such as cuda:0, cpu, or mps (default: auto).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize vectors for cosine similarity (default: enabled).",
    )
    parser.add_argument(
        "--show-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--verify-reference-retained-hash",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the retained JSONL SHA-256 from the exact reference export "
            "(default: enabled; only applies to --corpus retain)."
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


def _required_text(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} is empty or is not a string.")
    return value


def _load_unlearn_records(input_path: Path) -> list[UnlearnRecord]:
    if not input_path.is_file():
        raise FileNotFoundError(f"RULI score CSV does not exist: {input_path}")

    records: list[UnlearnRecord] = []
    all_ids: set[int] = set()
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if len(reader.fieldnames or []) != len(set(reader.fieldnames or [])):
            raise ValueError("Input CSV contains duplicate column names.")
        required_columns = {"sample_id", "text", "split"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for source_row, row in enumerate(reader):
            try:
                sample_id = int(row["sample_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV data row {source_row} has invalid sample_id "
                    f"{row['sample_id']!r}."
                ) from exc
            if sample_id in all_ids:
                raise ValueError(f"Duplicate sample_id in score CSV: {sample_id}")
            all_ids.add(sample_id)

            split = (row["split"] or "").strip().lower()
            if split not in {"unlearn", "out"}:
                raise ValueError(
                    f"CSV data row {source_row} has unsupported split {split!r}."
                )
            if split == "unlearn":
                records.append(
                    UnlearnRecord(
                        source_row=source_row,
                        sample_id=sample_id,
                        text=_required_text(row["text"], f"Sample {sample_id} text"),
                    )
                )

    if len(records) != EXPECTED_UNLEARN_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_UNLEARN_ROWS} UNLEARN rows; "
            f"found {len(records)}."
        )
    # Explicit IDs, rather than incidental CSV order, define embedding alignment.
    return sorted(records, key=lambda record: record.sample_id)


def _load_retain_records(input_path: Path) -> list[RetainRecord]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Retained corpus JSONL does not exist: {input_path}")

    records: list[RetainRecord] = []
    seen_row_ids: set[str] = set()
    seen_retained_indices: set[int] = set()
    sources: Counter[str] = Counter()
    required = {"row_id", "retained_index", "source", "source_index", "text"}
    with input_path.open("r", encoding="utf-8") as input_file:
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

            row_id = _required_text(raw["row_id"], "Retained row_id")
            retained_index = int(raw["retained_index"])
            source = _required_text(raw["source"], f"Retained {row_id} source")
            source_index = int(raw["source_index"])
            if row_id in seen_row_ids:
                raise ValueError(f"Duplicate retained row_id: {row_id}")
            if retained_index in seen_retained_indices:
                raise ValueError(
                    f"Duplicate retained_index in retained corpus: {retained_index}"
                )
            seen_row_ids.add(row_id)
            seen_retained_indices.add(retained_index)
            sources[source] += 1
            records.append(
                RetainRecord(
                    source_row=source_row,
                    row_id=row_id,
                    retained_index=retained_index,
                    source=source,
                    source_index=source_index,
                    text=_required_text(raw["text"], f"Retained {row_id} text"),
                )
            )

    if len(records) != EXPECTED_RETAIN_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_RETAIN_ROWS} retained rows; "
            f"found {len(records)}."
        )
    if dict(sources) != EXPECTED_RETAIN_SOURCES:
        raise ValueError(
            "Unexpected retained source counts: "
            f"expected {EXPECTED_RETAIN_SOURCES}, found {dict(sources)}."
        )
    expected_indices = set(range(EXPECTED_RETAIN_ROWS))
    if seen_retained_indices != expected_indices:
        missing = sorted(expected_indices - seen_retained_indices)[:10]
        unexpected = sorted(seen_retained_indices - expected_indices)[:10]
        raise ValueError(
            "retained_index values must be exactly 0..15199; "
            f"missing={missing}, unexpected={unexpected}."
        )
    # Explicit retained_index values, rather than JSONL order, define alignment.
    return sorted(records, key=lambda record: record.retained_index)


def _validate_embeddings(
    embeddings: np.ndarray, expected_rows: int, normalized: bool
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix; received shape {embeddings.shape}."
        )
    if embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} embedding rows; received {embeddings.shape[0]}."
        )
    if embeddings.shape[1] == 0:
        raise ValueError("Embedding vectors have zero dimensions.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")
    if normalized:
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("Model returned vectors that are not L2-normalized.")


def _encode_records(
    records: Sequence[UnlearnRecord | RetainRecord],
    model_name: str,
    revision: str | None,
    device: str | None,
    batch_size: int,
    normalize: bool,
    show_progress_bar: bool,
) -> tuple[np.ndarray, Any]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Run `python -m pip install -e .` "
            "from the repository root."
        ) from exc

    model_kwargs: dict[str, Any] = {}
    if revision is not None:
        model_kwargs["revision"] = revision
    model = SentenceTransformer(model_name, device=device, **model_kwargs)
    embeddings = model.encode(
        [record.text for record in records],
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    _validate_embeddings(embeddings, len(records), normalize)
    return embeddings, model


def _resolved_model_revision(model: Any) -> str | None:
    """Best-effort extraction of the Hub commit recorded by Transformers."""
    try:
        first_module = model[0]
        config = first_module.auto_model.config
        revision = getattr(config, "_commit_hash", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        revision = None
    return str(revision) if revision else None


def _npz_arrays(
    corpus: str,
    records: Sequence[UnlearnRecord | RetainRecord],
    embeddings: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    common = {
        "embeddings": embeddings,
        "source_rows": np.asarray(
            [record.source_row for record in records], dtype=np.int64
        ),
        "text_sha256": np.asarray(
            [_text_hash(record.text) for record in records], dtype=np.str_
        ),
    }
    descriptions = {
        "embeddings": "float32 matrix aligned by the explicit identifier arrays",
        "source_rows": "zero-based physical data-row index in the input artifact",
        "text_sha256": "SHA-256 of the exact text passed to the encoder",
    }
    if corpus == "unlearn":
        if not all(isinstance(record, UnlearnRecord) for record in records):
            raise TypeError("UNLEARN output received a retained record.")
        common["sample_ids"] = np.asarray(
            [
                record.sample_id
                for record in records
                if isinstance(record, UnlearnRecord)
            ],
            dtype=np.int64,
        )
        descriptions["sample_ids"] = "target-dataset sample_id (alignment key)"
    else:
        if not all(isinstance(record, RetainRecord) for record in records):
            raise TypeError("RETAIN output received an UNLEARN record.")
        retain_records = [
            record for record in records if isinstance(record, RetainRecord)
        ]
        common.update(
            {
                "row_ids": np.asarray(
                    [record.row_id for record in retain_records], dtype=np.str_
                ),
                "retained_indices": np.asarray(
                    [record.retained_index for record in retain_records],
                    dtype=np.int64,
                ),
                "sources": np.asarray(
                    [record.source for record in retain_records], dtype=np.str_
                ),
                "source_indices": np.asarray(
                    [record.source_index for record in retain_records],
                    dtype=np.int64,
                ),
            }
        )
        descriptions.update(
            {
                "row_ids": "retained row_id (primary alignment key)",
                "retained_indices": "explicit retained-corpus index",
                "sources": "target_in or wikitext_attack provenance",
                "source_indices": "index within the originating source dataset",
            }
        )
    return common, descriptions


def _save_outputs(
    output_path: Path,
    metadata_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output must have a .npz suffix.")
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("Embedding and metadata outputs must be different files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)
    metadata["output_npz_sha256"] = _sha256_file(output_path)
    with metadata_path.open("w", encoding="utf-8", newline="\n") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


def main() -> None:
    args = parse_args()
    if not args.normalize:
        raise ValueError(
            "Experiment 1 semantic-support embeddings must be L2-normalized."
        )
    default_input = (
        RESULTS_DIR / "ruli_scores.csv"
        if args.corpus == "unlearn"
        else RESULTS_DIR / "retained_corpus.jsonl"
    )
    default_output = RESULTS_DIR / f"{args.corpus}_embeddings.npz"
    input_path = args.input or default_input
    output_path = args.output or default_output
    metadata_path = args.metadata_output or output_path.with_suffix(".metadata.json")
    resolved_paths = {
        input_path.resolve(),
        output_path.resolve(),
        metadata_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError("Input, embedding output, and metadata output must differ.")

    records: Sequence[UnlearnRecord | RetainRecord]
    if args.corpus == "unlearn":
        records = _load_unlearn_records(input_path)
    else:
        retained_hash = _sha256_file(input_path)
        if (
            args.verify_reference_retained_hash
            and retained_hash != REFERENCE_RETAIN_SHA256
        ):
            raise ValueError(
                "Retained corpus does not match the reference SHA-256: "
                f"expected {REFERENCE_RETAIN_SHA256}, found {retained_hash}."
            )
        records = _load_retain_records(input_path)
    print(f"[VERIFY] Loaded {len(records)} explicitly identified {args.corpus} rows")
    print(f"[INFO] Encoding with {args.model!r}...")
    embeddings, model = _encode_records(
        records=records,
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        normalize=args.normalize,
        show_progress_bar=args.show_progress_bar,
    )
    arrays, array_descriptions = _npz_arrays(args.corpus, records, embeddings)
    input_path_resolved = input_path.resolve()
    output_path_resolved = output_path.resolve()
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_role": args.corpus,
        "input_artifact": {
            "path": str(input_path_resolved),
            "sha256": _sha256_file(input_path),
            "bytes": input_path.stat().st_size,
        },
        "output_npz": str(output_path_resolved),
        "record_count": len(records),
        "record_groups": (
            {"unlearn": EXPECTED_UNLEARN_ROWS}
            if args.corpus == "unlearn"
            else EXPECTED_RETAIN_SOURCES
        ),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "model_name": args.model,
        "model_revision_requested": args.revision,
        "model_revision_resolved": _resolved_model_revision(model),
        "model_device": str(model.device),
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ("sentence-transformers", "torch", "transformers", "numpy")
        },
        "normalized": args.normalize,
        "normalization_validation": {"rtol": 1e-4, "atol": 1e-5},
        "batch_size": args.batch_size,
        "alignment": (
            "sorted by sample_id"
            if args.corpus == "unlearn"
            else "sorted by retained_index; row_id is the primary identity"
        ),
        "reference_retained_sha256": (
            REFERENCE_RETAIN_SHA256 if args.corpus == "retain" else None
        ),
        "reference_retained_hash_check_enabled": (
            args.verify_reference_retained_hash if args.corpus == "retain" else None
        ),
        "npz_arrays": array_descriptions,
    }
    _save_outputs(output_path, metadata_path, arrays, metadata)
    print(
        f"[INFO] Wrote {embeddings.shape[0]} x {embeddings.shape[1]} embeddings "
        f"to {output_path_resolved}"
    )
    print(f"[VERIFY] Output SHA-256: {metadata['output_npz_sha256']}")
    print(f"[INFO] Wrote provenance metadata to {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
