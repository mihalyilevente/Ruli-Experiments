"""Build aligned semantic embeddings from an exported RULI score CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class EmbeddingRecord:
    """One source CSV row selected for embedding."""

    source_row: int
    sample_id: int
    split: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode text from ruli_scores.csv and save sample-aligned semantic "
            "embeddings plus provenance metadata."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "ruli_scores.csv",
        help="RULI score CSV produced by export_ruli_scores.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "embeddings.npz",
        help="Output NPZ containing embeddings and row-alignment arrays.",
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
        "--split",
        action="append",
        choices=("unlearn", "out"),
        dest="splits",
        help=(
            "Only embed this split. Repeat to select both. By default all rows "
            "are embedded."
        ),
    )
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
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(
    input_path: Path, splits: Sequence[str] | None
) -> list[EmbeddingRecord]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    selected_splits = set(splits) if splits else None
    records: list[EmbeddingRecord] = []
    seen_ids: set[int] = set()

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"sample_id", "text", "split"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Input CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for source_row, row in enumerate(reader):
            split = (row["split"] or "").strip().lower()
            if split not in {"unlearn", "out"}:
                raise ValueError(
                    f"CSV data row {source_row} has unsupported split {split!r}."
                )
            if selected_splits is not None and split not in selected_splits:
                continue

            try:
                sample_id = int(row["sample_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV data row {source_row} has invalid sample_id "
                    f"{row['sample_id']!r}."
                ) from exc

            if sample_id in seen_ids:
                raise ValueError(f"Duplicate selected sample_id: {sample_id}")
            seen_ids.add(sample_id)

            text = row["text"] or ""
            if not text.strip():
                raise ValueError(f"Sample {sample_id} has empty text.")
            records.append(
                EmbeddingRecord(
                    source_row=source_row,
                    sample_id=sample_id,
                    split=split,
                    text=text,
                )
            )

    if not records:
        split_description = ", ".join(sorted(selected_splits or [])) or "all"
        raise ValueError(f"No rows matched the requested splits: {split_description}")
    return records


def _encode_records(
    records: Sequence[EmbeddingRecord],
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
    texts = [record.text for record in records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    _validate_embeddings(embeddings, len(records), normalize)
    return embeddings, model


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


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _save_outputs(
    output_path: Path,
    metadata_path: Path,
    records: Sequence[EmbeddingRecord],
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--output must have a .npz suffix.")
    if output_path.resolve() == metadata_path.resolve():
        raise ValueError("Embedding and metadata outputs must be different files.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        sample_ids=np.asarray([record.sample_id for record in records], dtype=np.int64),
        source_rows=np.asarray(
            [record.source_row for record in records], dtype=np.int64
        ),
        splits=np.asarray([record.split for record in records], dtype=np.str_),
        text_sha256=np.asarray(
            [_text_hash(record.text) for record in records], dtype=np.str_
        ),
    )
    metadata["output_npz_sha256"] = _sha256_file(output_path)
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")


def main() -> None:
    args = parse_args()
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    if args.input.resolve() in {args.output.resolve(), metadata_path.resolve()}:
        raise ValueError("Input and output paths must be different.")

    records = _load_records(args.input, args.splits)
    selected_splits = sorted({record.split for record in records})
    print(
        f"[INFO] Loaded {len(records)} rows from {args.input.resolve()} "
        f"(splits: {', '.join(selected_splits)})"
    )
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

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input.resolve()),
        "input_csv_sha256": _sha256_file(args.input),
        "output_npz": str(args.output.resolve()),
        "record_count": len(records),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "selected_splits": selected_splits,
        "model_name": args.model,
        "model_revision_requested": args.revision,
        "model_device": str(model.device),
        "library_versions": {
            package: importlib.metadata.version(package)
            for package in ("sentence-transformers", "torch", "transformers", "numpy")
        },
        "normalized": args.normalize,
        "batch_size": args.batch_size,
        "npz_arrays": {
            "embeddings": "float32 matrix aligned by row",
            "sample_ids": "target-dataset sample ID",
            "source_rows": "zero-based data-row index in the input CSV",
            "splits": "unlearn or out",
            "text_sha256": "SHA-256 of the exact text passed to the encoder",
        },
    }
    _save_outputs(args.output, metadata_path, records, embeddings, metadata)
    print(
        f"[INFO] Wrote {embeddings.shape[0]} x {embeddings.shape[1]} embeddings "
        f"to {args.output.resolve()}"
    )
    print(f"[INFO] Wrote provenance metadata to {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
