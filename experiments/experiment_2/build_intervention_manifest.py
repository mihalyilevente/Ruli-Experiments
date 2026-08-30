"""Build the frozen Experiment 2 semantic-intervention manifest.

This script consumes, but never modifies, the frozen Experiment 1 artifacts.  It
does not import or alter RULI training code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
EXPERIMENT_1_RESULTS = SCRIPT_DIR.parent / "experiment_1" / "results"
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
RESULTS_DIR = SCRIPT_DIR / "results"

PROTOCOL_VERSION = "1.0"
EXPERIMENT_1_REFERENCE_COMMIT = "0482c2fe4fb4271c6cb9d8973b254db853c28250"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
INTERVENTION_THRESHOLD = 0.75
UNRELATED_THRESHOLD = 0.70

EXPECTED_UNLEARN_COUNT = 200
EXPECTED_RETAIN_COUNT = 15_200
EXPECTED_TARGET_IN_COUNT = 200
EXPECTED_WIKITEXT_COUNT = 15_000
EXPECTED_EVALUATION_COUNT = 600
EXPECTED_S_COUNT = 28
EXPECTED_NEGATIVE_CONTROL_COUNT = 121

FROZEN_FILE_SHA256 = {
    "semantic_support_csv": (
        "de83299449607bd20220cee059784a172e4647f831c870d7799300bbc8f4c334"
    ),
    "retained_corpus": (
        "69a753cb427bcc4997bd0f4ceddba01d9bfa9b31a6736cc6b3bea1be16e305ee"
    ),
    "unlearn_embeddings": (
        "66b647a9335871cd6373a3f805fdcd81bbf09fe7f4a367a85d0ef44634065b69"
    ),
    "retain_embeddings": (
        "55f9afa6aba700063559107cd38e640ecfe6aa811b0416c18fb9044617893e9b"
    ),
    "shadow_artifact": (
        "272edae999dfb34cc37e1fdf9bcedf0779959049cf6a919968f1cd6e93c9caf9"
    ),
}
FROZEN_TARGET_DATASET_STORAGE_SHA256 = (
    "5cb5ea1116ac08c537e7f4877f850c9ae71c65ebba0894686aede46db4d3dcd5"
)
FROZEN_TARGET_DATASET_FINGERPRINT = "d4fe55339dd51c18"

HEADING_PATTERN = re.compile(r"\s*(?P<marks>=+)\s+.+?\s+(?P=marks)\s*")


@dataclass(frozen=True)
class Sample:
    """The protocol-relevant identity and text properties of one target row."""

    sample_id: int
    text: str
    token_count: int
    is_heading: bool
    text_sha256: str


@dataclass(frozen=True)
class Match:
    """One deterministic length match from a removed support row to a candidate."""

    source: Sample
    candidate: Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive and validate the frozen Experiment 2 S, U, P, and R sets, "
            "then write the intervention manifest. Protocol thresholds are fixed."
        )
    )
    parser.add_argument(
        "--semantic-support",
        type=Path,
        default=EXPERIMENT_1_RESULTS / "unlearn_semantic_support.csv",
    )
    parser.add_argument(
        "--retained-corpus",
        type=Path,
        default=EXPERIMENT_1_RESULTS / "retained_corpus.jsonl",
    )
    parser.add_argument(
        "--unlearn-embeddings",
        type=Path,
        default=EXPERIMENT_1_RESULTS / "unlearn_embeddings.npz",
    )
    parser.add_argument(
        "--retain-embeddings",
        type=Path,
        default=EXPERIMENT_1_RESULTS / "retain_embeddings.npz",
    )
    parser.add_argument(
        "--target-data-path",
        type=Path,
        default=(
            DEFAULT_RULI_ROOT
            / "text"
            / "data"
            / "WikiText-103-local"
            / "gpt2"
            / "selective_dataset_prefixed_smoke_700"
        ),
    )
    parser.add_argument(
        "--shadow-path",
        type=Path,
        default=(
            DEFAULT_RULI_ROOT
            / "core"
            / "attack"
            / "attack_inferences"
            / "WikiText103"
            / "shadow_9_attack_random_npo_gpt2.pth"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "intervention_manifest.json",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_heading(text: str) -> bool:
    return HEADING_PATTERN.fullmatch(text) is not None


def _required_int(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {description}: {value!r}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    return result


def _token_ids(value: Any, description: str) -> list[int]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid {description} JSON.") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{description} must be an integer array.")
    if not value or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{description} must be a nonempty integer array.")
    return [int(item) for item in value]


def _verify_frozen_file(path: Path, artifact_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{artifact_name} does not exist: {path}")
    actual = _sha256_file(path)
    expected = FROZEN_FILE_SHA256[artifact_name]
    if actual != expected:
        raise ValueError(
            f"{artifact_name} SHA-256 mismatch: expected {expected}, found {actual}. "
            "The frozen protocol does not permit bypassing this check."
        )
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": actual,
    }


def _dataset_storage_manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(f"Target dataset does not exist: {path}")
    files = sorted(
        {
            file
            for pattern in ("data-*.arrow", "dataset_info.json", "state.json")
            for file in path.glob(pattern)
            if file.is_file()
        },
        key=lambda file: file.name,
    )
    if not files:
        raise ValueError(f"No saved Hugging Face dataset files found in {path}")
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for file in files:
        file_hash = _sha256_file(file)
        relative_name = file.relative_to(path).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append(
            {
                "path": relative_name,
                "bytes": file.stat().st_size,
                "sha256": file_hash,
            }
        )
    storage_hash = digest.hexdigest()
    if storage_hash != FROZEN_TARGET_DATASET_STORAGE_SHA256:
        raise ValueError(
            "target_dataset storage SHA-256 mismatch: expected "
            f"{FROZEN_TARGET_DATASET_STORAGE_SHA256}, found {storage_hash}. "
            "The frozen protocol does not permit bypassing this check."
        )
    return {
        "path": str(path.resolve()),
        "storage_sha256": storage_hash,
        "files": entries,
    }


def _load_semantic_support(path: Path) -> dict[int, dict[str, str]]:
    required = {
        "sample_id",
        "text",
        "token_ids",
        "split",
        "unlearn_text_sha256",
        "gpt2_token_count",
        "is_wikitext_heading",
        "target_in_neighbor_count_ge_0_75",
    }
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError("Semantic-support CSV contains duplicate columns.")
        missing = required.difference(fields)
        if missing:
            raise ValueError(
                "Semantic-support CSV is missing: " + ", ".join(sorted(missing))
            )
        for source_row, raw in enumerate(reader, start=2):
            sample_id = _required_int(raw["sample_id"], f"row {source_row} sample_id")
            if sample_id in rows:
                raise ValueError(f"Duplicate semantic-support sample_id {sample_id}.")
            if (raw["split"] or "").strip().lower() != "unlearn":
                raise ValueError(
                    f"Semantic-support sample {sample_id} is not in UNLEARN."
                )
            text = raw["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Semantic-support sample {sample_id} has empty text.")
            ids = _token_ids(raw["token_ids"], f"sample {sample_id} token_ids")
            token_count = _required_int(
                raw["gpt2_token_count"], f"sample {sample_id} GPT-2 token count"
            )
            if token_count != len(ids):
                raise ValueError(
                    f"Semantic-support sample {sample_id} token count disagrees "
                    "with token_ids."
                )
            expected_hash = _sha256_text(text)
            if raw["unlearn_text_sha256"] != expected_hash:
                raise ValueError(
                    f"Semantic-support sample {sample_id} text hash is invalid."
                )
            heading = _required_int(
                raw["is_wikitext_heading"], f"sample {sample_id} heading flag"
            )
            if heading not in {0, 1} or bool(heading) != _is_heading(text):
                raise ValueError(
                    f"Semantic-support sample {sample_id} heading flag is invalid."
                )
            count = _required_int(
                raw["target_in_neighbor_count_ge_0_75"],
                f"sample {sample_id} target-IN neighbor count",
            )
            if count < 0 or count > EXPECTED_TARGET_IN_COUNT:
                raise ValueError(
                    f"Semantic-support sample {sample_id} has invalid neighbor count."
                )
            rows[sample_id] = dict(raw)
    if len(rows) != EXPECTED_UNLEARN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_UNLEARN_COUNT} semantic-support rows; "
            f"found {len(rows)}."
        )
    return rows


def _load_retained_corpus(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "row_id",
        "retained_index",
        "source",
        "source_index",
        "text",
        "token_ids",
    }
    by_row_id: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    sources: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as input_file:
        for source_line, line in enumerate(input_file, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on retained line {source_line}."
                ) from exc
            if not isinstance(raw, Mapping):
                raise ValueError(f"Retained line {source_line} is not an object.")
            missing = required.difference(raw)
            if missing:
                raise ValueError(
                    f"Retained line {source_line} is missing: "
                    + ", ".join(sorted(missing))
                )
            row_id = str(raw["row_id"])
            retained_index = _required_int(
                raw["retained_index"], f"retained row {row_id} index"
            )
            if not row_id or row_id in by_row_id:
                raise ValueError(f"Empty or duplicate retained row_id {row_id!r}.")
            if retained_index in by_index:
                raise ValueError(f"Duplicate retained_index {retained_index}.")
            source = str(raw["source"])
            source_index = _required_int(
                raw["source_index"], f"retained row {row_id} source_index"
            )
            text = raw["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Retained row {row_id} has empty text.")
            token_ids = _token_ids(raw["token_ids"], f"retained row {row_id} token_ids")
            row = dict(raw)
            row.update(
                retained_index=retained_index,
                source=source,
                source_index=source_index,
                text=text,
                token_ids=token_ids,
            )
            by_row_id[row_id] = row
            by_index[retained_index] = row
            sources[source] += 1
    if len(by_row_id) != EXPECTED_RETAIN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RETAIN_COUNT} retained rows; found {len(by_row_id)}."
        )
    if set(by_index) != set(range(EXPECTED_RETAIN_COUNT)):
        raise ValueError("retained_index values must be exactly 0..15199.")
    expected_sources = {
        "target_in": EXPECTED_TARGET_IN_COUNT,
        "wikitext_attack": EXPECTED_WIKITEXT_COUNT,
    }
    if dict(sources) != expected_sources:
        raise ValueError(
            f"Unexpected retained source counts: expected {expected_sources}, "
            f"found {dict(sources)}."
        )
    return by_row_id, [by_index[index] for index in range(EXPECTED_RETAIN_COUNT)]


def _load_npz(path: Path, required: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"Embedding NPZ {path} is missing: {', '.join(sorted(missing))}"
            )
        return {name: np.asarray(archive[name]) for name in required}


def _validate_matrix(matrix: np.ndarray, rows: int, description: str) -> None:
    if matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0:
        raise ValueError(
            f"{description} embedding matrix has invalid shape {matrix.shape}."
        )
    if matrix.dtype.kind != "f" or not np.isfinite(matrix).all():
        raise ValueError(
            f"{description} embedding matrix is not finite floating point."
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise ValueError(f"{description} embeddings are not L2-normalized.")


def _align_unlearn_embeddings(
    arrays: Mapping[str, np.ndarray], support_rows: Mapping[int, Mapping[str, str]]
) -> tuple[np.ndarray, list[int]]:
    matrix = arrays["embeddings"]
    _validate_matrix(matrix, EXPECTED_UNLEARN_COUNT, "UNLEARN")
    sample_ids = arrays["sample_ids"]
    text_hashes = arrays["text_sha256"]
    if sample_ids.shape != (EXPECTED_UNLEARN_COUNT,) or text_hashes.shape != (
        EXPECTED_UNLEARN_COUNT,
    ):
        raise ValueError("UNLEARN embedding alignment arrays have invalid shapes.")
    ids = [int(value) for value in sample_ids]
    if len(set(ids)) != EXPECTED_UNLEARN_COUNT or set(ids) != set(support_rows):
        raise ValueError("UNLEARN embedding IDs do not match semantic-support IDs.")
    for index, sample_id in enumerate(ids):
        expected = _sha256_text(support_rows[sample_id]["text"])
        if str(text_hashes[index]) != expected:
            raise ValueError(f"UNLEARN embedding text hash mismatch for {sample_id}.")
    return matrix, ids


def _align_retain_embeddings(
    arrays: Mapping[str, np.ndarray], retained: Mapping[str, Mapping[str, Any]]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = arrays["embeddings"]
    _validate_matrix(matrix, EXPECTED_RETAIN_COUNT, "RETAIN")
    one_dimensional = (
        "row_ids",
        "retained_indices",
        "sources",
        "source_indices",
        "text_sha256",
    )
    for name in one_dimensional:
        if arrays[name].shape != (EXPECTED_RETAIN_COUNT,):
            raise ValueError(f"RETAIN embedding array {name!r} has invalid shape.")
    row_ids = [str(value) for value in arrays["row_ids"]]
    if len(set(row_ids)) != EXPECTED_RETAIN_COUNT or set(row_ids) != set(retained):
        raise ValueError("RETAIN embedding row_ids do not match retained corpus.")
    aligned: list[dict[str, Any]] = []
    for index, row_id in enumerate(row_ids):
        row = retained[row_id]
        expected = {
            "retained_index": int(row["retained_index"]),
            "source": str(row["source"]),
            "source_index": int(row["source_index"]),
            "text_sha256": _sha256_text(str(row["text"])),
        }
        actual = {
            "retained_index": int(arrays["retained_indices"][index]),
            "source": str(arrays["sources"][index]),
            "source_index": int(arrays["source_indices"][index]),
            "text_sha256": str(arrays["text_sha256"][index]),
        }
        if actual != expected:
            raise ValueError(f"RETAIN embedding provenance mismatch for {row_id}.")
        aligned.append(
            {
                "row_id": row_id,
                **expected,
                "text": row["text"],
                "token_ids": row["token_ids"],
            }
        )
    return matrix, aligned


def _load_shadow_ids(path: Path) -> list[int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to read the frozen shadow artifact."
        ) from exc
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or not isinstance(
        value.get("in_original"), Mapping
    ):
        raise ValueError("Shadow artifact has no in_original mapping.")
    raw_ids = sorted(value["in_original"].keys())
    ids = [_required_int(item, "shadow target ID") for item in raw_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("Shadow target IDs are not unique after integer conversion.")
    if len(ids) <= EXPECTED_EVALUATION_COUNT:
        raise ValueError(
            "The frozen shadow artifact provides no reserve IDs after the first 600."
        )
    return ids


def _plain_dataset_row(value: Mapping[str, Any], sample_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if hasattr(item, "tolist"):
            item = item.tolist()
        result[str(key)] = item
    if "input_ids" not in result:
        raise ValueError(f"Target dataset sample {sample_id} has no input_ids.")
    result["input_ids"] = _token_ids(
        result["input_ids"], f"target dataset sample {sample_id} input_ids"
    )
    return result


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def _sample(sample_id: int, text: str, token_count: int) -> Sample:
    if not text.strip():
        raise ValueError(f"Target sample {sample_id} decodes to empty text.")
    return Sample(
        sample_id=sample_id,
        text=text,
        token_count=token_count,
        is_heading=_is_heading(text),
        text_sha256=_sha256_text(text),
    )


def _minimum_length_matching(
    sources: Sequence[Sample], candidates: Sequence[Sample], label: str
) -> list[Match]:
    """Globally minimize total absolute token-count difference.

    Absolute-distance matching has an optimal non-crossing solution after sorting
    both sides by length.  Dynamic programming also selects the best subset when
    there are more candidates than sources.  Equal-cost solutions use the
    lexicographically ascending candidate-ID sequence.
    """

    if len(candidates) < len(sources):
        raise ValueError(
            f"Only {len(candidates)} eligible {label} candidates exist for "
            f"{len(sources)} support examples; the protocol may not be relaxed."
        )
    source_order = sorted(sources, key=lambda item: (item.token_count, item.sample_id))
    candidate_order = sorted(
        candidates, key=lambda item: (item.token_count, item.sample_id)
    )
    # State: (total cost, matched candidate IDs, matched candidate positions).
    previous: list[tuple[int, tuple[int, ...], tuple[int, ...]] | None] = [
        (0, (), ()) for _ in range(len(candidate_order) + 1)
    ]
    for source_index, source in enumerate(source_order, start=1):
        current: list[tuple[int, tuple[int, ...], tuple[int, ...]] | None] = [
            None
        ] * (len(candidate_order) + 1)
        for candidate_count in range(1, len(candidate_order) + 1):
            skip = current[candidate_count - 1]
            prior = previous[candidate_count - 1]
            take = None
            if prior is not None:
                candidate = candidate_order[candidate_count - 1]
                take = (
                    prior[0] + abs(source.token_count - candidate.token_count),
                    (*prior[1], candidate.sample_id),
                    (*prior[2], candidate_count - 1),
                )
            if skip is None:
                current[candidate_count] = take
            elif take is None:
                current[candidate_count] = skip
            else:
                current[candidate_count] = min(
                    skip, take, key=lambda state: (state[0], state[1])
                )
        previous = current
        if previous[-1] is None:
            raise AssertionError(
                f"Internal {label} matching failure at source {source_index}."
            )
    result = previous[-1]
    if result is None or len(result[2]) != len(source_order):
        raise AssertionError(f"Internal {label} matching failed.")
    return [
        Match(source=source, candidate=candidate_order[position])
        for source, position in zip(source_order, result[2])
    ]


def _encode_reserve(samples: Sequence[Sample]) -> tuple[np.ndarray, str | None]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to screen reserve examples."
        ) from exc
    model = SentenceTransformer(
        EMBEDDING_MODEL,
        revision=EMBEDDING_REVISION,
        device="cpu",
    )
    matrix = np.asarray(
        model.encode(
            [sample.text for sample in samples],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    _validate_matrix(matrix, len(samples), "reserve candidate")
    try:
        resolved = getattr(model[0].auto_model.config, "_commit_hash", None)
    except (AttributeError, IndexError, KeyError, TypeError):
        resolved = None
    if resolved is not None and str(resolved) != EMBEDDING_REVISION:
        raise ValueError(
            f"Resolved embedding revision {resolved!r} differs from frozen "
            f"revision {EMBEDDING_REVISION!r}."
        )
    return matrix, str(resolved) if resolved is not None else None


def _sample_record(sample: Sample, **extra: Any) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "original_dataset_index": sample.sample_id,
        "gpt2_token_count": sample.token_count,
        "is_wikitext_heading": sample.is_heading,
        "text_sha256": sample.text_sha256,
        **extra,
    }


def _matching_records(
    matches: Sequence[Match], candidate_key: str
) -> list[dict[str, Any]]:
    return [
        {
            "support_sample_id": match.source.sample_id,
            candidate_key: match.candidate.sample_id,
            "support_gpt2_token_count": match.source.token_count,
            f"{candidate_key.removesuffix('_sample_id')}_gpt2_token_count": (
                match.candidate.token_count
            ),
            "absolute_token_count_difference": abs(
                match.source.token_count - match.candidate.token_count
            ),
        }
        for match in matches
    ]


def _condition_order(
    original_ids: Sequence[int], matches: Sequence[Match]
) -> list[int]:
    replacements = {
        match.source.sample_id: match.candidate.sample_id for match in matches
    }
    return [replacements.get(sample_id, sample_id) for sample_id in original_ids]


def _write_manifest(path: Path, manifest: dict[str, Any]) -> tuple[str, str]:
    content_hash = _canonical_sha256(manifest)
    manifest["manifest_hash"] = {
        "algorithm": "sha256",
        "scope": (
            "canonical UTF-8 JSON of all manifest fields except manifest_hash; "
            "sorted keys and compact separators"
        ),
        "sha256": content_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(
                manifest,
                output_file,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            output_file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_hash, _sha256_file(path)


def main() -> None:
    args = parse_args()
    paths = {
        "semantic_support_csv": args.semantic_support.resolve(),
        "retained_corpus": args.retained_corpus.resolve(),
        "unlearn_embeddings": args.unlearn_embeddings.resolve(),
        "retain_embeddings": args.retain_embeddings.resolve(),
        "shadow_artifact": args.shadow_path.resolve(),
    }
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".json":
        raise ValueError("--output must have a .json suffix.")
    if output_path in set(paths.values()):
        raise ValueError("Manifest output must differ from every input artifact.")

    input_artifacts = {
        name: _verify_frozen_file(path, name) for name, path in paths.items()
    }
    target_storage = _dataset_storage_manifest(args.target_data_path.resolve())
    input_artifacts["target_dataset"] = target_storage

    support_rows = _load_semantic_support(paths["semantic_support_csv"])
    retained_by_id, retained_order = _load_retained_corpus(paths["retained_corpus"])
    unlearn_arrays = _load_npz(
        paths["unlearn_embeddings"], {"embeddings", "sample_ids", "text_sha256"}
    )
    retain_arrays = _load_npz(
        paths["retain_embeddings"],
        {
            "embeddings",
            "row_ids",
            "retained_indices",
            "sources",
            "source_indices",
            "text_sha256",
        },
    )
    unlearn_matrix, unlearn_ids = _align_unlearn_embeddings(
        unlearn_arrays, support_rows
    )
    retain_matrix, aligned_retain = _align_retain_embeddings(
        retain_arrays, retained_by_id
    )
    if unlearn_matrix.shape[1] != retain_matrix.shape[1]:
        raise ValueError("Frozen UNLEARN and RETAIN embedding dimensions differ.")

    shadow_ids = _load_shadow_ids(paths["shadow_artifact"])
    in_ids = shadow_ids[:200]
    official_unlearn_ids = shadow_ids[200:400]
    out_ids = shadow_ids[400:600]
    reserve_ids = shadow_ids[600:]
    if set(unlearn_ids) != set(official_unlearn_ids):
        raise ValueError(
            "Frozen UNLEARN embedding IDs do not match shadow IDs 200:400."
        )

    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "datasets and transformers are required to validate the target dataset."
        ) from exc
    target_dataset = load_from_disk(str(args.target_data_path.resolve()))
    if (
        getattr(target_dataset, "_fingerprint", None)
        != FROZEN_TARGET_DATASET_FINGERPRINT
    ):
        raise ValueError(
            "Target dataset fingerprint differs from the frozen Experiment 1 value."
        )
    if min(shadow_ids) < 0 or max(shadow_ids) >= len(target_dataset):
        raise ValueError("Shadow target IDs fall outside the frozen target dataset.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # The frozen semantic CSV and target dataset must describe exactly the same
    # UNLEARN rows.  This also proves that current GPT-2 decoding reproduces the
    # text semantics used to build the frozen embeddings.
    for sample_id in official_unlearn_ids:
        target_row = _plain_dataset_row(target_dataset[sample_id], sample_id)
        recorded_token_ids = _token_ids(
            support_rows[sample_id]["token_ids"],
            f"semantic-support sample {sample_id} token_ids",
        )
        if target_row["input_ids"] != recorded_token_ids:
            raise ValueError(
                f"Target dataset tokens differ from UNLEARN sample {sample_id}."
            )
        if _decode_text(tokenizer, target_row["input_ids"]) != support_rows[
            sample_id
        ]["text"]:
            raise ValueError(
                f"GPT-2 decoding differs from UNLEARN sample {sample_id}."
            )

    target_in_embedding_rows = [
        index
        for index, row in enumerate(aligned_retain)
        if row["source"] == "target_in"
    ]
    if len(target_in_embedding_rows) != EXPECTED_TARGET_IN_COUNT:
        raise ValueError("Aligned RETAIN embeddings do not contain 200 target-IN rows.")
    target_in_samples: dict[int, Sample] = {}
    for embedding_index in target_in_embedding_rows:
        retained_row = aligned_retain[embedding_index]
        sample_id = int(retained_row["source_index"])
        if sample_id in target_in_samples:
            raise ValueError(f"Duplicate retained target-IN sample {sample_id}.")
        target_row = _plain_dataset_row(target_dataset[sample_id], sample_id)
        if target_row["input_ids"] != retained_row["token_ids"]:
            raise ValueError(
                f"Target dataset tokens differ from retained target-IN {sample_id}."
            )
        decoded = _decode_text(tokenizer, target_row["input_ids"])
        if decoded != retained_row["text"]:
            raise ValueError(
                f"GPT-2 decoding differs from retained target-IN {sample_id}."
            )
        target_in_samples[sample_id] = _sample(
            sample_id, decoded, len(target_row["input_ids"])
        )
    if set(target_in_samples) != set(in_ids):
        raise ValueError("Retained target-IN IDs do not match shadow IDs 0:200.")

    unlearn_index = {sample_id: index for index, sample_id in enumerate(unlearn_ids)}
    target_in_id_order = list(in_ids)
    if [
        int(aligned_retain[index]["source_index"])
        for index in target_in_embedding_rows
    ] != target_in_id_order:
        raise ValueError(
            "Target-IN embedding order differs from the ordered shadow IN IDs."
        )
    # Reproduce Experiment 1's complete matrix multiplication before taking the
    # target-IN columns.  This avoids making cohort membership depend on a
    # differently shaped BLAS operation near the frozen threshold.
    complete_similarity = unlearn_matrix @ retain_matrix.T
    if complete_similarity.shape != (
        EXPECTED_UNLEARN_COUNT,
        EXPECTED_RETAIN_COUNT,
    ):
        raise ValueError(
            f"Unexpected complete similarity shape {complete_similarity.shape}."
        )
    similarity = complete_similarity[:, target_in_embedding_rows]
    if not np.isfinite(complete_similarity).all():
        raise ValueError("UNLEARN-to-target-IN similarities contain non-finite values.")
    for sample_id, row in support_rows.items():
        recomputed = int(
            np.count_nonzero(
                similarity[unlearn_index[sample_id]] >= INTERVENTION_THRESHOLD
            )
        )
        recorded = int(row["target_in_neighbor_count_ge_0_75"])
        if recomputed != recorded:
            raise ValueError(
                f"Frozen semantic-support count mismatch for sample {sample_id}: "
                f"CSV={recorded}, embeddings={recomputed}."
            )

    s_ids = sorted(
        sample_id
        for sample_id, row in support_rows.items()
        if int(row["is_wikitext_heading"]) == 0
        and int(row["target_in_neighbor_count_ge_0_75"]) > 0
    )
    negative_control_ids = sorted(
        sample_id
        for sample_id, row in support_rows.items()
        if int(row["is_wikitext_heading"]) == 0
        and int(row["target_in_neighbor_count_ge_0_75"]) == 0
    )
    if len(s_ids) != EXPECTED_S_COUNT:
        raise ValueError(f"Expected |S|={EXPECTED_S_COUNT}; found {len(s_ids)}.")
    if len(negative_control_ids) != EXPECTED_NEGATIVE_CONTROL_COUNT:
        raise ValueError(
            "Expected 121 non-heading negative-control samples; found "
            f"{len(negative_control_ids)}."
        )
    s_samples = {
        sample_id: _sample(
            sample_id,
            support_rows[sample_id]["text"],
            int(support_rows[sample_id]["gpt2_token_count"]),
        )
        for sample_id in s_ids
    }
    if any(sample.is_heading for sample in s_samples.values()):
        raise ValueError("Primary target cohort S contains a WikiText heading.")
    primary_hashes = {sample.text_sha256 for sample in s_samples.values()}

    s_similarity_rows = np.asarray([unlearn_index[sample_id] for sample_id in s_ids])
    s_to_in = similarity[s_similarity_rows]
    support_mask = s_to_in >= INTERVENTION_THRESHOLD
    u_positions = np.flatnonzero(np.any(support_mask, axis=0)).tolist()
    u_ids = [target_in_id_order[position] for position in u_positions]
    if not u_ids:
        raise ValueError("U is empty at the frozen 0.75 threshold.")
    u_samples = [target_in_samples[sample_id] for sample_id in u_ids]
    support_pairs: list[dict[str, Any]] = []
    for s_position, sample_id in enumerate(s_ids):
        for in_position in np.flatnonzero(support_mask[s_position]).tolist():
            support_pairs.append(
                {
                    "s_sample_id": sample_id,
                    "u_sample_id": target_in_id_order[in_position],
                    "cosine_similarity": float(s_to_in[s_position, in_position]),
                }
            )
    supported_u_ids = {pair["u_sample_id"] for pair in support_pairs}
    if supported_u_ids != set(u_ids):
        raise AssertionError("U support-pair union is inconsistent.")

    max_s_similarity_by_in = np.max(s_to_in, axis=0)
    u_set = set(u_ids)
    placebo_candidates = [
        target_in_samples[sample_id]
        for position, sample_id in enumerate(target_in_id_order)
        if sample_id not in u_set
        and not target_in_samples[sample_id].is_heading
        and target_in_samples[sample_id].text_sha256 not in primary_hashes
        and float(max_s_similarity_by_in[position]) < UNRELATED_THRESHOLD
    ]
    placebo_matches = _minimum_length_matching(u_samples, placebo_candidates, "placebo")
    p_ids = sorted(match.candidate.sample_id for match in placebo_matches)
    if len(p_ids) != len(u_ids) or len(set(p_ids)) != len(u_ids):
        raise ValueError("Placebo matching did not produce |P|=|U| unique IDs.")

    reserve_samples: list[Sample] = []
    for sample_id in reserve_ids:
        row = _plain_dataset_row(target_dataset[sample_id], sample_id)
        text = _decode_text(tokenizer, row["input_ids"])
        candidate = _sample(sample_id, text, len(row["input_ids"]))
        if (
            not candidate.is_heading
            and candidate.text_sha256 not in primary_hashes
        ):
            reserve_samples.append(candidate)
    if len(reserve_samples) < len(u_ids):
        raise ValueError(
            f"Only {len(reserve_samples)} reserve rows pass heading/duplicate "
            f"filters for |U|={len(u_ids)}; the protocol may not be relaxed."
        )
    reserve_matrix, resolved_revision = _encode_reserve(reserve_samples)
    if reserve_matrix.shape[1] != unlearn_matrix.shape[1]:
        raise ValueError("Reserve and frozen UNLEARN embedding dimensions differ.")
    s_vectors = np.stack(
        [unlearn_matrix[unlearn_index[sample_id]] for sample_id in s_ids]
    )
    reserve_similarity = s_vectors @ reserve_matrix.T
    if not np.isfinite(reserve_similarity).all():
        raise ValueError("Primary-target-to-reserve similarities are non-finite.")
    reserve_maxima = np.max(reserve_similarity, axis=0)
    eligible_reserve = [
        sample
        for index, sample in enumerate(reserve_samples)
        if float(reserve_maxima[index]) < UNRELATED_THRESHOLD
    ]
    if len(eligible_reserve) < len(u_ids):
        raise ValueError(
            f"Only {len(eligible_reserve)} reserve rows have maximum S similarity "
            f"< {UNRELATED_THRESHOLD:.2f} for |U|={len(u_ids)}; the protocol "
            "may not be relaxed."
        )
    reserve_matches = _minimum_length_matching(u_samples, eligible_reserve, "reserve")
    r_ids = sorted(match.candidate.sample_id for match in reserve_matches)
    if len(r_ids) != len(u_ids) or len(set(r_ids)) != len(u_ids):
        raise ValueError("Reserve matching did not produce |R|=|U| unique IDs.")

    in_position = {
        sample_id: index for index, sample_id in enumerate(target_in_id_order)
    }
    reserve_position = {
        sample.sample_id: index for index, sample in enumerate(reserve_samples)
    }
    p_records = [
        _sample_record(
            target_in_samples[sample_id],
            maximum_s_cosine_similarity=float(
                max_s_similarity_by_in[in_position[sample_id]]
            ),
        )
        for sample_id in p_ids
    ]
    r_sample_by_id = {sample.sample_id: sample for sample in eligible_reserve}
    r_records = [
        _sample_record(
            r_sample_by_id[sample_id],
            maximum_s_cosine_similarity=float(
                reserve_maxima[reserve_position[sample_id]]
            ),
        )
        for sample_id in r_ids
    ]
    if any(
        record["maximum_s_cosine_similarity"] >= UNRELATED_THRESHOLD
        for record in p_records
    ):
        raise ValueError(
            "P contains a sample at or above the 0.70 exclusion threshold."
        )
    if any(
        record["maximum_s_cosine_similarity"] >= UNRELATED_THRESHOLD
        for record in r_records
    ):
        raise ValueError(
            "R contains a sample at or above the 0.70 exclusion threshold."
        )

    placebo_by_u = {
        match.source.sample_id: match.candidate for match in placebo_matches
    }
    reserve_by_u = {
        match.source.sample_id: match.candidate for match in reserve_matches
    }
    low_matches = [
        Match(target_in_samples[u_id], reserve_by_u[u_id]) for u_id in u_ids
    ]
    placebo_condition_matches = [
        Match(placebo_by_u[u_id], reserve_by_u[u_id]) for u_id in u_ids
    ]
    high_order = list(target_in_id_order)
    low_order = _condition_order(high_order, low_matches)
    placebo_order = _condition_order(high_order, placebo_condition_matches)
    for name, membership in (
        ("HIGH", high_order),
        ("LOW", low_order),
        ("PLACEBO", placebo_order),
    ):
        if (
            len(membership) != EXPECTED_TARGET_IN_COUNT
            or len(set(membership)) != EXPECTED_TARGET_IN_COUNT
        ):
            raise ValueError(f"{name} does not contain 200 unique target examples.")
    if set(low_order) != (set(in_ids) - set(u_ids)) | set(r_ids):
        raise ValueError("LOW membership is inconsistent with (IN - U) + R.")
    if set(placebo_order) != (set(in_ids) - set(p_ids)) | set(r_ids):
        raise ValueError("PLACEBO membership is inconsistent with (IN - P) + R.")
    if set(r_ids) & set(shadow_ids[:600]):
        raise ValueError("R intersects an official IN, UNLEARN, or OUT evaluation ID.")

    wikitext_rows = [
        row for row in retained_order if row["source"] == "wikitext_attack"
    ]
    wikitext_identity = [
        {
            "row_id": str(row["row_id"]),
            "retained_index": int(row["retained_index"]),
            "source_index": int(row["source_index"]),
            "text_sha256": _sha256_text(str(row["text"])),
        }
        for row in wikitext_rows
    ]
    if len(wikitext_identity) != EXPECTED_WIKITEXT_COUNT:
        raise ValueError("The frozen WikiText background does not contain 15,000 rows.")
    wikitext_membership_hash = _canonical_sha256({"rows": wikitext_identity})
    shared_wikitext = {
        "artifact": "retained_corpus",
        "source": "wikitext_attack",
        "count": EXPECTED_WIKITEXT_COUNT,
        "membership_sha256": wikitext_membership_hash,
    }

    s_records = [
        _sample_record(
            s_samples[sample_id],
            target_in_neighbor_count_ge_0_75=int(
                support_rows[sample_id]["target_in_neighbor_count_ge_0_75"]
            ),
        )
        for sample_id in s_ids
    ]
    u_records = [
        _sample_record(
            target_in_samples[sample_id],
            retained_row_id=f"target_in:{sample_id}",
            maximum_s_cosine_similarity=float(
                max_s_similarity_by_in[in_position[sample_id]]
            ),
        )
        for sample_id in u_ids
    ]
    conditions = {
        "HIGH": {
            "target_count": EXPECTED_TARGET_IN_COUNT,
            "removed_original_target_ids": [],
            "replacement_target_ids": [],
            "replacement_pairs": [],
            "ordered_target_dataset_ids": high_order,
            "wikitext_membership": shared_wikitext,
        },
        "LOW": {
            "target_count": EXPECTED_TARGET_IN_COUNT,
            "removed_original_target_ids": sorted(u_ids),
            "replacement_target_ids": r_ids,
            "replacement_pairs": [
                {
                    "removed_target_id": u_id,
                    "replacement_target_id": reserve_by_u[u_id].sample_id,
                }
                for u_id in sorted(u_ids)
            ],
            "ordered_target_dataset_ids": low_order,
            "wikitext_membership": shared_wikitext,
        },
        "PLACEBO": {
            "target_count": EXPECTED_TARGET_IN_COUNT,
            "removed_original_target_ids": p_ids,
            "replacement_target_ids": r_ids,
            "replacement_pairs": [
                {
                    "matched_support_target_id": u_id,
                    "removed_target_id": placebo_by_u[u_id].sample_id,
                    "replacement_target_id": reserve_by_u[u_id].sample_id,
                }
                for u_id in sorted(u_ids)
            ],
            "ordered_target_dataset_ids": placebo_order,
            "wikitext_membership": shared_wikitext,
        },
    }
    if (
        conditions["LOW"]["replacement_target_ids"]
        != conditions["PLACEBO"]["replacement_target_ids"]
    ):
        raise AssertionError("LOW and PLACEBO do not use exactly the same R list.")
    wikitext_hashes = {
        condition["wikitext_membership"]["membership_sha256"]
        for condition in conditions.values()
    }
    if len(wikitext_hashes) != 1:
        raise AssertionError("WikiText membership differs across conditions.")

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "experiment_1_reference_commit": EXPERIMENT_1_REFERENCE_COMMIT,
            "intervention_cosine_threshold_inclusive": INTERVENTION_THRESHOLD,
            "unrelated_cosine_threshold_exclusive": UNRELATED_THRESHOLD,
            "evaluation_partition": (
                "sorted(shadow_results['in_original'].keys()): IN=0:200, "
                "UNLEARN=200:400, OUT=400:600, reserve=600:end"
            ),
            "automatic_threshold_relaxation": False,
        },
        "embedding": {
            "model_name": EMBEDDING_MODEL,
            "revision_requested": EMBEDDING_REVISION,
            "revision_resolved_for_reserve": resolved_revision,
            "normalized": True,
            "device_for_reserve_screening": "cpu",
            "library_versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "datasets",
                    "numpy",
                    "sentence-transformers",
                    "torch",
                    "transformers",
                )
            },
        },
        "input_artifacts": input_artifacts,
        "evaluation_ids": {
            "ordered_shadow_target_ids": shadow_ids,
            "in_ids": in_ids,
            "unlearn_ids": official_unlearn_ids,
            "out_ids": out_ids,
            "reserve_candidate_ids": reserve_ids,
        },
        "sets": {
            "S_sample_ids": s_ids,
            "U_sample_ids": sorted(u_ids),
            "P_sample_ids": p_ids,
            "R_sample_ids": r_ids,
            "negative_control_sample_ids": negative_control_ids,
        },
        "samples": {
            "S": s_records,
            "U": sorted(u_records, key=lambda record: record["sample_id"]),
            "P": p_records,
            "R": r_records,
        },
        "u_support_pairs": sorted(
            support_pairs, key=lambda pair: (pair["s_sample_id"], pair["u_sample_id"])
        ),
        "matching": {
            "algorithm": (
                "globally minimize total absolute GPT-2 token-count difference "
                "with order-preserving dynamic programming after sorting each side "
                "by (token count, sample ID); equal-cost solutions use the "
                "lexicographically ascending candidate-ID sequence"
            ),
            "placebo_matches": _matching_records(
                placebo_matches, "placebo_sample_id"
            ),
            "reserve_matches": _matching_records(
                reserve_matches, "reserve_sample_id"
            ),
        },
        "shared_wikitext_background": {
            **shared_wikitext,
            "rows": wikitext_identity,
        },
        "conditions": conditions,
        "validation": {
            "passed": True,
            "S_count": len(s_ids),
            "U_count": len(u_ids),
            "P_count": len(p_ids),
            "R_count": len(r_ids),
            "negative_control_count": len(negative_control_ids),
            "condition_target_counts": {
                name: len(condition["ordered_target_dataset_ids"])
                for name, condition in conditions.items()
            },
            "wikitext_count": len(wikitext_identity),
            "all_frozen_hashes_match": True,
            "all_protocol_invariants_pass": True,
            "checks": {
                "S_has_exactly_28_samples": len(s_ids) == EXPECTED_S_COUNT,
                "S_contains_no_headings": not any(
                    sample.is_heading for sample in s_samples.values()
                ),
                "every_U_member_has_a_pair_at_or_above_0_75": (
                    supported_u_ids == set(u_ids)
                ),
                "every_P_maximum_S_similarity_is_below_0_70": all(
                    record["maximum_s_cosine_similarity"] < UNRELATED_THRESHOLD
                    for record in p_records
                ),
                "every_R_maximum_S_similarity_is_below_0_70": all(
                    record["maximum_s_cosine_similarity"] < UNRELATED_THRESHOLD
                    for record in r_records
                ),
                "P_and_R_counts_equal_U_count": (
                    len(p_ids) == len(u_ids) == len(r_ids)
                ),
                "condition_target_counts_are_equal": (
                    len(high_order) == len(low_order) == len(placebo_order)
                ),
                "LOW_and_PLACEBO_use_identical_R": (
                    conditions["LOW"]["replacement_target_ids"]
                    == conditions["PLACEBO"]["replacement_target_ids"]
                ),
                "wikitext_background_is_shared_and_has_15000_rows": (
                    len(wikitext_hashes) == 1
                    and len(wikitext_identity) == EXPECTED_WIKITEXT_COUNT
                ),
            },
        },
    }
    validation_checks = manifest["validation"]["checks"]
    if not isinstance(validation_checks, Mapping) or not all(
        validation_checks.values()
    ):
        failed = [
            name
            for name, passed in validation_checks.items()
            if not bool(passed)
        ]
        raise AssertionError(
            "Manifest validation checks failed: " + ", ".join(sorted(failed))
        )
    content_hash, file_hash = _write_manifest(output_path, manifest)
    print(
        f"[VERIFY] |S|={len(s_ids)} |U|={len(u_ids)} "
        f"|P|={len(p_ids)} |R|={len(r_ids)}"
    )
    print(f"[SELECT] S={s_ids}")
    print(f"[SELECT] U={sorted(u_ids)}")
    print(f"[SELECT] P={p_ids}")
    print(f"[SELECT] R={r_ids}")
    print(f"[INFO] Wrote {output_path}")
    print(f"[VERIFY] Manifest content SHA-256: {content_hash}")
    print(f"[VERIFY] Manifest file SHA-256: {file_hash}")


if __name__ == "__main__":
    main()
