"""Export the exact retained corpus constructed by official RULI inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
RESULTS_DIR = SCRIPT_DIR / "results"
REFERENCE_SEED = 42
REFERENCE_ATTACK_SIZE = 15_000
REFERENCE_IN_SIZE = 200
OFFICIAL_MINIMUM_SHADOW_IDS = 600
SOURCE_INDEX_COLUMN = "__ruli_filtered_train_index"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export IN + WikiText attack samples using the exact dataset "
            "construction performed by RULI text/mia_inference.py. This command "
            "does not load a model, run inference, train, or require a GPU."
        )
    )
    parser.add_argument(
        "--ruli-root",
        type=Path,
        default=DEFAULT_RULI_ROOT,
        help="Path to the sibling upstream RULI checkout.",
    )
    parser.add_argument(
        "--target-data-path",
        type=Path,
        default=None,
        help=(
            "Target dataset saved by RULI. Defaults to "
            "<ruli-root>/text/data/WikiText-103-local/gpt2/"
            "selective_dataset_prefixed."
        ),
    )
    parser.add_argument(
        "--shadow-path",
        type=Path,
        default=None,
        help=(
            "Existing official shadow result file. Defaults to <ruli-root>/core/"
            "attack/attack_inferences/WikiText103/"
            "shadow_9_attack_random_npo_gpt2.pth."
        ),
    )
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--attack-size", type=int, default=REFERENCE_ATTACK_SIZE)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "retained_corpus.jsonl",
        help="Destination JSON Lines corpus.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to <output stem>.metadata.json.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_storage_manifest(path: Path) -> dict[str, Any]:
    """Hash stable saved-dataset files while ignoring derived HF cache files."""
    entries: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    patterns = ("data-*.arrow", "dataset_info.json", "state.json")
    files = sorted(
        {file for pattern in patterns for file in path.glob(pattern) if file.is_file()},
        key=lambda file: file.name,
    )
    if not files:
        raise ValueError(f"No saved Hugging Face dataset files found in {path}")
    for file in files:
        file_hash = _sha256_file(file)
        relative_name = file.relative_to(path).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append(
            {"path": relative_name, "sha256": file_hash, "bytes": file.stat().st_size}
        )
    return {"sha256": digest.hexdigest(), "files": entries}


def _git_provenance(repository: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def _plain_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _load_upstream_data(ruli_text_dir: Path, model_name: str) -> Any:
    """Call RULI's own load_data() with its working-directory semantics intact."""
    utils_path = ruli_text_dir / "utils.py"
    if not utils_path.is_file():
        raise FileNotFoundError(f"RULI text/utils.py does not exist: {utils_path}")

    module_name = "_ruli_text_utils_for_retain_export"
    spec = importlib.util.spec_from_file_location(module_name, utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream RULI module: {utils_path}")
    module = importlib.util.module_from_spec(spec)

    previous_cwd = Path.cwd()
    sys.path.insert(0, str(ruli_text_dir))
    sys.modules[module_name] = module
    try:
        os.chdir(ruli_text_dir)
        spec.loader.exec_module(module)
        train_dataset, _valid_dataset, _normal_texts = module.load_data(
            "WikiText103", SimpleNamespace(model_name=model_name)
        )
    finally:
        os.chdir(previous_cwd)
        sys.path.remove(str(ruli_text_dir))
        sys.modules.pop(module_name, None)
    return train_dataset


def _load_shadow_results(path: Path) -> Mapping[str, Any]:
    import torch

    if not path.is_file():
        raise FileNotFoundError(
            f"Shadow result file does not exist: {path}. Pass the exact file used "
            "by the official mia_inference.py run with --shadow-path."
        )
    shadow_results = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(shadow_results, Mapping):
        raise ValueError("Shadow result artifact is not a mapping.")
    if "in_original" not in shadow_results:
        raise ValueError("Shadow result artifact has no 'in_original' mapping.")
    if not isinstance(shadow_results["in_original"], Mapping):
        raise ValueError("shadow_results['in_original'] is not a mapping.")
    return shadow_results


def _attack_source_indices(
    train_dataset: Any, attack_dataset: Any, seed: int, attack_size: int
) -> list[int]:
    """Recover logical filtered-train indices using the same public shuffle API."""
    if SOURCE_INDEX_COLUMN in train_dataset.column_names:
        raise ValueError(f"Unexpected source dataset column: {SOURCE_INDEX_COLUMN}")
    indexed_train = train_dataset.add_column(
        SOURCE_INDEX_COLUMN, list(range(len(train_dataset)))
    )
    indexed_attack = indexed_train.shuffle(seed=seed).select(range(attack_size))

    source_indices: list[int] = []
    for attack_index in range(attack_size):
        attack_sample = _plain_value(attack_dataset[attack_index])
        indexed_sample = _plain_value(indexed_attack[attack_index])
        source_index = int(indexed_sample.pop(SOURCE_INDEX_COLUMN))
        if attack_sample != indexed_sample:
            raise ValueError(
                "Indexed WikiText reconstruction diverged from RULI attack_dataset "
                f"at attack index {attack_index}."
            )
        source_indices.append(source_index)
    if len(set(source_indices)) != attack_size:
        raise ValueError("WikiText shuffle unexpectedly selected duplicate source rows.")
    return source_indices


def _sample_row(
    *,
    sample: Mapping[str, Any],
    retained_index: int,
    source: str,
    source_index: int,
    source_selection_index: int,
    tokenizer: Any,
) -> dict[str, Any]:
    if "input_ids" not in sample:
        raise ValueError(f"Retained {source} sample {source_index} has no input_ids.")
    token_ids = [int(token_id) for token_id in _plain_value(sample["input_ids"])]
    if not token_ids:
        raise ValueError(f"Retained {source} sample {source_index} is empty.")
    text = sample.get("text")
    text_origin = "dataset_text"
    if text is None:
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text_origin = "decoded_input_ids"

    row = {
        "row_id": f"{source}:{source_index}",
        "retained_index": retained_index,
        "source": source,
        "source_index": source_index,
        "source_selection_index": source_selection_index,
        "text": str(text),
        "text_origin": text_origin,
        "token_ids": token_ids,
    }
    if "attention_mask" in sample:
        row["attention_mask"] = [
            int(value) for value in _plain_value(sample["attention_mask"])
        ]
    return row


def _iter_rows(
    *,
    in_data: Any,
    in_ids: Sequence[int],
    attack_dataset: Any,
    attack_source_indices: Sequence[int],
    tokenizer: Any,
) -> Iterator[dict[str, Any]]:
    retained_index = 0
    for selection_index, target_index in enumerate(in_ids):
        yield _sample_row(
            sample=in_data[selection_index],
            retained_index=retained_index,
            source="target_in",
            source_index=target_index,
            source_selection_index=selection_index,
            tokenizer=tokenizer,
        )
        retained_index += 1
    for attack_index, source_index in enumerate(attack_source_indices):
        yield _sample_row(
            sample=attack_dataset[attack_index],
            retained_index=retained_index,
            source="wikitext_attack",
            source_index=source_index,
            source_selection_index=attack_index,
            tokenizer=tokenizer,
        )
        retained_index += 1


def _write_jsonl(rows: Iterator[dict[str, Any]], output: Path) -> tuple[int, Counter[str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    count = 0
    sources: Counter[str] = Counter()
    seen_row_ids: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in rows:
                row_id = str(row["row_id"])
                if row_id in seen_row_ids:
                    raise ValueError(f"Duplicate retained row ID: {row_id}")
                if int(row["retained_index"]) != count:
                    raise ValueError("Retained rows are not contiguously indexed.")
                seen_row_ids.add(row_id)
                sources[str(row["source"])] += 1
                json.dump(row, output_file, ensure_ascii=False, allow_nan=False)
                output_file.write("\n")
                count += 1
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count, sources


def _write_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if args.attack_size <= 0:
        raise ValueError("--attack-size must be positive.")

    ruli_root = args.ruli_root.resolve()
    ruli_text_dir = ruli_root / "text"
    target_data_path = (
        args.target_data_path.resolve()
        if args.target_data_path is not None
        else ruli_text_dir
        / "data"
        / "WikiText-103-local"
        / "gpt2"
        / "selective_dataset_prefixed"
    )
    shadow_path = (
        args.shadow_path.resolve()
        if args.shadow_path is not None
        else ruli_root
        / "core"
        / "attack"
        / "attack_inferences"
        / "WikiText103"
        / "shadow_9_attack_random_npo_gpt2.pth"
    )
    output = args.output.resolve()
    metadata_output = (
        args.metadata_output.resolve()
        if args.metadata_output is not None
        else output.with_suffix(".metadata.json")
    )
    if output == metadata_output:
        raise ValueError("Corpus and metadata output paths must be different.")
    if not target_data_path.is_dir():
        raise FileNotFoundError(f"Target dataset does not exist: {target_data_path}")

    from datasets import load_from_disk
    from torch.utils.data import Subset
    from transformers import AutoTokenizer

    print(f"[INFO] Loading target dataset from {target_data_path}")
    target_dataset = load_from_disk(str(target_data_path))

    print("[INFO] Calling upstream RULI load_data('WikiText103', args)")
    train_dataset = _load_upstream_data(ruli_text_dir, args.model_name)
    if len(train_dataset) < args.attack_size:
        raise ValueError(
            f"RULI filtered train dataset has {len(train_dataset)} rows, fewer than "
            f"the requested attack size {args.attack_size}."
        )

    # Keep these statements equivalent to official text/mia_inference.py.
    attack_dataset = train_dataset.shuffle(seed=args.seed).select(
        range(args.attack_size)
    )
    print(f"[INFO] Loading shadow results on CPU from {shadow_path}")
    shadow_results = _load_shadow_results(shadow_path)
    total_indices = sorted(shadow_results["in_original"].keys())
    if len(total_indices) < OFFICIAL_MINIMUM_SHADOW_IDS:
        raise ValueError(
            "Official mia_inference.py requires at least 600 sorted in_original "
            f"IDs; found {len(total_indices)}."
        )
    in_ids = total_indices[:REFERENCE_IN_SIZE]
    if not all(isinstance(index, int) for index in in_ids):
        try:
            in_ids = [int(index) for index in in_ids]
        except (TypeError, ValueError) as error:
            raise ValueError("Selected IN target IDs are not integer indices.") from error
    if min(in_ids) < 0 or max(in_ids) >= len(target_dataset):
        raise ValueError("Selected IN target IDs fall outside the target dataset.")
    in_data = Subset(target_dataset, in_ids)

    # Equivalent to Subset(target_dataset, in_ids), expressed through indexed access
    # below so source IDs remain available in every exported row.
    print("[INFO] Recovering original filtered WikiText indices")
    attack_source_indices = _attack_source_indices(
        train_dataset, attack_dataset, args.seed, args.attack_size
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"[INFO] Writing retained corpus to {output}")
    row_count, source_counts = _write_jsonl(
        _iter_rows(
            in_data=in_data,
            in_ids=in_ids,
            attack_dataset=attack_dataset,
            attack_source_indices=attack_source_indices,
            tokenizer=tokenizer,
        ),
        output,
    )
    expected_count = REFERENCE_IN_SIZE + args.attack_size
    if row_count != expected_count:
        raise ValueError(
            f"Expected {expected_count} retained rows, exported {row_count}."
        )
    if source_counts != {
        "target_in": REFERENCE_IN_SIZE,
        "wikitext_attack": args.attack_size,
    }:
        raise ValueError(f"Unexpected retained source counts: {dict(source_counts)}")

    reference_configuration = (
        args.seed == REFERENCE_SEED and args.attack_size == REFERENCE_ATTACK_SIZE
    )
    if reference_configuration and row_count != 15_200:
        raise ValueError(
            f"Reference configuration must produce exactly 15,200 rows, got {row_count}."
        )

    wikitext_cache_path = (
        ruli_text_dir
        / "data"
        / "WikiText-103-local"
        / "gpt2"
        / "tokenized_train_subset"
    )
    corpus_hash = _sha256_file(output)
    metadata = {
        "schema_version": 1,
        "construction": {
            "definition": "Subset(target_dataset, in_ids) + attack_dataset",
            "in_ids_expression": (
                "sorted(shadow_results['in_original'].keys())[:200]"
            ),
            "attack_expression": (
                "train_dataset.shuffle(seed=42).select(range(15000))"
                if reference_configuration
                else (
                    f"train_dataset.shuffle(seed={args.seed}).select("
                    f"range({args.attack_size}))"
                )
            ),
            "upstream_loader": "Ruli/text/utils.py:load_data('WikiText103', args)",
        },
        "parameters": {
            "seed": args.seed,
            "attack_size": args.attack_size,
            "in_size": REFERENCE_IN_SIZE,
            "model_name": args.model_name,
            "selected_in_target_ids": list(in_ids),
        },
        "sources": {
            "ruli_root": str(ruli_root),
            "ruli_git": _git_provenance(ruli_root),
            "ruli_utils_path": str(ruli_text_dir / "utils.py"),
            "ruli_utils_sha256": _sha256_file(ruli_text_dir / "utils.py"),
            "ruli_mia_inference_path": str(ruli_text_dir / "mia_inference.py"),
            "ruli_mia_inference_sha256": _sha256_file(
                ruli_text_dir / "mia_inference.py"
            ),
            "target_dataset": {
                "path": str(target_data_path),
                "fingerprint": getattr(target_dataset, "_fingerprint", None),
                "storage": _dataset_storage_manifest(target_data_path),
            },
            "wikitext_train": {
                "dataset": "wikitext",
                "configuration": "wikitext-103-v1",
                "raw_split": "train",
                "raw_selection": "range(50000)",
                "tokenization": {
                    "tokenizer": args.model_name,
                    "truncation": True,
                    "max_length": 128,
                    "removed_columns": ["text"],
                    "empty_input_ids_filtered": True,
                },
                "cache_path": str(wikitext_cache_path),
                "cache_storage": _dataset_storage_manifest(wikitext_cache_path),
                "filtered_train_fingerprint": getattr(
                    train_dataset, "_fingerprint", None
                ),
                "filtered_train_row_count": len(train_dataset),
                "attack_fingerprint": getattr(attack_dataset, "_fingerprint", None),
            },
            "shadow_results": {
                "path": str(shadow_path),
                "sha256": _sha256_file(shadow_path),
                "in_original_id_count": len(total_indices),
            },
        },
        "output": {
            "path": str(output),
            "format": "jsonl",
            "row_count": row_count,
            "source_counts": dict(sorted(source_counts.items())),
            "sha256": corpus_hash,
            "bytes": output.stat().st_size,
        },
        "reference_validation": {
            "is_default_reference_configuration": reference_configuration,
            "expected_row_count": 15_200,
            "expected_source_counts": {
                "target_in": 200,
                "wikitext_attack": 15_000,
            },
            "passed": row_count == 15_200 if reference_configuration else None,
        },
    }
    _write_json(metadata, metadata_output)

    print(
        f"[VERIFY] Exported {row_count} rows: "
        f"{source_counts['target_in']} target IN + "
        f"{source_counts['wikitext_attack']} WikiText attack"
    )
    print(f"[VERIFY] Corpus SHA-256: {corpus_hash}")
    print(f"[INFO] Wrote provenance metadata to {metadata_output}")


if __name__ == "__main__":
    main()
