#!/usr/bin/env python3
"""Run the frozen, single-seed Experiment 2A training intervention.

This runner deliberately imports the current upstream RULI training helpers instead
of copying their SFT, prefix-training, or NPO implementations.  The intervention
manifest is immutable input: this file validates it but never derives or repairs it.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
DEFAULT_MANIFEST = SCRIPT_DIR / "results" / "intervention_manifest.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "results" / "experiment_2a" / "seed_42"

MODEL_NAME = "gpt2"
SEED = 42
SFT_EPOCHS = 5
PREFIX_EPOCHS = 1
NPO_EPOCHS = 15
FINAL_FT_EPOCHS = 2
ATTACK_SIZE = 15_000
TARGET_COUNT = 200
EVALUATION_COUNT = 600
CONDITION_NAMES = ("HIGH", "LOW", "PLACEBO")
SOURCE_INDEX_COLUMN = "__experiment_2a_source_index__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen seed-42 Experiment 2A HIGH, LOW, and PLACEBO "
            "branches from one shared post-NPO checkpoint."
        )
    )
    parser.add_argument("--ruli-root", type=Path, default=DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=SEED)
    validation_mode = parser.add_mutually_exclusive_group()
    validation_mode.add_argument(
        "--validate-manifest-only",
        action="store_true",
        help="Validate only the immutable manifest; load no RULI artifacts.",
    )
    validation_mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and exact input datasets, but do not train.",
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


def _integer_list(value: Any, description: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array.")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{description} must contain only integer IDs.")
        result.append(item)
    return result


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object.")
    return value


def _load_and_validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen intervention manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid intervention manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("The intervention manifest root must be a JSON object.")

    declared_hash = _mapping(
        manifest.get("manifest_hash"), "manifest.manifest_hash"
    ).get("sha256")
    content_without_hash = dict(manifest)
    content_without_hash.pop("manifest_hash", None)
    actual_content_hash = _canonical_sha256(content_without_hash)
    if declared_hash != actual_content_hash:
        raise ValueError(
            "Frozen manifest canonical SHA-256 mismatch: "
            f"declared {declared_hash}, found {actual_content_hash}."
        )

    validation = _mapping(manifest.get("validation"), "manifest.validation")
    for field in (
        "passed",
        "all_protocol_invariants_pass",
        "all_frozen_hashes_match",
    ):
        if validation.get(field) is not True:
            raise ValueError(f"Manifest validation field {field!r} is not true.")
    checks = _mapping(validation.get("checks"), "manifest.validation.checks")
    failed_checks = [name for name, passed in checks.items() if passed is not True]
    if not checks or failed_checks:
        raise ValueError(
            "Manifest protocol checks did not all pass: "
            + (", ".join(sorted(failed_checks)) or "no checks recorded")
        )

    evaluation = _mapping(manifest.get("evaluation_ids"), "evaluation_ids")
    shadow_ids = _integer_list(
        evaluation.get("ordered_shadow_target_ids"),
        "evaluation_ids.ordered_shadow_target_ids",
    )
    in_ids = _integer_list(evaluation.get("in_ids"), "evaluation_ids.in_ids")
    unlearn_ids = _integer_list(
        evaluation.get("unlearn_ids"), "evaluation_ids.unlearn_ids"
    )
    out_ids = _integer_list(evaluation.get("out_ids"), "evaluation_ids.out_ids")
    if len(shadow_ids) < EVALUATION_COUNT or len(set(shadow_ids)) != len(shadow_ids):
        raise ValueError("Manifest shadow IDs must contain at least 600 unique IDs.")
    if not (
        in_ids == shadow_ids[:200]
        and unlearn_ids == shadow_ids[200:400]
        and out_ids == shadow_ids[400:600]
    ):
        raise ValueError(
            "Manifest evaluation partitions do not match its shadow order."
        )
    if any(len(ids) != TARGET_COUNT for ids in (in_ids, unlearn_ids, out_ids)):
        raise ValueError("Manifest IN, UNLEARN, and OUT must each contain 200 IDs.")

    frozen_sets = _mapping(manifest.get("sets"), "manifest.sets")
    u_ids = _integer_list(frozen_sets.get("U_sample_ids"), "sets.U_sample_ids")
    p_ids = _integer_list(frozen_sets.get("P_sample_ids"), "sets.P_sample_ids")
    r_ids = _integer_list(frozen_sets.get("R_sample_ids"), "sets.R_sample_ids")
    if not (len(u_ids) == len(p_ids) == len(r_ids)):
        raise ValueError("Frozen U, P, and R sets do not have equal sizes.")
    for name, ids in (("U", u_ids), ("P", p_ids), ("R", r_ids)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"Frozen {name} contains duplicate IDs.")

    conditions = _mapping(manifest.get("conditions"), "manifest.conditions")
    memberships: dict[str, list[int]] = {}
    for name in CONDITION_NAMES:
        condition = _mapping(conditions.get(name), f"conditions.{name}")
        ids = _integer_list(
            condition.get("ordered_target_dataset_ids"),
            f"conditions.{name}.ordered_target_dataset_ids",
        )
        if len(ids) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT:
            raise ValueError(f"{name} must contain exactly 200 unique target IDs.")
        if condition.get("target_count") != TARGET_COUNT:
            raise ValueError(f"{name}.target_count is not 200.")
        memberships[name] = ids
    if memberships["HIGH"] != in_ids:
        raise ValueError("HIGH is not the exact ordered official IN partition.")

    high_set = set(memberships["HIGH"])
    u_set, p_set, r_set = set(u_ids), set(p_ids), set(r_ids)
    if not u_set <= high_set:
        raise ValueError("U is not a subset of the original HIGH target IDs.")
    if not p_set <= high_set or u_set & p_set:
        raise ValueError("P must be disjoint from U and contained in HIGH.")
    if r_set & set(shadow_ids[:EVALUATION_COUNT]):
        raise ValueError("R overlaps the first 600 official evaluation IDs.")
    if set(memberships["LOW"]) != (high_set - u_set) | r_set:
        raise ValueError("LOW is not exactly (HIGH - U) + R.")
    if set(memberships["PLACEBO"]) != (high_set - p_set) | r_set:
        raise ValueError("PLACEBO is not exactly (HIGH - P) + R.")

    low = _mapping(conditions["LOW"], "conditions.LOW")
    placebo = _mapping(conditions["PLACEBO"], "conditions.PLACEBO")
    if set(
        _integer_list(low.get("removed_original_target_ids"), "LOW removed")
    ) != u_set:
        raise ValueError("LOW removed IDs do not equal frozen U.")
    if set(
        _integer_list(placebo.get("removed_original_target_ids"), "PLACEBO removed")
    ) != p_set:
        raise ValueError("PLACEBO removed IDs do not equal frozen P.")
    low_replacements = _integer_list(
        low.get("replacement_target_ids"), "LOW replacement_target_ids"
    )
    placebo_replacements = _integer_list(
        placebo.get("replacement_target_ids"), "PLACEBO replacement_target_ids"
    )
    if low_replacements != placebo_replacements or set(low_replacements) != r_set:
        raise ValueError("LOW and PLACEBO do not use the identical frozen R list.")

    shared_background = _mapping(
        manifest.get("shared_wikitext_background"), "shared_wikitext_background"
    )
    rows = shared_background.get("rows")
    if not isinstance(rows, list) or len(rows) != ATTACK_SIZE:
        raise ValueError("Frozen WikiText background must contain 15,000 rows.")
    if shared_background.get("count") != ATTACK_SIZE:
        raise ValueError("Frozen WikiText background count is not 15,000.")
    seen_background_ids: set[str] = set()
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"shared_wikitext_background.rows[{index}]")
        expected_retained_index = TARGET_COUNT + index
        if row.get("retained_index") != expected_retained_index:
            raise ValueError("Frozen WikiText retained indices are not contiguous.")
        source_index = row.get("source_index")
        row_id = row.get("row_id")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError("Frozen WikiText source_index is not an integer.")
        if row_id != f"wikitext_attack:{source_index}":
            raise ValueError("Frozen WikiText row_id/source_index mismatch.")
        if row_id in seen_background_ids:
            raise ValueError("Frozen WikiText background contains duplicate row IDs.")
        seen_background_ids.add(row_id)
    actual_background_hash = _canonical_sha256({"rows": rows})
    declared_background_hash = shared_background.get("membership_sha256")
    if actual_background_hash != declared_background_hash:
        raise ValueError("Frozen WikiText membership SHA-256 is internally invalid.")
    for name in CONDITION_NAMES:
        condition_background = _mapping(
            _mapping(conditions[name], f"conditions.{name}").get(
                "wikitext_membership"
            ),
            f"conditions.{name}.wikitext_membership",
        )
        for field in ("artifact", "source", "count", "membership_sha256"):
            if condition_background.get(field) != shared_background.get(field):
                raise ValueError(
                    f"{name} does not reference the shared WikiText corpus."
                )

    summary = {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "canonical_content_sha256": actual_content_hash,
        "declared_canonical_content_sha256": declared_hash,
        "protocol_invariants": "passed",
        "condition_target_counts": {
            name: len(memberships[name]) for name in CONDITION_NAMES
        },
        "wikitext_count": len(rows),
        "wikitext_membership_sha256": actual_background_hash,
    }
    return manifest, summary


def _verify_file_artifact(
    path: Path, expected: Mapping[str, Any], name: str
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    actual_hash = _sha256_file(path)
    if actual_hash != expected.get("sha256"):
        raise ValueError(
            f"{name} SHA-256 mismatch: expected {expected.get('sha256')}, "
            f"found {actual_hash}."
        )
    if path.stat().st_size != expected.get("bytes"):
        raise ValueError(f"{name} byte size differs from the frozen manifest.")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual_hash}


def _verify_target_dataset_storage(
    path: Path, expected: Mapping[str, Any]
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Target dataset does not exist: {path}")
    expected_files = expected.get("files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("Manifest target dataset has no frozen storage files.")
    actual_files = sorted(
        {
            file
            for pattern in ("data-*.arrow", "dataset_info.json", "state.json")
            for file in path.glob(pattern)
            if file.is_file()
        },
        key=lambda file: file.name,
    )
    expected_by_name = {
        str(_mapping(entry, "target dataset file")["path"]): entry
        for entry in expected_files
    }
    if [file.name for file in actual_files] != sorted(expected_by_name):
        raise ValueError("Target dataset storage file list differs from the manifest.")
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for file in actual_files:
        expected_file = _mapping(expected_by_name[file.name], file.name)
        file_hash = _sha256_file(file)
        size = file.stat().st_size
        if (
            file_hash != expected_file.get("sha256")
            or size != expected_file.get("bytes")
        ):
            raise ValueError(f"Target dataset storage file mismatch: {file.name}")
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        entries.append({"path": file.name, "bytes": size, "sha256": file_hash})
    storage_hash = digest.hexdigest()
    if storage_hash != expected.get("storage_sha256"):
        raise ValueError("Target dataset storage SHA-256 differs from the manifest.")
    return {"path": str(path), "storage_sha256": storage_hash, "files": entries}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _plain_value(value.tolist())
    return value


def _validate_background_dataset(
    train_dataset: Any,
    attack_dataset: Any,
    tokenizer: Any,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if len(attack_dataset) != ATTACK_SIZE:
        raise ValueError(f"Attack dataset has {len(attack_dataset)} rows, not 15,000.")
    if SOURCE_INDEX_COLUMN in train_dataset.column_names:
        raise ValueError(f"Unexpected WikiText column: {SOURCE_INDEX_COLUMN}")
    indexed_attack = (
        train_dataset.add_column(SOURCE_INDEX_COLUMN, list(range(len(train_dataset))))
        .shuffle(seed=SEED)
        .select(range(ATTACK_SIZE))
    )
    expected_rows = _mapping(
        manifest.get("shared_wikitext_background"), "shared_wikitext_background"
    )["rows"]
    actual_rows: list[dict[str, Any]] = []
    source_indices: list[int] = []
    for index in range(ATTACK_SIZE):
        sample = _plain_value(attack_dataset[index])
        indexed_sample = _plain_value(indexed_attack[index])
        source_index = int(indexed_sample.pop(SOURCE_INDEX_COLUMN))
        if sample != indexed_sample:
            raise ValueError(
                f"Indexed WikiText reconstruction diverged at attack row {index}."
            )
        token_ids = [int(token_id) for token_id in sample["input_ids"]]
        if not token_ids:
            raise ValueError(f"WikiText attack row {index} is empty.")
        text = sample.get("text")
        if text is None:
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        identity = {
            "row_id": f"wikitext_attack:{source_index}",
            "retained_index": TARGET_COUNT + index,
            "source_index": source_index,
            "text_sha256": _sha256_text(str(text)),
        }
        if identity != expected_rows[index]:
            raise ValueError(
                "Loaded WikiText background differs from the frozen manifest at "
                f"attack row {index}."
            )
        actual_rows.append(identity)
        source_indices.append(source_index)
    if len(set(source_indices)) != ATTACK_SIZE:
        raise ValueError("Loaded WikiText background contains duplicate source rows.")
    membership_hash = _canonical_sha256({"rows": actual_rows})
    expected_hash = manifest["shared_wikitext_background"]["membership_sha256"]
    if membership_hash != expected_hash:
        raise ValueError("Loaded WikiText membership hash differs from the manifest.")
    return {
        "source": "wikitext_attack",
        "selection": "train_dataset.shuffle(seed=42).select(range(15000))",
        "selection_seed": SEED,
        "count": ATTACK_SIZE,
        "membership_sha256": membership_hash,
        "ordered_source_indices_sha256": _canonical_sha256(
            {"source_indices": source_indices}
        ),
        "train_dataset_fingerprint": getattr(train_dataset, "_fingerprint", None),
        "attack_dataset_fingerprint": getattr(attack_dataset, "_fingerprint", None),
    }


def _load_ruli_utils(ruli_text_dir: Path) -> Any:
    expected_utils = (ruli_text_dir / "utils.py").resolve()
    expected_unlearner = (ruli_text_dir / "unlearner.py").resolve()
    for path in (expected_utils, expected_unlearner):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required upstream RULI source does not exist: {path}"
            )
    sys.path.insert(0, str(ruli_text_dir))
    module = importlib.import_module("utils")
    if Path(module.__file__).resolve() != expected_utils:
        raise ImportError(f"Imported the wrong utils module: {module.__file__}")
    return module


@contextlib.contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    original = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _reset_rng(seed: int, torch: Any, numpy: Any) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_sha256(model: Any, torch: Any) -> str:
    """Hash names, dtypes, shapes, and raw bytes for the complete state dict."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(cpu_tensor.dtype),
                "shape": list(cpu_tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw_bytes).to_bytes(8, "big"))
        digest.update(raw_bytes)
    return digest.hexdigest()


def _save_checkpoint(model: Any, tokenizer: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _git_metadata(path: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), *arguments],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _software_metadata(torch: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "safetensors",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cuda_devices": cuda_devices,
    }


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
            json.dump(
                payload,
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


def _cleanup_cuda(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("Experiment 2A currently supports only the frozen seed 42.")

    manifest, manifest_metadata = _load_and_validate_manifest(args.manifest)
    print(
        "[VERIFY] Frozen manifest passed: "
        f"SHA-256={manifest_metadata['file_sha256']}"
    )
    if args.validate_manifest_only:
        return

    ruli_root = args.ruli_root.resolve()
    ruli_text_dir = ruli_root / "text"
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
    target_data_path = (
        args.target_data_path.resolve()
        if args.target_data_path is not None
        else ruli_text_dir
        / "data"
        / "WikiText-103-local"
        / "gpt2"
        / "selective_dataset_prefixed_smoke_700"
    )
    input_artifacts = _mapping(manifest.get("input_artifacts"), "input_artifacts")
    shadow_metadata = _verify_file_artifact(
        shadow_path,
        _mapping(input_artifacts.get("shadow_artifact"), "shadow_artifact"),
        "Frozen 9-shadow artifact",
    )
    target_storage = _verify_target_dataset_storage(
        target_data_path,
        _mapping(input_artifacts.get("target_dataset"), "target_dataset"),
    )

    import numpy as np
    import torch
    from datasets import load_from_disk
    from torch.utils.data import ConcatDataset, Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        not args.validate_only
        and str(args.device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested {args.device}, but PyTorch reports no CUDA device."
        )

    ruli_source_files = {
        name: {
            "path": str((ruli_text_dir / name).resolve()),
            "sha256": _sha256_file(ruli_text_dir / name),
        }
        for name in ("mia_inference.py", "train_text.py", "unlearner.py", "utils.py")
        if (ruli_text_dir / name).is_file()
    }
    if set(ruli_source_files) != {
        "mia_inference.py",
        "train_text.py",
        "unlearner.py",
        "utils.py",
    }:
        raise FileNotFoundError(
            "RULI text sources mia_inference.py, train_text.py, unlearner.py, "
            "and utils.py are all required."
        )
    ruli_utils = _load_ruli_utils(ruli_text_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    target_dataset = load_from_disk(str(target_data_path))
    with _working_directory(ruli_text_dir):
        train_dataset, valid_dataset, _ = ruli_utils.load_data(
            "WikiText103", SimpleNamespace(model_name=MODEL_NAME)
        )
    if len(train_dataset) < ATTACK_SIZE:
        raise ValueError(
            "Filtered WikiText train data contains fewer than 15,000 rows."
        )
    attack_dataset = train_dataset.shuffle(seed=SEED).select(range(ATTACK_SIZE))

    shadow_results = torch.load(
        shadow_path, map_location="cpu", weights_only=False
    )
    if not isinstance(shadow_results, Mapping) or not isinstance(
        shadow_results.get("in_original"), Mapping
    ):
        raise ValueError("Shadow artifact has no in_original mapping.")
    actual_shadow_ids = sorted(int(key) for key in shadow_results["in_original"])
    expected_shadow_ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if actual_shadow_ids != expected_shadow_ids:
        raise ValueError("Loaded shadow ID order differs from the frozen manifest.")
    in_ids = actual_shadow_ids[:200]
    unlearn_ids = actual_shadow_ids[200:400]
    out_ids = actual_shadow_ids[400:600]
    if not (
        in_ids == manifest["evaluation_ids"]["in_ids"]
        and unlearn_ids == manifest["evaluation_ids"]["unlearn_ids"]
        and out_ids == manifest["evaluation_ids"]["out_ids"]
    ):
        raise ValueError("Loaded shadow partitions differ from the manifest.")
    all_condition_ids = {
        sample_id
        for name in CONDITION_NAMES
        for sample_id in manifest["conditions"][name]["ordered_target_dataset_ids"]
    }
    if min(all_condition_ids) < 0 or max(all_condition_ids) >= len(target_dataset):
        raise ValueError("A frozen condition ID falls outside the target dataset.")

    background_metadata = _validate_background_dataset(
        train_dataset, attack_dataset, tokenizer, manifest
    )
    print(
        "[VERIFY] Exact target, shadow, and 15,000-row WikiText artifacts match "
        "the frozen manifest."
    )

    condition_target_ids = {
        name: list(manifest["conditions"][name]["ordered_target_dataset_ids"])
        for name in CONDITION_NAMES
    }
    condition_datasets = {
        name: ConcatDataset(
            [Subset(target_dataset, condition_target_ids[name]), attack_dataset]
        )
        for name in CONDITION_NAMES
    }
    if any(
        dataset.datasets[1] is not attack_dataset
        for dataset in condition_datasets.values()
    ):
        raise AssertionError("Conditions do not share the identical WikiText object.")
    if any(
        len(dataset) != TARGET_COUNT + ATTACK_SIZE
        for dataset in condition_datasets.values()
    ):
        raise AssertionError("A final retain dataset does not contain 15,200 rows.")

    if args.validate_only:
        print("[VERIFY] Full pre-training validation passed; no model was trained.")
        return

    output_root = args.output_root.resolve()
    checkpoints = {
        "shared_post_npo_pre_final_ft": output_root / "post_npo_pre_final_ft",
        "HIGH": output_root / "HIGH_final",
        "LOW": output_root / "LOW_final",
        "PLACEBO": output_root / "PLACEBO_final",
    }
    metadata_path = output_root / "run_metadata.json"
    occupied = [
        path for path in [*checkpoints.values(), metadata_path] if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing Experiment 2A output(s): "
            + ", ".join(str(path) for path in occupied)
        )
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root = output_root / "trainer_work"

    in_data = Subset(target_dataset, in_ids)
    unlearn_data = Subset(target_dataset, unlearn_ids)
    initial_train_data = ConcatDataset([in_data, unlearn_data, attack_dataset])
    high_retain_data = condition_datasets["HIGH"]

    print("[INFO] Training common GPT-2 model: SFT=5 epochs, prefix=1 epoch.")
    _reset_rng(SEED, torch, np)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(args.device)
    model_config = model.config.to_dict()
    initial_pretrained_parameter_hash = _parameter_sha256(model, torch)
    with _working_directory(scratch_root / "common"):
        model = ruli_utils.train_sft(
            model, initial_train_data, valid_dataset, tokenizer, SFT_EPOCHS
        )
        model = ruli_utils.train_prefix(
            model, initial_train_data, valid_dataset, tokenizer, PREFIX_EPOCHS
        )
        unlearning_args = SimpleNamespace(
            device=args.device,
            unlearn_epochs=NPO_EPOCHS,
            unlearn_method="npo",
        )
        print("[INFO] Running common upstream RULI NPO for 15 epochs.")
        model = ruli_utils.unlearn_model(
            model,
            unlearn_data,
            high_retain_data,
            valid_dataset,
            tokenizer,
            unlearning_args,
        )

    shared_parameter_hash = _parameter_sha256(model, torch)
    _save_checkpoint(
        model, tokenizer, checkpoints["shared_post_npo_pre_final_ft"]
    )
    print(
        "[VERIFY] Saved shared post-NPO checkpoint with parameter SHA-256 "
        f"{shared_parameter_hash}."
    )
    del model
    _cleanup_cuda(torch)

    starting_hashes: dict[str, str] = {}
    final_hashes: dict[str, str] = {}
    for condition_name in CONDITION_NAMES:
        print(f"[INFO] Training independent {condition_name} final-FT branch.")
        _reset_rng(SEED, torch, np)
        condition_model = AutoModelForCausalLM.from_pretrained(
            checkpoints["shared_post_npo_pre_final_ft"]
        ).to(args.device)
        starting_hash = _parameter_sha256(condition_model, torch)
        starting_hashes[condition_name] = starting_hash
        if starting_hash != shared_parameter_hash:
            raise RuntimeError(
                f"{condition_name} did not load byte-identical post-NPO parameters."
            )
        _reset_rng(SEED, torch, np)
        with _working_directory(scratch_root / condition_name):
            condition_model = ruli_utils.train_sft(
                condition_model,
                condition_datasets[condition_name],
                valid_dataset,
                tokenizer,
                FINAL_FT_EPOCHS,
            )
        final_hashes[condition_name] = _parameter_sha256(condition_model, torch)
        _save_checkpoint(
            condition_model, tokenizer, checkpoints[condition_name]
        )
        del condition_model
        _cleanup_cuda(torch)
    if len(set(starting_hashes.values())) != 1:
        raise RuntimeError("Condition starting parameter hashes are not identical.")

    metadata = {
        "schema_version": 1,
        "experiment": "2A",
        "seed": SEED,
        "manifest": manifest_metadata,
        "git": {
            "ruli": _git_metadata(ruli_root),
            "ruli_experiments": _git_metadata(REPOSITORY_ROOT),
        },
        "upstream_ruli_source_files": ruli_source_files,
        "model_and_hyperparameters": {
            "model": MODEL_NAME,
            "model_config": model_config,
            "initial_pretrained_parameter_sha256": (
                initial_pretrained_parameter_hash
            ),
            "tokenizer": {
                "class": tokenizer.__class__.__name__,
                "pad_token": tokenizer.pad_token,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token": tokenizer.eos_token,
                "eos_token_id": tokenizer.eos_token_id,
            },
            "initial_sft_epochs": SFT_EPOCHS,
            "prefix_epochs": PREFIX_EPOCHS,
            "unlearn_method": "npo",
            "npo_epochs": NPO_EPOCHS,
            "final_retain_sft_epochs": FINAL_FT_EPOCHS,
            "attack_size": ATTACK_SIZE,
            "device": args.device,
            "training_implementation": "imported from Ruli/text/utils.py",
            "upstream_effective_settings": {
                "sft": {
                    "per_device_train_batch_size": 16,
                    "per_device_eval_batch_size": 16,
                    "learning_rate": 5e-5,
                    "weight_decay": 0.01,
                    "evaluation_strategy": "epoch",
                    "save_strategy": "epoch",
                    "load_best_model_at_end": True,
                    "metric_for_best_model": "eval_loss",
                    "early_stopping_patience": 2,
                },
                "prefix": {
                    "loss_type": "gdr",
                    "per_device_train_batch_size": 4,
                    "learning_rate": 1e-5,
                },
                "npo": {
                    "loss_type": "npo",
                    "beta": 0.1,
                    "per_device_train_batch_size": 16,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 5e-5,
                },
            },
        },
        "ordered_target_dataset_ids": condition_target_ids,
        "background_dataset": background_metadata,
        "input_artifacts": {
            "shadow": shadow_metadata,
            "target_dataset": target_storage,
        },
        "dataset_sizes": {
            "target_dataset": len(target_dataset),
            "wikitext_filtered_train": len(train_dataset),
            "wikitext_validation": len(valid_dataset),
            "initial_sft_and_prefix": len(initial_train_data),
            "npo_forget": len(unlearn_data),
            "npo_retain": len(high_retain_data),
            "final_retain": {
                name: len(condition_datasets[name]) for name in CONDITION_NAMES
            },
            "condition_target": {
                name: len(condition_target_ids[name]) for name in CONDITION_NAMES
            },
        },
        "checkpoints": {
            "shared_post_npo_pre_final_ft": {
                "path": str(checkpoints["shared_post_npo_pre_final_ft"]),
                "parameter_sha256": shared_parameter_hash,
            },
            **{
                name: {
                    "path": str(checkpoints[name]),
                    "starting_parameter_sha256": starting_hashes[name],
                    "final_parameter_sha256": final_hashes[name],
                }
                for name in CONDITION_NAMES
            },
        },
        "starting_parameter_identity": {
            "passed": len(set(starting_hashes.values())) == 1
            and next(iter(starting_hashes.values())) == shared_parameter_hash,
            "sha256": shared_parameter_hash,
            "method": (
                "SHA-256 over sorted state_dict names, dtypes, shapes, and raw "
                "contiguous tensor bytes before each condition's final FT"
            ),
        },
        "rng_policy": {
            "python_random": SEED,
            "numpy": SEED,
            "torch_cpu": SEED,
            "torch_cuda_all": SEED,
            "reset_before_each_checkpoint_load": True,
            "reset_immediately_before_each_final_sft": True,
        },
        "software": _software_metadata(torch),
    }
    _atomic_json(metadata, metadata_path)
    print(f"[INFO] Experiment 2A training complete. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
