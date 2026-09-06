#!/usr/bin/env python3
"""Run one preregistered seed of the frozen Experiment 2B intervention.

Experiment 2B branches before initial SFT.  Every condition loads one persisted
GPT-2 starting state, then independently runs the unmodified upstream RULI SFT,
prefix-training, NPO, and final retain-fine-tuning helpers with its frozen retain
configuration.  The intervention manifest is immutable input.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
DEFAULT_MANIFEST = SCRIPT_DIR / "results" / "intervention_manifest.json"

MODEL_NAME = "gpt2"
DEFAULT_TRAINING_SEED = 42
PREREGISTERED_TRAINING_SEEDS = (42, 43, 44, 45, 46)
FROZEN_ARTIFACT_SEED = 42
SFT_EPOCHS = 5
PREFIX_EPOCHS = 1
NPO_EPOCHS = 15
FINAL_FT_EPOCHS = 2
ATTACK_SIZE = 15_000
TARGET_COUNT = 200
UNLEARN_COUNT = 200
CONDITION_NAMES = ("HIGH", "LOW", "PLACEBO")


def _load_experiment_2a_runner() -> Any:
    path = SCRIPT_DIR / "run_experiment_2a.py"
    spec = importlib.util.spec_from_file_location("_experiment_2a_runner_for_2b", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Experiment 2A runner helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = _load_experiment_2a_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one preregistered Experiment 2B seed by independently running "
            "HIGH, LOW, and PLACEBO from one byte-identical GPT-2 starting state."
        )
    )
    parser.add_argument("--ruli-root", type=Path, default=DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to results/experiment_2b/seed_<SEED>.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    validation_mode = parser.add_mutually_exclusive_group()
    validation_mode.add_argument(
        "--validate-manifest-only",
        action="store_true",
        help="Validate only the immutable manifest; load no RULI artifacts or model.",
    )
    validation_mode.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate all frozen inputs, dataset construction, and shared initial "
            "parameter identity without training or writing experiment outputs."
        ),
    )
    return parser.parse_args()


def _validate_training_seed(seed: int) -> None:
    if seed not in PREREGISTERED_TRAINING_SEEDS:
        allowed = ", ".join(str(value) for value in PREREGISTERED_TRAINING_SEEDS)
        raise ValueError(
            f"Experiment 2B seed must be preregistered; expected one of {allowed}, "
            f"found {seed}."
        )


def _seed_output_root(seed: int) -> Path:
    return SCRIPT_DIR / "results" / "experiment_2b" / f"seed_{seed}"


def _validate_seed_output_path(path: Path, seed: int, argument: str) -> Path:
    return COMMON._validate_seed_output_path(path, seed, argument)


def _integer_ids(value: Any, description: str) -> list[int]:
    return COMMON._integer_list(value, description)


def _validate_experiment_2b_protocol(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Apply the Experiment 2B-specific, fail-closed frozen-design checks."""

    sets = COMMON._mapping(manifest.get("sets"), "manifest.sets")
    evaluation = COMMON._mapping(
        manifest.get("evaluation_ids"), "manifest.evaluation_ids"
    )
    conditions = COMMON._mapping(manifest.get("conditions"), "manifest.conditions")

    s_ids = _integer_ids(sets.get("S_sample_ids"), "sets.S_sample_ids")
    u_ids = _integer_ids(sets.get("U_sample_ids"), "sets.U_sample_ids")
    p_ids = _integer_ids(sets.get("P_sample_ids"), "sets.P_sample_ids")
    r_ids = _integer_ids(sets.get("R_sample_ids"), "sets.R_sample_ids")
    if len(s_ids) != 28 or len(set(s_ids)) != 28:
        raise ValueError("Experiment 2B requires exactly 28 unique supported S IDs.")
    for name, ids in (("U", u_ids), ("P", p_ids), ("R", r_ids)):
        if len(ids) != 32 or len(set(ids)) != 32:
            raise ValueError(f"Experiment 2B requires exactly 32 unique {name} IDs.")

    in_ids = _integer_ids(evaluation.get("in_ids"), "evaluation_ids.in_ids")
    unlearn_ids = _integer_ids(
        evaluation.get("unlearn_ids"), "evaluation_ids.unlearn_ids"
    )
    out_ids = _integer_ids(evaluation.get("out_ids"), "evaluation_ids.out_ids")
    if any(len(ids) != TARGET_COUNT for ids in (in_ids, unlearn_ids, out_ids)):
        raise ValueError("IN, UNLEARN, and OUT must each contain exactly 200 IDs.")
    official_ids = set(in_ids) | set(unlearn_ids) | set(out_ids)
    if set(r_ids) & official_ids:
        raise ValueError("Frozen R overlaps official IN, UNLEARN, or OUT IDs.")

    condition_target_ids = {
        name: _integer_ids(
            COMMON._mapping(conditions.get(name), f"conditions.{name}").get(
                "ordered_target_dataset_ids"
            ),
            f"conditions.{name}.ordered_target_dataset_ids",
        )
        for name in CONDITION_NAMES
    }
    for name, ids in condition_target_ids.items():
        if len(ids) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT:
            raise ValueError(f"{name} must contain exactly 200 unique target IDs.")

    high_set = set(in_ids)
    u_set, p_set, r_set = set(u_ids), set(p_ids), set(r_ids)
    if condition_target_ids["HIGH"] != in_ids:
        raise ValueError("HIGH must preserve the ordered official IN partition.")
    if set(condition_target_ids["LOW"]) != (high_set - u_set) | r_set:
        raise ValueError("LOW must be exactly (official IN - U) + R.")
    if set(condition_target_ids["PLACEBO"]) != (high_set - p_set) | r_set:
        raise ValueError("PLACEBO must be exactly (official IN - P) + R.")
    low = COMMON._mapping(conditions["LOW"], "conditions.LOW")
    placebo = COMMON._mapping(conditions["PLACEBO"], "conditions.PLACEBO")
    if (
        set(_integer_ids(low.get("removed_original_target_ids"), "LOW removed"))
        != u_set
    ):
        raise ValueError("LOW removes IDs other than exactly frozen U.")
    if set(
        _integer_ids(placebo.get("removed_original_target_ids"), "PLACEBO removed")
    ) != p_set:
        raise ValueError("PLACEBO removes IDs other than exactly frozen P.")
    low_r = _integer_ids(low.get("replacement_target_ids"), "LOW replacements")
    placebo_r = _integer_ids(
        placebo.get("replacement_target_ids"), "PLACEBO replacements"
    )
    if low_r != r_ids or placebo_r != r_ids or low_r != placebo_r:
        raise ValueError("LOW and PLACEBO must add the exact same ordered frozen R.")

    background = COMMON._mapping(
        manifest.get("shared_wikitext_background"),
        "manifest.shared_wikitext_background",
    )
    if background.get("count") != ATTACK_SIZE:
        raise ValueError("Experiment 2B requires exactly 15,000 WikiText rows.")

    validation = {
        "manifest_protocol_invariants": True,
        "S_count_is_28": True,
        "U_P_R_counts_are_32": True,
        "condition_target_counts_are_200": True,
        "LOW_is_HIGH_minus_U_plus_R": True,
        "PLACEBO_is_HIGH_minus_P_plus_same_R": True,
        "R_disjoint_from_official_evaluation_ids": True,
        "shared_wikitext_count_is_15000": True,
        "unlearn_partition_count_is_200": len(unlearn_ids) == UNLEARN_COUNT,
        "out_partition_count_is_200": len(out_ids) == TARGET_COUNT,
    }
    if not all(validation.values()):
        raise AssertionError("An Experiment 2B protocol validation did not pass.")
    return condition_target_ids, validation


def _assert_initial_parameter_identity(
    shared_hash: str, branch_hashes: Mapping[str, str], stage: str
) -> dict[str, Any]:
    if set(branch_hashes) != set(CONDITION_NAMES):
        raise RuntimeError(f"{stage} did not include all three condition branches.")
    mismatches = {
        name: value for name, value in branch_hashes.items() if value != shared_hash
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={value}" for name, value in sorted(mismatches.items())
        )
        raise RuntimeError(
            f"{stage} initial parameters differ from shared state "
            f"{shared_hash}: {detail}"
        )
    return {
        "passed": True,
        "shared_sha256": shared_hash,
        "condition_sha256": dict(branch_hashes),
    }


def _create_and_verify_shared_initial(
    checkpoint: Path,
    seed: int,
    auto_model: Any,
    tokenizer: Any,
    torch: Any,
    numpy: Any,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    """Persist one initial model and prove three independent reloads match it."""

    COMMON._reset_rng(seed, torch, numpy)
    initial_model = auto_model.from_pretrained(MODEL_NAME)
    model_config = initial_model.config.to_dict()
    shared_hash = COMMON._parameter_sha256(initial_model, torch)
    COMMON._save_checkpoint(initial_model, tokenizer, checkpoint)
    del initial_model
    gc.collect()

    branch_hashes: dict[str, str] = {}
    for condition in CONDITION_NAMES:
        COMMON._reset_rng(seed, torch, numpy)
        branch = auto_model.from_pretrained(checkpoint)
        branch_hashes[condition] = COMMON._parameter_sha256(branch, torch)
        del branch
        gc.collect()
    _assert_initial_parameter_identity(shared_hash, branch_hashes, "Pre-training check")
    return shared_hash, model_config, branch_hashes


def _artifact_paths(args: argparse.Namespace, ruli_root: Path) -> tuple[Path, Path]:
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
    return shadow_path, target_data_path


def _checkpoint_paths(output_root: Path) -> dict[str, Path]:
    paths = {"initial_shared": output_root / "initial_shared"}
    for condition in CONDITION_NAMES:
        paths[f"{condition}_pre_npo"] = output_root / f"{condition}_pre_npo"
        paths[f"{condition}_final"] = output_root / f"{condition}_final"
    return paths


def _build_condition_datasets(
    target_dataset: Any,
    attack_dataset: Any,
    unlearn_ids: list[int],
    condition_target_ids: Mapping[str, list[int]],
    subset_type: Any,
    concat_type: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the four frozen injection points from shared dataset objects."""

    unlearn_data = subset_type(target_dataset, unlearn_ids)
    condition_target_data = {
        name: subset_type(target_dataset, condition_target_ids[name])
        for name in CONDITION_NAMES
    }
    condition_retain_data = {
        name: concat_type([condition_target_data[name], attack_dataset])
        for name in CONDITION_NAMES
    }
    condition_initial_data = {
        name: concat_type(
            [condition_target_data[name], unlearn_data, attack_dataset]
        )
        for name in CONDITION_NAMES
    }
    if any(
        dataset.datasets[-1] is not attack_dataset
        for dataset in (
            *condition_retain_data.values(),
            *condition_initial_data.values(),
        )
    ):
        raise AssertionError("Conditions do not share the identical WikiText object.")
    if any(
        dataset.datasets[1] is not unlearn_data
        for dataset in condition_initial_data.values()
    ):
        raise AssertionError("Conditions do not share the identical UNLEARN object.")
    if any(
        len(data) != TARGET_COUNT + ATTACK_SIZE
        for data in condition_retain_data.values()
    ):
        raise AssertionError("A condition retain dataset does not contain 15,200 rows.")
    if any(
        len(data) != TARGET_COUNT + UNLEARN_COUNT + ATTACK_SIZE
        for data in condition_initial_data.values()
    ):
        raise AssertionError(
            "A condition initial dataset does not contain 15,400 rows."
        )
    return (
        unlearn_data,
        condition_target_data,
        condition_retain_data,
        condition_initial_data,
    )


def main() -> None:
    args = parse_args()
    _validate_training_seed(args.seed)
    manifest, manifest_metadata = COMMON._load_and_validate_manifest(args.manifest)
    condition_target_ids, protocol_validation = _validate_experiment_2b_protocol(
        manifest
    )
    print(
        "[VERIFY] Frozen Experiment 2B manifest passed: "
        f"SHA-256={manifest_metadata['file_sha256']}"
    )
    if args.validate_manifest_only:
        return

    ruli_root = args.ruli_root.resolve()
    ruli_text_dir = ruli_root / "text"
    shadow_path, target_data_path = _artifact_paths(args, ruli_root)
    input_artifacts = COMMON._mapping(
        manifest.get("input_artifacts"), "input_artifacts"
    )
    shadow_metadata = COMMON._verify_file_artifact(
        shadow_path,
        COMMON._mapping(input_artifacts.get("shadow_artifact"), "shadow_artifact"),
        "Frozen 9-shadow artifact",
    )
    target_storage = COMMON._verify_target_dataset_storage(
        target_data_path,
        COMMON._mapping(input_artifacts.get("target_dataset"), "target_dataset"),
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
            "sha256": COMMON._sha256_file(ruli_text_dir / name),
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

    ruli_utils = COMMON._load_ruli_utils(ruli_text_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    target_dataset = load_from_disk(str(target_data_path))
    with COMMON._working_directory(ruli_text_dir):
        train_dataset, valid_dataset, _ = ruli_utils.load_data(
            "WikiText103", SimpleNamespace(model_name=MODEL_NAME)
        )
    if len(train_dataset) < ATTACK_SIZE:
        raise ValueError(
            "Filtered WikiText train data contains fewer than 15,000 rows."
        )
    attack_dataset = train_dataset.shuffle(seed=FROZEN_ARTIFACT_SEED).select(
        range(ATTACK_SIZE)
    )

    shadow_results = torch.load(shadow_path, map_location="cpu", weights_only=False)
    if not isinstance(shadow_results, Mapping) or not isinstance(
        shadow_results.get("in_original"), Mapping
    ):
        raise ValueError("Shadow artifact has no in_original mapping.")
    actual_shadow_ids = sorted(int(key) for key in shadow_results["in_original"])
    expected_shadow_ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if actual_shadow_ids != expected_shadow_ids:
        raise ValueError("Loaded shadow target IDs differ from the frozen assumptions.")
    if (
        actual_shadow_ids[:200] != manifest["evaluation_ids"]["in_ids"]
        or actual_shadow_ids[200:400] != manifest["evaluation_ids"]["unlearn_ids"]
        or actual_shadow_ids[400:600] != manifest["evaluation_ids"]["out_ids"]
    ):
        raise ValueError("Loaded shadow partitions differ from the frozen manifest.")

    all_target_ids = {
        sample_id for ids in condition_target_ids.values() for sample_id in ids
    }
    if min(all_target_ids) < 0 or max(all_target_ids) >= len(target_dataset):
        raise ValueError(
            "A frozen Experiment 2B target ID is outside the target dataset."
        )
    background_metadata = COMMON._validate_background_dataset(
        train_dataset, attack_dataset, tokenizer, manifest
    )

    unlearn_ids = list(manifest["evaluation_ids"]["unlearn_ids"])
    (
        unlearn_data,
        condition_target_data,
        condition_retain_data,
        condition_initial_data,
    ) = _build_condition_datasets(
        target_dataset,
        attack_dataset,
        unlearn_ids,
        condition_target_ids,
        Subset,
        ConcatDataset,
    )
    print(
        "[VERIFY] Target, shadow, condition retain sets, shared 15,000-row "
        "WikiText background, UNLEARN, and OUT partitions match the frozen design."
    )

    output_root: Path | None = None
    checkpoints: dict[str, Path] | None = None
    metadata_path: Path | None = None
    scratch_root: Path | None = None
    if not args.validate_only:
        output_root = _validate_seed_output_path(
            args.output_root
            if args.output_root is not None
            else _seed_output_root(args.seed),
            args.seed,
            "--output-root",
        )
        checkpoints = _checkpoint_paths(output_root)
        metadata_path = output_root / "run_metadata.json"
        occupied = [
            path for path in [*checkpoints.values(), metadata_path] if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                "Refusing to overwrite existing Experiment 2B output(s): "
                + ", ".join(str(path) for path in occupied)
            )
        output_root.mkdir(parents=True, exist_ok=True)
        scratch_root = output_root / "trainer_work"

    if args.validate_only:
        with tempfile.TemporaryDirectory(prefix="experiment_2b_initial_") as directory:
            initial_path = Path(directory) / "initial_shared"
            shared_hash, _, preflight_hashes = _create_and_verify_shared_initial(
                initial_path,
                args.seed,
                AutoModelForCausalLM,
                tokenizer,
                torch,
                np,
            )
        print(
            "[VERIFY] Validation-only initial branches are byte-identical: "
            f"{shared_hash}; HIGH={preflight_hashes['HIGH']}, "
            f"LOW={preflight_hashes['LOW']}, PLACEBO={preflight_hashes['PLACEBO']}."
        )
        print("[VERIFY] Full Experiment 2B preflight passed; no model was trained.")
        return

    assert checkpoints is not None
    assert metadata_path is not None
    assert scratch_root is not None
    shared_hash, model_config, preflight_hashes = _create_and_verify_shared_initial(
        checkpoints["initial_shared"],
        args.seed,
        AutoModelForCausalLM,
        tokenizer,
        torch,
        np,
    )
    print(
        "[VERIFY] Persisted one shared initial GPT-2 state and independently "
        f"reloaded all three branches with parameter SHA-256 {shared_hash}."
    )

    COMMON._configure_training_arguments_seed(ruli_utils, args.seed)
    actual_starting_hashes: dict[str, str] = {}
    pre_npo_hashes: dict[str, str] = {}
    post_npo_hashes: dict[str, str] = {}
    final_hashes: dict[str, str] = {}
    for condition in CONDITION_NAMES:
        print(f"[INFO] Running complete Experiment 2B pipeline for {condition}.")
        COMMON._reset_rng(args.seed, torch, np)
        model = AutoModelForCausalLM.from_pretrained(checkpoints["initial_shared"])
        starting_hash = COMMON._parameter_sha256(model, torch)
        actual_starting_hashes[condition] = starting_hash
        if starting_hash != shared_hash:
            raise RuntimeError(
                f"{condition} did not load byte-identical shared initial parameters."
            )
        model = model.to(args.device)
        COMMON._reset_rng(args.seed, torch, np)
        condition_work = scratch_root / condition
        with COMMON._working_directory(condition_work / "initial_sft"):
            model = ruli_utils.train_sft(
                model,
                condition_initial_data[condition],
                valid_dataset,
                tokenizer,
                SFT_EPOCHS,
            )
        with COMMON._working_directory(condition_work / "prefix"):
            model = ruli_utils.train_prefix(
                model,
                condition_initial_data[condition],
                valid_dataset,
                tokenizer,
                PREFIX_EPOCHS,
            )
            pre_npo_hashes[condition] = COMMON._parameter_sha256(model, torch)
            COMMON._save_checkpoint(
                model, tokenizer, checkpoints[f"{condition}_pre_npo"]
            )
        unlearning_args = SimpleNamespace(
            device=args.device,
            unlearn_epochs=NPO_EPOCHS,
            unlearn_method="npo",
        )
        with COMMON._working_directory(condition_work / "npo"):
            model = ruli_utils.unlearn_model(
                model,
                unlearn_data,
                condition_retain_data[condition],
                valid_dataset,
                tokenizer,
                unlearning_args,
            )
        post_npo_hashes[condition] = COMMON._parameter_sha256(model, torch)
        with COMMON._working_directory(condition_work / "final_retain_ft"):
            model = ruli_utils.train_sft(
                model,
                condition_retain_data[condition],
                valid_dataset,
                tokenizer,
                FINAL_FT_EPOCHS,
            )
        final_hashes[condition] = COMMON._parameter_sha256(model, torch)
        COMMON._save_checkpoint(model, tokenizer, checkpoints[f"{condition}_final"])
        del model
        COMMON._cleanup_cuda(torch)

    actual_identity = _assert_initial_parameter_identity(
        shared_hash, actual_starting_hashes, "Actual training branches"
    )
    metadata = {
        "schema_version": 1,
        "experiment": "2B",
        "seed": args.seed,
        "manifest": manifest_metadata,
        "git": {
            "ruli": COMMON._git_metadata(ruli_root),
            "ruli_experiments": COMMON._git_metadata(REPOSITORY_ROOT),
        },
        "upstream_ruli_source_files": ruli_source_files,
        "model_and_hyperparameters": {
            "model": MODEL_NAME,
            "model_config": model_config,
            "initial_model_parameter_sha256": shared_hash,
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
            "training_implementation": "imported unchanged from Ruli/text/utils.py",
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
            "target_dataset": {
                **target_storage,
                "loaded_fingerprint": getattr(target_dataset, "_fingerprint", None),
            },
        },
        "dataset_sizes": {
            "target_dataset": len(target_dataset),
            "wikitext_filtered_train": len(train_dataset),
            "wikitext_validation": len(valid_dataset),
            "condition_target": {
                name: len(condition_target_data[name]) for name in CONDITION_NAMES
            },
            "initial_sft_and_prefix": {
                name: len(condition_initial_data[name]) for name in CONDITION_NAMES
            },
            "npo_forget": len(unlearn_data),
            "npo_retain": {
                name: len(condition_retain_data[name]) for name in CONDITION_NAMES
            },
            "final_retain": {
                name: len(condition_retain_data[name]) for name in CONDITION_NAMES
            },
        },
        "dataset_injection": {
            name: {
                "initial_sft": "condition target-IN + shared UNLEARN + shared WikiText",
                "prefix_training": (
                    "condition target-IN + shared UNLEARN + shared WikiText"
                ),
                "npo_forget": "shared official UNLEARN",
                "npo_retain": "condition target-IN + shared WikiText",
                "final_retain_ft": "condition target-IN + shared WikiText",
            }
            for name in CONDITION_NAMES
        },
        "checkpoints": {
            "initial_shared": {
                "path": str(checkpoints["initial_shared"]),
                "parameter_sha256": shared_hash,
            },
            **{
                name: {
                    "starting_parameter_sha256": actual_starting_hashes[name],
                    "pre_npo": {
                        "path": str(checkpoints[f"{name}_pre_npo"]),
                        "parameter_sha256": pre_npo_hashes[name],
                    },
                    "post_npo_parameter_sha256": post_npo_hashes[name],
                    "final": {
                        "path": str(checkpoints[f"{name}_final"]),
                        "parameter_sha256": final_hashes[name],
                    },
                }
                for name in CONDITION_NAMES
            },
        },
        "starting_parameter_identity": {
            "passed": actual_identity["passed"],
            "sha256": shared_hash,
            "preflight_reload_sha256": preflight_hashes,
            "actual_training_branch_sha256": actual_starting_hashes,
            "method": (
                "SHA-256 over sorted state_dict names, dtypes, shapes, and raw "
                "contiguous tensor bytes from one persisted initial_shared model; "
                "verified on three independent preflight reloads and again on each "
                "actual training branch before SFT"
            ),
        },
        "rng_policy": {
            "training_seed": args.seed,
            "python_random": args.seed,
            "numpy": args.seed,
            "torch_cpu": args.seed,
            "torch_cuda_all": args.seed,
            "transformers_training_arguments_seed": args.seed,
            "transformers_training_arguments_data_seed": args.seed,
            "frozen_wikitext_selection_seed": FROZEN_ARTIFACT_SEED,
            "reset_before_each_preflight_reload": True,
            "reset_before_each_actual_condition_load": True,
            "reset_immediately_before_each_condition_pipeline": True,
        },
        "validation_results": {
            **protocol_validation,
            "frozen_manifest_sha256": True,
            "target_dataset_storage_identity": True,
            "fixed_shadow_artifact_identity": True,
            "fixed_shadow_target_partition": True,
            "shared_wikitext_row_identity": True,
            "shared_initial_parameter_identity": True,
            "no_post_npo_identity_requirement": True,
        },
        "software": COMMON._software_metadata(torch),
        "deviations_from_reference_ruli_behavior": [],
    }
    COMMON._atomic_json(metadata, metadata_path)
    print(f"[INFO] Experiment 2B training complete. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
