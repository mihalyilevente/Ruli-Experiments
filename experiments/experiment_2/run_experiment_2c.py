#!/usr/bin/env python3
"""Frozen Experiment 2C: condition-specific SFT -> prefix -> pre -> NPO -> post."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(filename: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module("run_experiment_2b.py", "_2b_for_2c")
REFERENCE = _load_module("evaluate_experiment_2a.py", "_2a_eval_for_2c")
COMMON = BASE.COMMON
CONDITIONS = BASE.CONDITION_NAMES
STAGES = ("initial_sft", "prefix", "save_pre_npo", "npo", "save_post_npo")
HYPERPARAMETERS = {
    "model": "gpt2",
    "initial_sft_epochs": 5,
    "prefix_epochs": 1,
    "unlearn_method": "npo",
    "npo_epochs": 15,
    "final_retain_sft_epochs": 0,
    "attack_size": 15_000,
}
CAPTURE = {
    "pre_npo": "after_initial_sft_and_prefix_before_any_npo_update",
    "post_npo": "immediately_after_npo_before_any_retain_ft",
}


def _seed_output_root(seed: int) -> Path:
    return SCRIPT_DIR / "results" / "experiment_2c" / f"seed_{seed}"


def _output_root(path: Path | None, seed: int) -> Path:
    root = BASE._validate_seed_output_path(
        path if path is not None else _seed_output_root(seed), seed, "output"
    )
    if any(part.lower() in ("experiment_2a", "experiment_2b") for part in root.parts):
        raise ValueError("Experiment 2C output must not be inside 2A or 2B outputs.")
    return root


def _checkpoint_paths(root: Path) -> dict[str, Path]:
    return {
        "initial_shared": root / "initial_shared",
        **{f"{c}_{stage}": root / f"{c}_{stage}"
           for c in CONDITIONS for stage in CAPTURE},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        choices=BASE.PREREGISTERED_TRAINING_SEEDS)
    parser.add_argument("--ruli-root", type=Path, default=BASE.DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=BASE.DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-manifest-only", action="store_true",
                      help="Check the exact frozen manifest without loading artifacts.")
    mode.add_argument("--validate-only", action="store_true",
                      help="Full CPU preflight, including initial reloads; no training.")
    return parser.parse_args()


def _load_shadow(path, expected, manifest, torch):
    """Extend the validated 2A shadow loader with the original UNLEARN state."""
    shadow, metadata = REFERENCE._load_and_validate_shadow(
        path, expected, manifest, torch
    )
    raw = torch.load(path, map_location="cpu", weights_only=False)
    original = REFERENCE._normalize_shadow_mapping(
        raw.get("unlearn_original"), "unlearn_original"
    )
    ids = manifest["evaluation_ids"]["unlearn_ids"]
    if any(sample_id not in original for sample_id in ids):
        raise ValueError("Original-state shadows cannot align all UNLEARN IDs.")
    shadow["unlearn_original"] = {
        sample_id: REFERENCE._plain_shadow_observations(
            original[sample_id], "unlearn_original", sample_id
        ) for sample_id in ids
    }
    metadata["original_unlearn_observations_validated"] = len(ids)
    return shadow, metadata


def _kde_references(shadow, ids, gaussian_kde):
    definitions = {
        "pre_npo": ("unlearn_original", "out_original"),
        "post_npo": ("unlearn_unlearned", "out_unlearned"),
    }
    result = {}
    for stage, fields in definitions.items():
        result[stage] = {}
        for sample_id in ids:
            try:
                result[stage][sample_id] = tuple(
                    gaussian_kde(shadow[field][sample_id]) for field in fields
                )
            except Exception as exc:
                raise ValueError(f"Invalid {stage} KDE for sample {sample_id}") from exc
    return result


def _source_metadata(ruli_root: Path) -> dict:
    return {
        name: {"path": str(ruli_root / "text" / name),
               "sha256": COMMON._sha256_file(ruli_root / "text" / name)}
        for name in ("mia_inference.py", "train_text.py", "unlearner.py", "utils.py")
    }


def _run_branch(model, tokenizer, initial_data, retain_data, unlearn_data,
                valid_data, condition, checkpoints, scratch_root, ruli_utils,
                device, torch):
    """Only these three upstream calls can update a branch's parameters."""
    work = scratch_root / condition
    cleanup = {}
    events = []
    with COMMON._working_directory(work / "initial_sft"):
        model = ruli_utils.train_sft(
            model, initial_data, valid_data, tokenizer, BASE.SFT_EPOCHS
        )
    events.append("initial_sft")
    cleanup["initial_sft"] = BASE._cleanup_successful_trainer_stage(
        work / "initial_sft", scratch_root
    )
    with COMMON._working_directory(work / "prefix"):
        model = ruli_utils.train_prefix(
            model, initial_data, valid_data, tokenizer, BASE.PREFIX_EPOCHS
        )
    events.append("prefix")

    def capture(stage):
        path = checkpoints[f"{condition}_{stage}"]
        digest = COMMON._parameter_sha256(model, torch)
        COMMON._save_checkpoint(model, tokenizer, path)
        events.append(f"save_{stage}")
        return {"path": str(path), "parameter_sha256": digest,
                "capture": CAPTURE[stage], "final_retain_ft_updates": 0}

    pre = capture("pre_npo")
    cleanup["prefix"] = BASE._cleanup_successful_trainer_stage(
        work / "prefix", scratch_root
    )
    with COMMON._working_directory(work / "npo"):
        model = ruli_utils.unlearn_model(
            model, unlearn_data, retain_data, valid_data, tokenizer,
            SimpleNamespace(device=device, unlearn_epochs=BASE.NPO_EPOCHS,
                            unlearn_method="npo"),
        )
    events.append("npo")
    post = capture("post_npo")
    cleanup["npo"] = BASE._cleanup_successful_trainer_stage(
        work / "npo", scratch_root
    )
    BASE._remove_empty_scratch_parents(work, scratch_root)
    if tuple(events) != STAGES:
        raise RuntimeError("Experiment 2C checkpoint capture order changed.")
    return {"pre_npo": pre, "post_npo": post, "stage_order": events}, cleanup


def main() -> None:
    args = parse_args()
    manifest, manifest_metadata = REFERENCE._validate_manifest(args.manifest)
    condition_ids, validation = BASE._validate_experiment_2b_protocol(manifest)
    output_root = _output_root(args.output_root, args.seed)
    print("[VERIFY] Exact frozen Experiment 2 manifest and 2B memberships passed.")
    if args.validate_manifest_only:
        return

    ruli_root = args.ruli_root.resolve()
    shadow_path, target_path = BASE._artifact_paths(args, ruli_root)
    artifacts = manifest["input_artifacts"]
    # Check exact files before importing ML packages or downloading any model.
    COMMON._verify_file_artifact(
        shadow_path, artifacts["shadow_artifact"], "Frozen 9-shadow artifact"
    )
    COMMON._verify_target_dataset_storage(target_path, artifacts["target_dataset"])

    import numpy as np
    import torch
    from datasets import load_from_disk
    from torch.utils.data import ConcatDataset, Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.validate_only and str(args.device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable.")
    sources = _source_metadata(ruli_root)
    ruli_utils = COMMON._load_ruli_utils(ruli_root / "text")
    loss_metadata = REFERENCE._validate_reference_loss_path(ruli_utils, torch)
    shadow, shadow_metadata = _load_shadow(
        shadow_path, artifacts["shadow_artifact"], manifest, torch
    )
    _kde_references(shadow, manifest["evaluation_ids"]["unlearn_ids"],
                    ruli_utils.gaussian_kde)
    tokenizer = AutoTokenizer.from_pretrained(BASE.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    target, target_metadata, _ = REFERENCE._validate_target_dataset(
        target_path, artifacts["target_dataset"], manifest, tokenizer, load_from_disk
    )
    if any(i < 0 or i >= len(target) for ids in condition_ids.values() for i in ids):
        raise ValueError("A condition target ID is outside the frozen dataset.")
    with COMMON._working_directory(ruli_root / "text"):
        train, valid, _ = ruli_utils.load_data(
            "WikiText103", SimpleNamespace(model_name=BASE.MODEL_NAME)
        )
    attack = train.shuffle(seed=BASE.FROZEN_ARTIFACT_SEED).select(range(BASE.ATTACK_SIZE))
    background = COMMON._validate_background_dataset(train, attack, tokenizer, manifest)
    unlearn, targets, retain, initial = BASE._build_condition_datasets(
        target, attack, manifest["evaluation_ids"]["unlearn_ids"], condition_ids,
        Subset, ConcatDataset,
    )

    if args.validate_only:
        with tempfile.TemporaryDirectory(prefix="experiment_2c_initial_") as directory:
            digest, _, _ = BASE._create_and_verify_shared_initial(
                Path(directory) / "initial_shared", args.seed,
                AutoModelForCausalLM, tokenizer, torch, np,
            )
        print(f"[VERIFY] Full 2C preflight passed; identical initial SHA-256={digest}.")
        print("[VERIFY] No training or persistent experiment outputs.")
        return

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing nonempty Experiment 2C output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_paths(output_root)
    scratch = output_root / "trainer_work"
    digest, config, preflight = BASE._create_and_verify_shared_initial(
        checkpoints["initial_shared"], args.seed, AutoModelForCausalLM,
        tokenizer, torch, np,
    )
    COMMON._configure_training_arguments_seed(ruli_utils, args.seed)
    records = {"initial_shared": {"path": str(checkpoints["initial_shared"]),
                                   "parameter_sha256": digest}}
    starts, cleanup = {}, {}
    for condition in CONDITIONS:
        COMMON._reset_rng(args.seed, torch, np)
        model = AutoModelForCausalLM.from_pretrained(checkpoints["initial_shared"])
        starts[condition] = COMMON._parameter_sha256(model, torch)
        if starts[condition] != digest:
            raise RuntimeError(f"{condition} initial parameter mismatch.")
        model = model.to(args.device)
        COMMON._reset_rng(args.seed, torch, np)
        records[condition], cleanup[condition] = _run_branch(
            model, tokenizer, initial[condition], retain[condition], unlearn, valid,
            condition, checkpoints, scratch, ruli_utils, args.device, torch,
        )
        records[condition]["starting_parameter_sha256"] = starts[condition]
        del model
        COMMON._cleanup_cuda(torch)
    BASE._assert_initial_parameter_identity(digest, starts, "Actual 2C branches")
    if scratch.exists():
        raise RuntimeError(f"Unexpected Trainer scratch remains: {scratch}")
    metadata = {
        "schema_version": 1, "experiment": "2C", "seed": args.seed,
        "manifest": manifest_metadata,
        "model_and_hyperparameters": dict(HYPERPARAMETERS),
        "model_config": config,
        "upstream_ruli_source_files": sources,
        "training_implementation": "unchanged Ruli/text/utils.py helpers as in 2B",
        "loss": loss_metadata,
        "ordered_target_dataset_ids": condition_ids,
        "ordered_unlearn_ids": manifest["evaluation_ids"]["unlearn_ids"],
        "background_dataset": background,
        "input_artifacts": {"shadow": shadow_metadata, "target_dataset": target_metadata},
        "dataset_sizes": {
            "condition_target": {c: len(targets[c]) for c in CONDITIONS},
            "initial_sft_and_prefix": {c: len(initial[c]) for c in CONDITIONS},
            "npo_retain": {c: len(retain[c]) for c in CONDITIONS},
            "npo_forget": len(unlearn),
        },
        "checkpoints": records,
        "starting_parameter_identity": {
            "passed": True, "sha256": digest,
            "preflight_reload_sha256": preflight,
            "actual_training_branch_sha256": starts,
        },
        "rng_policy": {"training_seed": args.seed, "data_seed": args.seed,
                       "frozen_wikitext_selection_seed": BASE.FROZEN_ARTIFACT_SEED,
                       "reset_before_each_branch": True},
        "trainer_work_cleanup": cleanup,
        "validation_results": {
            **validation, "frozen_artifacts": True, "last_7_loss_behavior": True,
            "shared_initial_parameter_identity": True,
            "checkpoint_capture_order": True, "no_final_retain_ft": True,
        },
        "git": {"ruli": COMMON._git_metadata(ruli_root),
                "ruli_experiments": COMMON._git_metadata(BASE.REPOSITORY_ROOT)},
        "software": COMMON._software_metadata(torch),
        "deviations_from_reference_ruli_behavior": [
            "Omit final 2-epoch retain SFT to measure the immediate NPO effect."
        ],
    }
    COMMON._atomic_json(metadata, output_root / "run_metadata.json")
    print(f"[INFO] Experiment 2C training complete: {output_root}")


if __name__ == "__main__":
    main()
