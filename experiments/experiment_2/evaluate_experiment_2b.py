#!/usr/bin/env python3
"""Evaluate one frozen Experiment 2B seed with fixed-shadow RULI scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
DEFAULT_MANIFEST = SCRIPT_DIR / "results" / "intervention_manifest.json"
DEFAULT_TRAINING_SEED = 42
CONDITIONS = ("HIGH", "LOW", "PLACEBO")


def _load_module(filename: str, name: str) -> Any:
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load required Experiment 2 module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = _load_module("run_experiment_2b.py", "_experiment_2b_runner")
REFERENCE = _load_module("evaluate_experiment_2a.py", "_experiment_2a_evaluator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Experiment 2B seed's HIGH, LOW, and PLACEBO final "
            "checkpoints with the validated fixed-shadow RULI scoring path."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--ruli-root", type=Path, default=DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument(
        "--experiment-output",
        type=Path,
        help="Defaults to results/experiment_2b/seed_<SEED>.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate frozen artifacts, training metadata, target rows, upstream "
            "loss behavior, and checkpoint structure without loading model weights."
        ),
    )
    return parser.parse_args()


def _checkpoint_paths(experiment_output: Path) -> dict[str, Path]:
    root = experiment_output.resolve()
    paths = {"initial_shared": root / "initial_shared"}
    for condition in CONDITIONS:
        paths[f"{condition}_pre_npo"] = root / f"{condition}_pre_npo"
        paths[f"{condition}_final"] = root / f"{condition}_final"
    return paths


def _validate_training_run_metadata(
    experiment_output: Path,
    seed: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    path = experiment_output.resolve() / "run_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Training run metadata does not exist: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training run metadata is invalid JSON: {path}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("Training run metadata root must be a JSON object.")
    if metadata.get("experiment") != "2B" or metadata.get("seed") != seed:
        raise ValueError(
            "Training run metadata does not match Experiment 2B seed "
            f"{seed}: experiment={metadata.get('experiment')!r}, "
            f"seed={metadata.get('seed')!r}."
        )

    run_manifest = metadata.get("manifest")
    if not isinstance(run_manifest, Mapping) or run_manifest.get(
        "declared_canonical_content_sha256"
    ) != manifest["manifest_hash"]["sha256"]:
        raise ValueError("Training run used a different intervention manifest.")

    hyperparameters = metadata.get("model_and_hyperparameters")
    expected_hyperparameters = {
        "model": "gpt2",
        "initial_sft_epochs": 5,
        "prefix_epochs": 1,
        "unlearn_method": "npo",
        "npo_epochs": 15,
        "final_retain_sft_epochs": 2,
        "attack_size": 15_000,
    }
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("Training run has no model_and_hyperparameters metadata.")
    mismatches = {
        key: hyperparameters.get(key)
        for key, expected in expected_hyperparameters.items()
        if hyperparameters.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "Training run changed frozen model or hyperparameters: "
            + ", ".join(f"{key}={value!r}" for key, value in mismatches.items())
        )

    background = metadata.get("background_dataset")
    frozen_background = manifest["shared_wikitext_background"]
    if (
        not isinstance(background, Mapping)
        or background.get("selection_seed") != RUNNER.FROZEN_ARTIFACT_SEED
        or background.get("count") != 15_000
        or background.get("membership_sha256")
        != frozen_background["membership_sha256"]
    ):
        raise ValueError("Training run did not use the frozen WikiText background.")

    condition_ids = metadata.get("ordered_target_dataset_ids")
    if not isinstance(condition_ids, Mapping):
        raise ValueError("Training run has no condition target-ID metadata.")
    for condition in CONDITIONS:
        if condition_ids.get(condition) != manifest["conditions"][condition][
            "ordered_target_dataset_ids"
        ]:
            raise ValueError(f"Training run changed the frozen {condition} target IDs.")

    sizes = metadata.get("dataset_sizes")
    if not isinstance(sizes, Mapping):
        raise ValueError("Training run has no dataset-size metadata.")
    for field, expected in (
        ("initial_sft_and_prefix", 15_400),
        ("npo_retain", 15_200),
        ("final_retain", 15_200),
    ):
        values = sizes.get(field)
        if not isinstance(values, Mapping) or any(
            values.get(condition) != expected for condition in CONDITIONS
        ):
            raise ValueError(
                f"Training run did not inject condition retain data into {field}."
            )
    if sizes.get("npo_forget") != 200:
        raise ValueError("Training run changed the shared NPO UNLEARN size.")

    identity = metadata.get("starting_parameter_identity")
    records = metadata.get("checkpoints")
    if not isinstance(identity, Mapping) or identity.get("passed") is not True:
        raise ValueError("Training run did not pass shared-initial parameter identity.")
    if not isinstance(records, Mapping):
        raise ValueError("Training run has no checkpoint metadata.")
    shared_hash = identity.get("sha256")
    initial_record = records.get("initial_shared")
    if (
        not isinstance(initial_record, Mapping)
        or not shared_hash
        or initial_record.get("parameter_sha256") != shared_hash
    ):
        raise ValueError("Shared initial checkpoint hash metadata is inconsistent.")
    for proof_field in (
        "preflight_reload_sha256",
        "actual_training_branch_sha256",
    ):
        proof = identity.get(proof_field)
        if not isinstance(proof, Mapping) or any(
            proof.get(condition) != shared_hash for condition in CONDITIONS
        ):
            raise ValueError(
                f"Training run lacks byte-identical initial proof: {proof_field}."
            )
    for condition in CONDITIONS:
        record = records.get(condition)
        if not isinstance(record, Mapping) or record.get(
            "starting_parameter_sha256"
        ) != shared_hash:
            raise ValueError(
                f"{condition} did not start from the byte-identical initial state."
            )
        if not isinstance(record.get("pre_npo"), Mapping) or not isinstance(
            record.get("final"), Mapping
        ):
            raise ValueError(f"Training metadata lacks {condition} checkpoint records.")

    validations = metadata.get("validation_results")
    if not isinstance(validations, Mapping) or any(
        value is not True for value in validations.values()
    ):
        raise ValueError(
            "Training run did not record all validation results as passed."
        )

    return {
        "path": str(path),
        "sha256": RUNNER.COMMON._sha256_file(path),
        "seed": seed,
        "frozen_hyperparameters": "passed",
        "frozen_background": "passed",
        "condition_dataset_injection": "passed",
        "starting_parameter_identity": "passed",
        "starting_parameter_sha256": shared_hash,
        "post_npo_parameter_equality_required": False,
    }


def main() -> None:
    args = parse_args()
    RUNNER._validate_training_seed(args.seed)
    manifest, manifest_metadata = REFERENCE._validate_manifest(args.manifest)
    RUNNER._validate_experiment_2b_protocol(manifest)
    experiment_output = RUNNER._validate_seed_output_path(
        args.experiment_output
        if args.experiment_output is not None
        else RUNNER._seed_output_root(args.seed),
        args.seed,
        "--experiment-output",
    )
    ruli_root = args.ruli_root.resolve()
    ruli_text_dir = ruli_root / "text"
    shadow_path, target_data_path = RUNNER._artifact_paths(args, ruli_root)
    checkpoints = _checkpoint_paths(experiment_output)
    checkpoint_metadata = {
        name: REFERENCE._checkpoint_metadata(path, name)
        for name, path in checkpoints.items()
    }
    training_run_metadata = _validate_training_run_metadata(
        experiment_output, args.seed, manifest
    )

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if (
        not args.validate_only
        and str(args.device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"Requested {args.device}, but PyTorch reports no CUDA device."
        )

    input_artifacts = manifest.get("input_artifacts")
    if not isinstance(input_artifacts, Mapping):
        raise ValueError("manifest.input_artifacts must be a JSON object.")
    shadow, shadow_metadata = REFERENCE._load_and_validate_shadow(
        shadow_path,
        input_artifacts["shadow_artifact"],
        manifest,
        torch,
    )
    ruli_utils = RUNNER.COMMON._load_ruli_utils(ruli_text_dir)
    loss_metadata = REFERENCE._validate_reference_loss_path(ruli_utils, torch)
    tokenizer = AutoTokenizer.from_pretrained(checkpoints["initial_shared"])
    target_dataset, target_metadata, tokens = REFERENCE._validate_target_dataset(
        target_data_path,
        input_artifacts["target_dataset"],
        manifest,
        tokenizer,
        load_from_disk,
    )
    print(
        "[VERIFY] Frozen manifest, shadow artifact, target rows, upstream loss "
        f"path, training metadata, and all seven seed-{args.seed} checkpoints passed."
    )
    if args.validate_only:
        print(
            "[VERIFY] Validation-only mode loaded no model weights and wrote no "
            "files."
        )
        return

    output_dir = experiment_output / "evaluation"
    occupied = [
        output_dir / name
        for name in REFERENCE.OUTPUT_FILENAMES
        if (output_dir / name).exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs: "
            + ", ".join(str(path) for path in occupied)
        )

    REFERENCE._reset_determinism(torch, args.seed)
    device = torch.device(args.device)
    unlearn_ids = manifest["evaluation_ids"]["unlearn_ids"]
    out_ids = manifest["evaluation_ids"]["out_ids"]
    evaluation_ids = unlearn_ids + out_ids
    supported_order = manifest["sets"]["S_sample_ids"]
    negative_order = manifest["sets"]["negative_control_sample_ids"]
    supported_ids = set(supported_order)
    negative_ids = set(negative_order)
    kdes = REFERENCE._build_kde_references(
        shadow, evaluation_ids, ruli_utils.gaussian_kde
    )

    original_out_losses: dict[str, dict[int, float]] = {}
    for condition in CONDITIONS:
        print(f"[INFO] Computing reference OUT losses for {condition} pre-NPO model.")
        REFERENCE._reset_determinism(torch, args.seed)
        original_model = AutoModelForCausalLM.from_pretrained(
            checkpoints[f"{condition}_pre_npo"]
        ).to(device)
        losses = REFERENCE._run_reference_inference(
            original_model,
            target_dataset,
            tokenizer,
            device,
            ruli_utils,
            out_ids,
            tokens,
        )
        original_out_losses[condition] = dict(zip(out_ids, losses, strict=True))
        del original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_sample_rows: list[dict[str, Any]] = []
    aggregate_metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        print(f"[INFO] Evaluating Experiment 2B {condition} final model.")
        REFERENCE._reset_determinism(torch, args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoints[f"{condition}_final"]
        ).to(device)
        condition_rows, condition_metrics = REFERENCE._evaluate_condition(
            condition,
            model,
            target_dataset,
            tokenizer,
            device,
            ruli_utils,
            shadow,
            kdes,
            unlearn_ids,
            out_ids,
            tokens,
            supported_ids,
            negative_ids,
            original_out_losses[condition],
        )
        per_sample_rows.extend(condition_rows)
        aggregate_metrics[condition] = condition_metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    REFERENCE._validate_condition_row_alignment(per_sample_rows, unlearn_ids, out_ids)
    contrast = REFERENCE._contrast_rows(per_sample_rows, supported_order)
    primary_results = {
        "supported_S": REFERENCE._cohort_summary(per_sample_rows, supported_order),
        "all_UNLEARN": REFERENCE._cohort_summary(per_sample_rows, unlearn_ids),
        "negative_controls": REFERENCE._cohort_summary(
            per_sample_rows, negative_order
        ),
    }
    summary = {
        "schema_version": 1,
        "experiment": "2B",
        "seed": args.seed,
        "single_seed_diagnostic_only": True,
        "preregistered_hypothesis": {
            "contrast": "privacy_log_odds_LOW - privacy_log_odds_PLACEBO",
            "direction": "LOW < PLACEBO",
            "interpretation": "negative LOW_minus_PLACEBO supports the hypothesis",
        },
        "manifest_sha256": manifest_metadata["frozen_content_sha256"],
        "manifest": manifest_metadata,
        "shadow_artifact": shadow_metadata,
        "target_dataset": target_metadata,
        "checkpoints": checkpoint_metadata,
        "training_run_metadata": training_run_metadata,
        "original_checkpoint_policy": (
            "Each condition's saved post-prefix, pre-NPO checkpoint supplies its "
            "reference OUT losses; no post-NPO checkpoint is substituted."
        ),
        "sample_counts": {
            "IN_partition": REFERENCE.EXPECTED_SPLIT_COUNT,
            "UNLEARN_partition": len(unlearn_ids),
            "OUT_partition": len(out_ids),
            "supported_S": len(supported_order),
            "negative_controls": len(negative_order),
            "conditions": len(CONDITIONS),
            "per_sample_rows": len(per_sample_rows),
            "primary_contrast_rows": len(contrast),
        },
        "scoring": {
            "loss": loss_metadata,
            "privacy_log_odds": (
                "scipy.stats.gaussian_kde(unlearn_unlearned).logpdf(loss) - "
                "scipy.stats.gaussian_kde(out_unlearned).logpdf(loss)"
            ),
            "privacy_score": (
                "p_unlearn_unlearned / (p_unlearn_unlearned + "
                "p_out_unlearned + 1e-12); exact bounded reference formula"
            ),
            "efficacy_log_odds": (
                "scipy.stats.gaussian_kde(unlearn_unlearned).logpdf(loss) - "
                "scipy.stats.gaussian_kde(out_original).logpdf(loss)"
            ),
            "efficacy_score": (
                "p_unlearn_unlearned / (p_unlearn_unlearned + "
                "p_out_original + 1e-12); exact bounded reference formula"
            ),
            "kde_bandwidth": "scipy.stats.gaussian_kde default (Scott's rule)",
            "kde_device": "CPU",
            "efficacy_scope": (
                "UNLEARN and OUT; OUT loss uses each condition's own pre-NPO model"
            ),
        },
        "primary_descriptive_results": primary_results,
        "aggregate_reference_metrics": aggregate_metrics,
        "package_versions": REFERENCE._package_versions(),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "validation_status": {
            "passed": True,
            "manifest_hash": "passed",
            "shadow_alignment": "passed",
            "target_partition_alignment": "passed",
            "target_storage_and_fingerprint": "passed",
            "supported_sample_token_and_text_hashes": "passed",
            "checkpoint_structure": "passed",
            "shared_initial_parameter_identity": "passed",
            "condition_dataset_injection": "passed",
            "condition_sample_order_and_identity": "passed",
            "upstream_loss_behavior": "passed",
            "per_sample_bounded_scores_reproduce_upstream_metrics": "passed",
            "nonfinite_losses_or_kde_scores": 0,
        },
        "deviations_from_reference_behavior": [],
    }
    REFERENCE._write_outputs(output_dir, per_sample_rows, contrast, summary)
    print(f"[VERIFY] Wrote {len(per_sample_rows)} aligned per-sample rows.")
    print(f"[INFO] Experiment 2B evaluation outputs: {output_dir}")


if __name__ == "__main__":
    main()
