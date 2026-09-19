#!/usr/bin/env python3
"""Paired immediate pre/post-NPO losses; raw loss DiD is the primary outcome."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_2c_runner", SCRIPT_DIR / "run_experiment_2c.py"
)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)
REFERENCE = RUNNER.REFERENCE
COMMON = RUNNER.COMMON
CONDITIONS = RUNNER.CONDITIONS
OUTPUT_FILENAMES = (
    "per_sample_pre_post.csv", "primary_contrast.csv", "evaluation_summary.json"
)
KDE_NOTE = (
    "Pre-NPO uses unlearn_original versus out_original; post-NPO uses "
    "unlearn_unlearned versus out_unlearned. These different reference "
    "distributions make direct pre/post KDE-score subtraction a secondary "
    "diagnostic, not the primary difference-in-differences estimand. The fixed "
    "upstream unlearned shadow distributions include final retain FT, whereas "
    "the 2C target post_npo checkpoints intentionally precede that stage."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42,
                        choices=RUNNER.BASE.PREREGISTERED_TRAINING_SEEDS)
    parser.add_argument("--ruli-root", type=Path, default=RUNNER.BASE.DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=RUNNER.BASE.DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument("--experiment-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true", help=(
        "Validate frozen inputs, metadata, loss behavior, KDEs, and checkpoint "
        "structure without model inference; parameter hashes are checked at inference."
    ))
    return parser.parse_args()


def _mapping(value, label):
    return COMMON._mapping(value, label)


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, found {actual!r}.")


def _validate_training_run_metadata(root, seed, manifest):
    path = root / "run_metadata.json"
    metadata = _mapping(json.loads(path.read_text(encoding="utf-8")), "run metadata")
    _require_equal(metadata.get("experiment"), "2C", "experiment")
    _require_equal(metadata.get("seed"), seed, "seed")
    run_manifest = _mapping(metadata.get("manifest"), "run manifest")
    _require_equal(run_manifest.get("declared_canonical_content_sha256"),
                   manifest["manifest_hash"]["sha256"], "manifest hash")
    hyper = _mapping(metadata.get("model_and_hyperparameters"), "hyperparameters")
    for key, value in RUNNER.HYPERPARAMETERS.items():
        _require_equal(hyper.get(key), value, key)
    background = _mapping(metadata.get("background_dataset"), "background")
    for key, value in {
        "selection_seed": 42, "count": 15_000,
        "membership_sha256": manifest["shared_wikitext_background"]["membership_sha256"],
    }.items():
        _require_equal(background.get(key), value, f"background {key}")
    _require_equal(metadata.get("ordered_unlearn_ids"),
                   manifest["evaluation_ids"]["unlearn_ids"], "UNLEARN order")
    ids = _mapping(metadata.get("ordered_target_dataset_ids"), "condition IDs")
    sizes = _mapping(metadata.get("dataset_sizes"), "dataset sizes")
    _require_equal(sizes.get("npo_forget"), 200, "NPO forget size")
    for condition in CONDITIONS:
        _require_equal(ids.get(condition), manifest["conditions"][condition][
            "ordered_target_dataset_ids"], f"{condition} target IDs")
        for field, count in (("condition_target", 200),
                             ("initial_sft_and_prefix", 15_400),
                             ("npo_retain", 15_200)):
            values = _mapping(sizes.get(field), field)
            _require_equal(values.get(condition), count, f"{condition} {field}")
    identity = _mapping(metadata.get("starting_parameter_identity"), "initial identity")
    _require_equal(identity.get("passed"), True, "initial identity passed")
    digest = identity.get("sha256")
    _validate_hash(digest)
    for field in ("preflight_reload_sha256", "actual_training_branch_sha256"):
        proof = _mapping(identity.get(field), field)
        RUNNER.BASE._assert_initial_parameter_identity(digest, proof, field)
    records = _mapping(metadata.get("checkpoints"), "checkpoints")
    shared = _mapping(records.get("initial_shared"), "initial_shared")
    _require_equal(shared.get("parameter_sha256"), digest, "initial checkpoint hash")
    for condition in CONDITIONS:
        record = _mapping(records.get(condition), condition)
        _require_equal(record.get("starting_parameter_sha256"), digest,
                       f"{condition} actual start")
        _require_equal(record.get("stage_order"), list(RUNNER.STAGES),
                       f"{condition} stage order")
        if "final" in record:
            raise ValueError("A final retain-FT checkpoint is not a 2C outcome.")
        for stage, capture in RUNNER.CAPTURE.items():
            checkpoint = _mapping(record.get(stage), f"{condition} {stage}")
            _require_equal(checkpoint.get("capture"), capture, f"{condition} {stage}")
            _require_equal(checkpoint.get("final_retain_ft_updates"), 0, "retain FT")
            _validate_hash(checkpoint.get("parameter_sha256"))
    validations = _mapping(metadata.get("validation_results"), "validations")
    for key in ("frozen_artifacts", "last_7_loss_behavior",
                "shared_initial_parameter_identity", "checkpoint_capture_order",
                "no_final_retain_ft"):
        _require_equal(validations.get(key), True, key)
    if any(value is not True for value in validations.values()):
        raise ValueError("Training validation contains a failed check.")
    return metadata


def _validate_hash(digest):
    if not isinstance(digest, str) or len(digest) != 64 or any(
        c not in "0123456789abcdef" for c in digest
    ):
        raise ValueError("Missing or invalid checkpoint parameter SHA-256.")


def _verify_model_hash(model, expected, label, torch):
    _require_equal(COMMON._parameter_sha256(model, torch), expected,
                   f"{label} loaded parameter hash")


def _paired_rows(measurements, ids, supported, negative, seed):
    """Join on explicit IDs, rejecting any missing, extra, or reordered pass."""
    if set(measurements) != set(CONDITIONS):
        raise ValueError("Measurements must include exactly HIGH, LOW, PLACEBO.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation IDs.")
    for condition in CONDITIONS:
        if set(measurements[condition]) != set(RUNNER.CAPTURE):
            raise ValueError(f"Missing pre/post measurement for {condition}.")
        for stage in RUNNER.CAPTURE:
            samples = measurements[condition][stage]
            if list(samples) != list(ids):
                raise ValueError(f"{condition} {stage} sample identity/order mismatch.")
            for sample_id, values in samples.items():
                if not all(math.isfinite(values[field]) for field in (
                    "loss", "privacy_log_odds", "privacy_score"
                )):
                    raise FloatingPointError(f"Nonfinite measurement for {sample_id}.")
    long_rows, wide_rows = [], []
    for sample_id in ids:
        wide = {"sample_id": sample_id, "seed": seed}
        for condition in CONDITIONS:
            pre = measurements[condition]["pre_npo"][sample_id]
            post = measurements[condition]["post_npo"][sample_id]
            delta = post["loss"] - pre["loss"]
            if not math.isfinite(delta):
                raise FloatingPointError("Nonfinite delta loss.")
            long_rows.append({
                "sample_id": sample_id, "seed": seed, "condition": condition,
                "is_supported_S": sample_id in supported,
                "is_negative_control": sample_id in negative,
                "pre_npo_loss": pre["loss"], "post_npo_loss": post["loss"],
                "delta_loss": delta,
                "pre_privacy_log_odds": pre["privacy_log_odds"],
                "post_privacy_log_odds": post["privacy_log_odds"],
                "pre_privacy_score": pre["privacy_score"],
                "post_privacy_score": post["privacy_score"],
            })
            wide.update({f"{condition}_pre_loss": pre["loss"],
                         f"{condition}_post_loss": post["loss"],
                         f"{condition}_delta_loss": delta})
        for condition in ("LOW", "HIGH"):
            contrast = wide[f"{condition}_delta_loss"] - wide["PLACEBO_delta_loss"]
            if not math.isfinite(contrast):
                raise FloatingPointError("Nonfinite difference-in-differences.")
            wide[f"{condition}_minus_PLACEBO_delta"] = contrast
        wide_rows.append(wide)
    return long_rows, wide_rows


def _cohort_summary(wide_rows, ids):
    by_id = {row["sample_id"]: row for row in wide_rows}
    rows = [by_id[sample_id] for sample_id in ids]

    def describe(field):
        values = [row[field] for row in rows]
        return {"mean": statistics.mean(values), "median": statistics.median(values)}

    positives = sum(row["LOW_minus_PLACEBO_delta"] > 0 for row in rows)
    return {
        "n": len(rows),
        "conditions": {c: {phase: describe(f"{c}_{phase}")
                           for phase in ("pre_loss", "post_loss", "delta_loss")}
                       for c in CONDITIONS},
        "LOW_minus_PLACEBO_delta": {
            **describe("LOW_minus_PLACEBO_delta"), "number_positive": positives,
            "fraction_positive": positives / len(rows),
        },
        "HIGH_minus_PLACEBO_delta": describe("HIGH_minus_PLACEBO_delta"),
    }


def _write_outputs(output_dir, rows, primary, summary):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="write_2c_", dir=output_dir) as directory:
        temp = Path(directory)
        REFERENCE._write_csv(temp / OUTPUT_FILENAMES[0], rows)
        REFERENCE._write_csv(temp / OUTPUT_FILENAMES[1], primary)
        (temp / OUTPUT_FILENAMES[2]).write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name in OUTPUT_FILENAMES:
            (temp / name).replace(output_dir / name)


def main():
    args = parse_args()
    manifest, manifest_metadata = REFERENCE._validate_manifest(args.manifest)
    RUNNER.BASE._validate_experiment_2b_protocol(manifest)
    root = RUNNER._output_root(args.experiment_output, args.seed)
    training = _validate_training_run_metadata(root, args.seed, manifest)
    checkpoints = RUNNER._checkpoint_paths(root)
    checkpoint_metadata = {
        key: REFERENCE._checkpoint_metadata(path, key)
        for key, path in checkpoints.items()
    }
    ruli_root = args.ruli_root.resolve()
    sources = RUNNER._source_metadata(ruli_root)
    saved_sources = _mapping(training.get("upstream_ruli_source_files"), "RULI sources")
    for name, record in sources.items():
        saved = _mapping(saved_sources.get(name), name)
        _require_equal(record["sha256"], saved.get("sha256"), f"{name} source hash")
    shadow_path, target_path = RUNNER.BASE._artifact_paths(args, ruli_root)
    artifacts = manifest["input_artifacts"]
    COMMON._verify_file_artifact(
        shadow_path, artifacts["shadow_artifact"], "Frozen 9-shadow artifact"
    )
    COMMON._verify_target_dataset_storage(target_path, artifacts["target_dataset"])

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shadow, shadow_metadata = RUNNER._load_shadow(
        shadow_path, artifacts["shadow_artifact"], manifest, torch
    )
    ruli_utils = COMMON._load_ruli_utils(ruli_root / "text")
    loss_metadata = REFERENCE._validate_reference_loss_path(ruli_utils, torch)
    tokenizer = AutoTokenizer.from_pretrained(checkpoints["initial_shared"])
    dataset, target_metadata, raw_tokens = REFERENCE._validate_target_dataset(
        target_path, artifacts["target_dataset"], manifest, tokenizer, load_from_disk
    )
    ids = tuple(manifest["evaluation_ids"]["unlearn_ids"])
    tokens = {sample_id: tuple(raw_tokens[sample_id]) for sample_id in ids}
    token_digest = COMMON._canonical_sha256({str(i): list(tokens[i]) for i in ids})
    kdes = RUNNER._kde_references(shadow, ids, ruli_utils.gaussian_kde)
    if args.validate_only:
        print("[VERIFY] 2C metadata, frozen inputs, loss behavior, KDEs, and all "
              "seven checkpoint structures passed. No inference or output writes; "
              "loaded parameter hashes will be verified during evaluation.")
        return
    output_dir = root / "evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation output: {output_dir}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable.")
    device = torch.device(args.device)
    initial = AutoModelForCausalLM.from_pretrained(checkpoints["initial_shared"])
    _verify_model_hash(initial, training["checkpoints"]["initial_shared"][
        "parameter_sha256"], "initial_shared", torch)
    del initial
    measurements = {}
    for condition in CONDITIONS:
        measurements[condition] = {}
        for stage in RUNNER.CAPTURE:
            label = f"{condition}_{stage}"
            print(f"[INFO] Evaluating {label} on the same 200 UNLEARN samples.")
            REFERENCE._reset_determinism(torch, args.seed)
            model = AutoModelForCausalLM.from_pretrained(checkpoints[label])
            _verify_model_hash(model, training["checkpoints"][condition][stage][
                "parameter_sha256"], label, torch)
            model = model.to(device)
            losses = REFERENCE._run_reference_inference(
                model, dataset, tokenizer, device, ruli_utils, ids, tokens
            )
            values = {}
            for sample_id, loss in zip(ids, losses, strict=True):
                log_odds, score = REFERENCE._score_kde(
                    loss, kdes[stage][sample_id], stage, condition, sample_id
                )
                values[sample_id] = {"loss": loss, "privacy_log_odds": log_odds,
                                     "privacy_score": score}
            measurements[condition][stage] = values
            del model
            COMMON._cleanup_cuda(torch)
    supported = manifest["sets"]["S_sample_ids"]
    negative = manifest["sets"]["negative_control_sample_ids"]
    rows, wide = _paired_rows(measurements, ids, set(supported), set(negative), args.seed)
    indexed = {row["sample_id"]: row for row in wide}
    primary = [indexed[sample_id] for sample_id in supported]
    summary = {
        "schema_version": 1, "experiment": "2C", "seed": args.seed,
        "single_seed_diagnostic_only": True,
        "primary_estimand": {
            "cohort": "28 supported S samples",
            "delta_loss": "post_npo_loss - pre_npo_loss",
            "contrast": "(LOW_post - LOW_pre) - (PLACEBO_post - PLACEBO_pre)",
            "direction": "LOW_minus_PLACEBO_delta > 0",
            "interpretation": "positive means stronger immediate NPO loss increase in LOW",
        },
        "manifest": manifest_metadata, "shadow_artifact": shadow_metadata,
        "target_dataset": target_metadata, "checkpoints": checkpoint_metadata,
        "training_run_metadata": {"path": str(root / "run_metadata.json"),
                                  "sha256": COMMON._sha256_file(root / "run_metadata.json")},
        "sample_counts": {"UNLEARN": len(ids), "supported_S": len(supported),
                          "negative_controls": len(negative), "per_sample_rows": len(rows),
                          "primary_contrast_rows": len(primary)},
        "scoring": {
            "loss": loss_metadata, "privacy_note": KDE_NOTE,
            "privacy_log_odds": "log KDE_positive(loss) - log KDE_negative(loss)",
            "privacy_score": "p_positive / (p_positive + p_negative + 1e-12)",
            "kde_bandwidth": "scipy.stats.gaussian_kde default (Scott's rule)",
            "kde_device": "CPU",
        },
        "primary_descriptive_results": {
            "supported_S": _cohort_summary(wide, supported),
            "negative_controls": _cohort_summary(wide, negative),
            "all_UNLEARN": _cohort_summary(wide, ids),
        },
        "alignment": {"ordered_unlearn_ids": list(ids),
                      "token_sequences_sha256": token_digest,
                      "same_immutable_token_sequences_all_six_passes": True},
        "validation_status": {
            "passed": True, "all_seven_loaded_parameter_hashes": "passed",
            "pre_post_capture_stages": "passed", "no_final_retain_ft": "passed",
            "condition_sample_order_and_identity": "passed",
            "nonfinite_losses_or_kde_scores": 0,
        },
        "package_versions": REFERENCE._package_versions(),
        "device": str(device),
        "deviations_from_reference_ruli_behavior": [
            "Final retain SFT omitted; measure immediate pre/post-NPO checkpoints.",
            "Original-state shadow distributions used for secondary pre-NPO privacy.",
        ],
    }
    _write_outputs(output_dir, rows, primary, summary)
    print(f"[INFO] Wrote 600 paired rows and 28 primary contrasts to {output_dir}")


if __name__ == "__main__":
    main()
