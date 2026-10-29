"""Model-specific provenance checks around the unchanged Experiment 2B scorer."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (COMMON, CONDITIONS, DEFAULT_MODEL, HERE, MANIFEST, RULI, REFERENCE,
                    identity, load_bundle, load_model, load_shadow, manifest_at,
                    model_spec, read_json, source_metadata, model_paths)
from training import validate_stage_records


def validate_run(run, manifest, meta, shadow_info, seed, sources):
    if (run.get("experiment") != "model_generalization_2b" or run.get("complete") is not True
            or run.get("seed") != seed or run.get("data_identity") != identity(meta)
            or run.get("model") != meta["model"] or run.get("shadow_artifact") != shadow_info
            or run.get("comparison") != meta["comparison"]
            or run.get("prediction_policy") != meta["prediction_policy"]
            or run.get("assignment_sha256") != meta["comparison"]["assignment_payload_sha256"]
            or run.get("source_hashes") != sources or shadow_info.get("source_hashes") != sources
            or run.get("sets") != manifest["sets"]):
        raise ValueError("Run provenance/model/data/shadow/seed mismatch or incomplete run.")
    expected = {c: manifest["conditions"][c]["ordered_target_dataset_ids"] for c in CONDITIONS}
    if run.get("conditions") != expected:
        raise ValueError("Run conditions differ from frozen HIGH/LOW/PLACEBO membership.")
    for condition in CONDITIONS:
        validate_stage_records(run["stages"][condition], seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--ruli-root", type=Path, default=RULI)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--shadow-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    parser.add_argument("--experiment-output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    paths = model_paths(args.model_name)
    args.data = args.data or paths["data"]
    import torch
    manifest, manifest_info, _, _ = manifest_at(args.manifest)
    datasets, tokenizer, meta = load_bundle(args.data, manifest, args.model_name)
    shadow, shadow_info = load_shadow(args.shadow_path, meta, manifest)
    root = (args.experiment_output or paths["results"] / f"seed_{args.seed}").resolve()
    run = read_json(root / "run_metadata.json")
    validate_run(run, manifest, meta, shadow_info, args.seed, source_metadata(args.ruli_root))
    utils = COMMON._load_ruli_utils(args.ruli_root.resolve() / "text")
    loss_check = REFERENCE._validate_reference_loss_path(utils, torch)
    # Validate actual saved parameters, not only JSON declarations.
    checkpoints = {"initial_shared": (root / "initial_shared", run["initial_parameter_sha256"])}
    for c in CONDITIONS:
        for label, stage in (("pre_npo", "prefix"), ("final", "final_retain_ft")):
            checkpoints[f"{c}_{label}"] = (root / c / label, run["stages"][c][stage]["parameter_sha256"])
    for key, (path, expected) in checkpoints.items():
        model = load_model(args.model_name, checkpoint=path)
        if COMMON._parameter_sha256(model, torch) != expected:
            raise ValueError(f"Checkpoint parameters changed: {key}")
        del model
    if args.validate_only:
        print("[VERIFY] Data, model-specific shadows, training metadata, and seven saved checkpoints passed.")
        return
    output = root / "evaluation"
    if any((output / name).exists() for name in REFERENCE.OUTPUT_FILENAMES):
        raise FileExistsError("Evaluation outputs already exist; refusing to overwrite.")
    target = datasets["target"]
    unlearn_ids = manifest["evaluation_ids"]["unlearn_ids"]
    out_ids = manifest["evaluation_ids"]["out_ids"]
    ids = unlearn_ids + out_ids
    tokens = {i: target[i]["input_ids"] for i in ids}
    supported = manifest["sets"]["S_sample_ids"]
    negative = manifest["sets"]["negative_control_sample_ids"]
    kdes = REFERENCE._build_kde_references(shadow, ids, utils.gaussian_kde)
    rows, metrics = [], {}
    for condition in CONDITIONS:
        REFERENCE._reset_determinism(torch, args.seed)
        model = load_model(args.model_name, args.device, root / condition / "pre_npo")
        original_losses = REFERENCE._run_reference_inference(model, target, tokenizer, args.device, utils, out_ids, tokens)
        del model
        COMMON._cleanup_cuda(torch)
        model = load_model(args.model_name, args.device, root / condition / "final")
        condition_rows, metrics[condition] = REFERENCE._evaluate_condition(
            condition, model, target, tokenizer, args.device, utils, shadow, kdes,
            unlearn_ids, out_ids, tokens, set(supported), set(negative),
            dict(zip(out_ids, original_losses, strict=True)))
        rows.extend(condition_rows)
        del model
        COMMON._cleanup_cuda(torch)
    counts = read_json(args.data / "target_prediction_counts.json")
    for row in rows:
        row.update(counts[row["sample_id"]])
    REFERENCE._validate_condition_row_alignment(rows, unlearn_ids, out_ids)
    contrast = REFERENCE._contrast_rows(rows, supported)
    summary = {
        "schema_version": 1, "experiment": "model_generalization_2b", "seed": args.seed,
        "model": model_spec(args.model_name), "data_identity": identity(meta),
        "comparison": meta["comparison"], "prediction_policy": meta["prediction_policy"],
        "descriptive_metric_prediction_counts": {
            "six_predictions": sum(counts[i]["prediction_token_count"] == 6 for i in ids),
            "seven_predictions": sum(counts[i]["prediction_token_count"] == 7 for i in ids),
            "note": "Full 200 UNLEARN and 200 OUT denominators retained; Pythia includes 16 six-prediction examples."},
        "manifest": manifest_info, "shadow_artifact": shadow_info,
        "training_run_metadata_sha256": COMMON._sha256_file(root / "run_metadata.json"),
        "primary_contrast": "privacy_log_odds_LOW - privacy_log_odds_PLACEBO",
        "hypothesis": "LOW < PLACEBO; negative values support the hypothesis",
        "single_seed_diagnostic_only": True, "new_statistical_tests": False,
        "cohorts": {"supported_S": REFERENCE._cohort_summary(rows, supported),
                    "negative_controls": REFERENCE._cohort_summary(rows, negative),
                    "all_UNLEARN": REFERENCE._cohort_summary(rows, unlearn_ids)},
        "aggregate_metrics": metrics, "loss_validation": loss_check,
        "scoring": "Unchanged Experiment 2B fixed-shadow KDE log odds and bounded p1/(p1+p0+1e-12); pre-NPO OUT reference for efficacy.",
        "sample_counts": {"supported_S": len(supported), "negative_controls": len(negative),
                          "per_sample_rows": len(rows), "primary_contrast_rows": len(contrast)},
        "package_versions": COMMON._software_metadata(torch),
    }
    REFERENCE._write_outputs(output, rows, contrast, summary)
    print(f"[COMPLETE] Evaluation: {output}")


if __name__ == "__main__":
    main()
