"""Separate 2B-style model generalization runner; training requires explicit --train."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (BASE, COMMON, CONDITIONS, DEFAULT_MODEL, HERE, MANIFEST, RULI,
                    identity, load_bundle, load_model, load_shadow, manifest_at,
                    model_spec, read_json, require_preflight, source_metadata, write_json,
                    model_paths, ASSIGNMENT_PATH, validate_assignments, digest)
from training import smoke, run_stages, validate_stage_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--ruli-root", type=Path, default=RULI)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--seed", type=int, choices=range(42, 47), default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validation-report", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--prepare-seed-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    paths = model_paths(args.model_name)
    args.data = args.data or paths["data"]
    args.validation_report = args.validation_report or paths["validation"]
    if args.smoke_test:
        result = smoke(args.model_name, args.ruli_root.resolve(), args.device, args.report)
        print(f"[VERIFY] {args.model_name} architecture smoke passed; research_result={result['research_result']}")
        return
    manifest, manifest_info, condition_ids, checks = manifest_at(args.manifest)
    datasets, tokenizer, data_info = load_bundle(args.data, manifest, args.model_name)
    assignments = read_json(ASSIGNMENT_PATH)
    validate_assignments(assignments, manifest)
    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, Subset
    forget, targets, retains, initials = BASE._build_condition_datasets(
        datasets["target"], datasets["background"], manifest["evaluation_ids"]["unlearn_ids"],
        condition_ids, Subset, ConcatDataset)
    shadow_info = None
    if args.shadow_path:
        _, shadow_info = load_shadow(args.shadow_path, data_info, manifest)
        if shadow_info["source_hashes"] != source_metadata(args.ruli_root):
            raise ValueError("Shadows were trained with different RULI source code.")
    if args.train and shadow_info is None:
        raise ValueError("--train requires a separate complete model-specific --shadow-path.")
    # --validate-only actually exercises forward/backward, in a disposable model.
    if args.validate_only:
        architecture = smoke(args.model_name, args.ruli_root.resolve(), args.device)
        gpt2 = architecture if args.model_name == "gpt2" else smoke("gpt2", args.ruli_root.resolve(), args.device)
        result = {"passed": True, "data_identity": identity(data_info), "protocol": checks,
                  "source_hashes": source_metadata(args.ruli_root),
                  "comparison": data_info["comparison"], "prediction_policy": data_info["prediction_policy"],
                  "assignment_sha256": digest(assignments),
                  "architecture_smoke": architecture, "gpt2_regression_smoke": gpt2,
                  "shadow_ready": shadow_info is not None,
                  "shadow_artifact": shadow_info, "research_training_started": False}
        write_json(args.report or args.validation_report, result)
        print("[VERIFY] Fresh matched comparison: raw identity, cohorts, shared assignments, prefix, NPO and capped last-seven loss passed.")
        return
    preflight = require_preflight(args.validation_report, data_info, args.ruli_root) if args.train else None
    output = (args.output_root or paths["results"] / f"seed_{args.seed}").resolve()
    prepared = output.exists() and args.train and (output / "plan.json").is_file()
    if prepared and ((output / "run_metadata.json").exists() or any((output / c).exists() for c in CONDITIONS)):
        raise FileExistsError("Run already started; automatic partial-training resume is not supported.")
    if output.exists() and not prepared:
        raise FileExistsError(f"Refusing to overwrite run {output}")
    COMMON._reset_rng(args.seed, torch, np)
    model = load_model(args.model_name, checkpoint=output / "initial_shared" if prepared else None)
    initial_hash = COMMON._parameter_sha256(model, torch)
    parameter_count = sum(p.numel() for p in model.parameters())
    architecture = type(model).__name__
    model_config = model.config.to_dict()
    if not prepared:
        COMMON._save_checkpoint(model, tokenizer, output / "initial_shared")
    del model
    metadata = {
        "schema_version": 1, "experiment": "model_generalization_2b",
        "seed": args.seed, "model": model_spec(args.model_name),
        "architecture": architecture, "parameter_count": parameter_count,
        "model_config": model_config,
        "data_identity": identity(data_info), "manifest": manifest_info,
        "comparison": data_info["comparison"], "prediction_policy": data_info["prediction_policy"],
        "assignment_sha256": digest(assignments),
        "sets": manifest["sets"], "conditions": condition_ids,
        "source_hashes": source_metadata(args.ruli_root),
        "package_versions": COMMON._software_metadata(torch),
        "initial_parameter_sha256": initial_hash, "shadow_artifact": shadow_info,
        "condition_dataset_hashes": {
            c: {"raw": COMMON._canonical_sha256({"ids": condition_ids[c], "data": identity(data_info)}),
                "tokenized": COMMON._canonical_sha256({"rows": [datasets['target'][i] for i in condition_ids[c]]})}
            for c in CONDITIONS},
        "validation": checks, "preflight_report": preflight, "complete": False,
    }
    # Verify all independent reloads even when only preparing seed 42.
    for condition in CONDITIONS:
        model = load_model(args.model_name, checkpoint=output / "initial_shared")
        if COMMON._parameter_sha256(model, torch) != initial_hash:
            raise ValueError(f"{condition} initial parameters differ.")
        del model
    if prepared:
        plan = read_json(output / "plan.json")
        for field in ("seed", "model", "data_identity", "sets", "conditions", "source_hashes", "initial_parameter_sha256"):
            if plan[field] != metadata[field]:
                raise ValueError(f"Prepared seed plan changed: {field}")
    else:
        write_json(output / "plan.json", metadata)
    if args.prepare_seed_only:
        print(f"[PREPARED] Shared initial checkpoint and reviewed plan: {output}; no training.")
        return
    utils = COMMON._load_ruli_utils(args.ruli_root.resolve() / "text")
    stages = {}
    for condition in CONDITIONS:
        COMMON._reset_rng(args.seed, torch, np)
        model = load_model(args.model_name, args.device, output / "initial_shared")
        if COMMON._parameter_sha256(model, torch) != initial_hash:
            raise ValueError("Initial parameter identity changed.")
        COMMON._reset_rng(args.seed, torch, np)
        model, stages[condition] = run_stages(
            model, tokenizer, initials[condition], retains[condition], forget,
            datasets["validation"], utils, output / condition, args.seed, args.device)
        validate_stage_records(stages[condition], args.seed)
        del model
        COMMON._cleanup_cuda(torch)
    metadata.update(complete=True, stages=stages)
    write_json(output / "run_metadata.json", metadata)
    print(f"[COMPLETE] One seed only: {output}")


if __name__ == "__main__":
    main()
