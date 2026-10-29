"""Fresh matched comparison: one frozen new map for GPT-2 and Pythia, nine shadows each."""
from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

from common import (COMMON, DEFAULT_MODEL, HERE, MANIFEST, RULI, REFERENCE, SHADOW_FIELDS,
                    digest, identity, load_bundle, load_model, load_tokenizer, manifest_at,
                    model_spec, read_json, require_preflight, source_metadata, validate_assignments,
                    validate_shadow_losses, write_json, model_paths, ASSIGNMENT_PATH,
                    prediction_policy)
from training import run_stages, validate_stage_records


def shadow_partition(mapping, index):
    return {role: [int(idx) for idx, roles in sorted(mapping.items(), key=lambda item: int(item[0]))
                   if index in roles[role]] for role in ("IN", "OUT", "UNLEARN")}


def train_one(index, mapping, datasets, tokenizer, name, utils, output, seed, device,
              smoke=False):
    import numpy as np
    import torch
    from torch.utils.data import ConcatDataset, Subset
    partition = shadow_partition(mapping, index)
    if any(not ids for ids in partition.values()):
        raise ValueError(f"Shadow {index} has an empty role.")
    target = datasets["target"]
    combined_ids = partition["IN"] + partition["UNLEARN"]
    # Match LanguageMIA.train_shadow_models' ordering and nesting exactly.
    combined = ConcatDataset([ConcatDataset([Subset(target, combined_ids)]), datasets["background"]])
    retain = ConcatDataset([ConcatDataset([Subset(target, partition["IN"])]), datasets["background"]])
    forget = ConcatDataset([Subset(target, partition["UNLEARN"])])
    COMMON._reset_rng(seed, torch, np)
    model = load_model(name, device)
    initial_hash = COMMON._parameter_sha256(model, torch)
    tokens = {i: target[i]["input_ids"] for i in range(len(target))}
    results = {field: {} for field in SHADOW_FIELDS}

    def measure(stage, trained):
        # Upstream shadow inference preserves the model's current training mode.
        # Use LanguageMIA's actual method (rather than MIAEvaluator's eval wrapper).
        from train_text import LanguageMIA
        mia = object.__new__(LanguageMIA)
        suffix = "original" if stage == "pre_npo" else "unlearned"
        for role in ("IN", "OUT", "UNLEARN"):
            ids = partition[role]
            losses = mia.perform_lm_inference(trained, [tokens[i] for i in ids], tokenizer, device)
            if len(losses) != len(ids):
                raise ValueError("Shadow inference dropped samples.")
            results[f"{role.lower()}_{suffix}"] = dict(zip(ids, losses, strict=True))

    model, records = run_stages(model, tokenizer, combined, retain, forget, datasets["validation"],
                               utils, output, seed, device, smoke, measure)
    del model
    COMMON._cleanup_cuda(torch)
    return results, {"shadow_index": index, "training_seed": seed,
                     "partition": partition, "initial_parameter_sha256": initial_hash,
                     "stages": records, "smoke_only": smoke}


def assemble(root, meta, manifest, assignment_payload, expected_sources, seed):
    import torch
    ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    mapping = validate_assignments(assignment_payload, manifest)
    merged = {field: {idx: [] for idx in ids} for field in SHADOW_FIELDS}
    artifact = root / f"shadow_9_attack_random_npo_{meta['model']['model_name'].split('/')[-1].replace('-', '')}.pth"
    if artifact.exists() or artifact.with_suffix(".metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite {artifact}")
    sources = []
    initial_hashes = set()
    for index in range(9):
        path = root / f"shadow_{index}" / "losses.json"
        payload = read_json(path)
        expected = shadow_partition(mapping, index)
        if (payload["data_identity"] != identity(meta)
                or payload.get("comparison") != meta["comparison"]
                or payload.get("prediction_policy") != meta["prediction_policy"]
                or digest(payload.get("target_prediction_counts")) != meta["prediction_policy"]["per_sample_sha256"]
                or payload["assignment_sha256"] != digest(assignment_payload)
                or payload["source_hashes"] != expected_sources
                or payload["run"]["shadow_index"] != index
                or payload["run"]["training_seed"] != seed
                or payload["run"]["partition"] != expected
                or payload["run"]["smoke_only"] is not False):
            raise ValueError(f"Shadow {index} provenance mismatch or smoke contamination.")
        initial_hashes.add(payload["run"]["initial_parameter_sha256"])
        validate_stage_records(payload["run"]["stages"], seed)
        for field in SHADOW_FIELDS:
            values = payload["losses"][field]
            role = field.split("_")[0].upper()
            if sorted(map(int, values)) != expected[role]:
                raise ValueError("Shadow loss IDs differ from the frozen fresh-comparison assignments.")
            for idx, value in values.items():
                merged[field][int(idx)].append(float(value))
        sources.append({"path": str(path.resolve()), "sha256": COMMON._sha256_file(path)})
    if len(initial_hashes) != 1:
        raise ValueError("Shadows did not start from identical pretrained parameters.")
    validate_shadow_losses(merged, ids)
    # Fail on unusable KDEs before writing the final artifact.
    from scipy.stats import gaussian_kde
    REFERENCE._build_kde_references(merged, list(range(200, 600)), gaussian_kde)
    torch.save(merged, artifact)
    write_json(artifact.with_suffix(".metadata.json"), {
        "complete": True, "smoke_only": False, "data_identity": identity(meta),
        "sha256": COMMON._sha256_file(artifact), "shadow_count": 9,
        "observations_per_sample_per_role": 3, "assignment_provenance": assignment_payload,
        "comparison": meta["comparison"], "prediction_policy": meta["prediction_policy"],
        "source_hashes": expected_sources, "individual_shadows": sources,
        "training_seed": seed, "initial_parameter_sha256": next(iter(initial_hashes)),
    })
    print(f"[COMPLETE] Separate nine-shadow artifact: {artifact}")


def shadow_smoke(args):
    """Synthetic assignments exercise one real shadow path, never a research artifact."""
    from datasets import Dataset
    tokenizer = load_tokenizer(args.model_name)
    encoded = tokenizer([f"Synthetic example number {i} tests the complete shadow model training path." for i in range(18)])
    data = Dataset.from_dict({k: encoded[k] for k in ("input_ids", "attention_mask")})
    rng = random.Random(42)
    mapping = {}
    for idx in range(18):
        order = list(range(9))
        rng.shuffle(order)
        mapping[str(idx)] = dict(IN=order[:3], OUT=order[3:6], UNLEARN=order[6:])
    utils = COMMON._load_ruli_utils(args.ruli_root.resolve() / "text")
    with tempfile.TemporaryDirectory(prefix="ruli_shadow_smoke_") as directory:
        losses, record = train_one(0, mapping, {k: data for k in ("target", "background", "validation")},
                                   tokenizer, args.model_name, utils, Path(directory), 42, args.device, True)
    result = {"passed": True, "smoke_only": True, "research_result": False,
              "assignments": "synthetic fixture, not the original GPT-2 assignments",
              "model": model_spec(args.model_name), "run": record, "losses": losses}
    if args.report:
        write_json(args.report, result)
    print("[VERIFY] One synthetic shadow traversed all four stages and six loss distributions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--ruli-root", type=Path, default=RULI)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--assignments", type=Path, default=ASSIGNMENT_PATH)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--shadow-index", type=int, choices=range(9), default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validation-report", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    paths = model_paths(args.model_name)
    args.data = args.data or paths["data"]
    args.output_root = args.output_root or paths["shadows"]
    args.validation_report = args.validation_report or paths["validation"]
    if args.smoke_test:
        shadow_smoke(args)
        return
    manifest, _, _, _ = manifest_at(args.manifest)
    assignment_payload = read_json(args.assignments)
    mapping = validate_assignments(assignment_payload, manifest)
    datasets, tokenizer, meta = load_bundle(args.data, manifest, args.model_name)
    prediction_records, policy = prediction_policy(datasets["target"], args.model_name, manifest)
    sources = source_metadata(args.ruli_root)
    if args.validate_only:
        result = {"passed": True, "data_identity": identity(meta),
                  "comparison": meta["comparison"], "prediction_policy": policy,
                  "shadow_index": args.shadow_index, "partition": shadow_partition(mapping, args.shadow_index),
                  "assignment_sha256": digest(assignment_payload), "training_started": False}
        if args.report:
            write_json(args.report, result)
        print("[VERIFY] Fresh matched comparison, not an exact reproduction: shared assignments, cohort prediction counts, data and shadow partition passed; no training.")
        return
    root = args.output_root.resolve()
    if args.assemble:
        assemble(root, meta, manifest, assignment_payload, sources, args.seed)
        return
    preflight = require_preflight(args.validation_report, meta, args.ruli_root)
    output = root / f"shadow_{args.shadow_index}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    utils = COMMON._load_ruli_utils(args.ruli_root.resolve() / "text")
    losses, record = train_one(args.shadow_index, mapping, datasets, tokenizer, args.model_name,
                               utils, output, args.seed, args.device)
    write_json(output / "losses.json", {"data_identity": identity(meta),
               "comparison": meta["comparison"], "prediction_policy": policy,
               "target_prediction_counts": prediction_records,
               "assignment_sha256": digest(assignment_payload), "source_hashes": sources,
               "preflight_report": preflight, "run": record, "losses": losses})


if __name__ == "__main__":
    main()
