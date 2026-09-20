#!/usr/bin/env python3
"""Frozen 2D: measure PRE-NPO -> POST-NPO -> two retain epochs -> POST-FT."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location("_2c_for_2d", SCRIPT_DIR / "run_experiment_2c.py")
assert _SPEC is not None and _SPEC.loader is not None
PREVIOUS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(PREVIOUS)
BASE, REFERENCE, COMMON = PREVIOUS.BASE, PREVIOUS.REFERENCE, PREVIOUS.COMMON
CONDITIONS = PREVIOUS.CONDITIONS
HYPERPARAMETERS = {**PREVIOUS.HYPERPARAMETERS, "final_retain_sft_epochs": 2}
CAPTURE = {
    **PREVIOUS.CAPTURE,
    "post_ft": "after_exactly_two_retain_ft_epochs_using_upstream_best_model_selection",
}
STAGES = (
    "initial_sft", "prefix", "save_pre_npo", "measure_pre_npo", "npo",
    "save_post_npo", "measure_post_npo", "final_retain_ft",
    "save_post_ft", "measure_post_ft",
)
REQUIRED_CHECKS = (
    "frozen_artifacts", "last_7_loss_behavior", "shared_initial_parameter_identity",
    "checkpoint_capture_order", "post_npo_measured_before_final_ft",
    "exactly_two_final_retain_epochs", "immutable_tokens_and_sample_order",
    "measurement_preserves_parameters_and_rng", "finite_boundary_losses",
)


def _output_root(path, seed):
    root = BASE._validate_seed_output_path(
        path if path is not None else SCRIPT_DIR / "results" / "experiment_2d" / f"seed_{seed}",
        seed, "output",
    )
    if any(p.lower() in ("experiment_2a", "experiment_2b", "experiment_2c") for p in root.parts):
        raise ValueError("Experiment 2D output must not be inside 2A, 2B, or 2C outputs.")
    return root


def _checkpoint_paths(root):
    return {"initial_shared": root / "initial_shared", **{
        f"{c}_{stage}": root / f"{c}_{stage}" for c in CONDITIONS for stage in CAPTURE
    }}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=BASE.PREREGISTERED_TRAINING_SEEDS)
    parser.add_argument("--ruli-root", type=Path, default=BASE.DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=BASE.DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-manifest-only", action="store_true")
    mode.add_argument("--validate-only", action="store_true",
                      help="Full CPU preflight and temporary initial reloads; no training.")
    return parser.parse_args()


def _immutable_tokens(raw_tokens, ids):
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate evaluation IDs.")
    return MappingProxyType({i: tuple(raw_tokens[i]) for i in ids})


def _token_digest(tokens, ids):
    # Include ordered IDs explicitly: canonical JSON sorts mapping keys.
    return COMMON._canonical_sha256({
        "ordered_ids": list(ids), "tokens": {str(i): list(tokens[i]) for i in ids},
    })


def _kde_references(shadow, ids, gaussian_kde):
    kdes = PREVIOUS._kde_references(shadow, ids, gaussian_kde)
    kdes["post_ft"] = kdes["post_npo"]
    return kdes


@contextmanager
def _preserve_measurement_state(model, torch, numpy):
    """Instrumentation must consume no training randomness or change model modes."""
    python_state, numpy_state = random.getstate(), numpy.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        numpy.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _measure_boundary(model, dataset, tokenizer, device, utils, ids, tokens,
                      torch, numpy, expected_hash, condition, stage, seed, path):
    with _preserve_measurement_state(model, torch, numpy):
        if COMMON._parameter_sha256(model, torch) != expected_hash:
            raise ValueError("Parameters changed between checkpoint and measurement.")
        losses = REFERENCE._run_reference_inference(
            model, dataset, tokenizer, device, utils, ids, tokens,
        )
        if len(losses) != len(ids) or not all(math.isfinite(v) for v in losses):
            raise FloatingPointError("Incomplete or nonfinite boundary losses.")
        if COMMON._parameter_sha256(model, torch) != expected_hash:
            raise ValueError("Measurement changed checkpoint parameters.")
    payload = {
        "experiment": "2D", "seed": seed, "condition": condition, "stage": stage,
        "parameter_sha256": expected_hash, "ordered_sample_ids": list(ids),
        "token_sequences_sha256": _token_digest(tokens, ids),
        "losses": list(losses), "parameters_unchanged": True,
        "rng_and_model_modes_restored": True,
    }
    COMMON._atomic_json(payload, path)
    return {"path": str(path), "file_sha256": COMMON._sha256_file(path),
            "parameter_sha256": expected_hash,
            "token_sequences_sha256": payload["token_sequences_sha256"],
            "sample_count": len(ids)}


@contextmanager
def _record_final_ft_completion(utils, path):
    """Observe train-end state without altering upstream training or callbacks.

    With save_total_limit=1, Trainer can delete the last checkpoint when an
    earlier checkpoint is best. Its retained checkpoint state is not proof of
    the completed epoch count. Capture the actual train-end state instead.
    """
    from transformers import TrainerCallback

    class CompletionRecorder(TrainerCallback):
        def on_train_end(self, args, state, control, **kwargs):
            if args.num_train_epochs != 2:
                raise ValueError("Final retain FT did not request exactly 2 epochs.")
            state.save_to_json(str(path))

    original = utils.Trainer

    def observed_trainer(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.add_callback(CompletionRecorder())
        return trainer

    utils.Trainer = observed_trainer
    try:
        yield
    finally:
        utils.Trainer = original


def _verify_final_ft_epochs(work):
    """Require actual train-end evidence, independent of best-checkpoint pruning."""
    path = work / "completed_trainer_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    epoch = state.get("epoch")
    if state.get("num_train_epochs") != 2 or epoch != 2:
        raise ValueError("Final retain FT did not complete exactly 2 epochs.")
    return {"requested_epochs": 2, "completed_epochs": 2,
            "trainer_states": [{"path": str(path), "sha256": COMMON._sha256_file(path),
                                "epoch": epoch, "global_step": state.get("global_step"),
                                "best_model_checkpoint": state.get("best_model_checkpoint"),
                                "train_end_state": state}],
            "upstream_best_model_selection_preserved": True}


def _run_branch(model, tokenizer, initial_data, retain_data, unlearn_data,
                valid_data, condition, checkpoints, scratch_root, ruli_utils,
                device, torch, measure):
    """Four unchanged upstream calls, with synchronous save/measurement barriers."""
    work = scratch_root / condition
    cleanup, records, events = {}, {}, []

    def capture(stage):
        digest = COMMON._parameter_sha256(model, torch)
        path = checkpoints[f"{condition}_{stage}"]
        COMMON._save_checkpoint(model, tokenizer, path)
        events.append(f"save_{stage}")
        measurement = measure(model, condition, stage, digest)
        events.append(f"measure_{stage}")
        records[stage] = {
            "path": str(path), "parameter_sha256": digest, "capture": CAPTURE[stage],
            "completed_final_retain_epochs": 2 if stage == "post_ft" else 0,
            "measurement": measurement,
        }

    for stage in BASE.TRAINER_STAGE_NAMES:
        with COMMON._working_directory(work / stage):
            if stage == "initial_sft":
                model = ruli_utils.train_sft(model, initial_data, valid_data, tokenizer, BASE.SFT_EPOCHS)
            elif stage == "prefix":
                model = ruli_utils.train_prefix(model, initial_data, valid_data, tokenizer, BASE.PREFIX_EPOCHS)
            elif stage == "npo":
                model = ruli_utils.unlearn_model(
                    model, unlearn_data, retain_data, valid_data, tokenizer,
                    SimpleNamespace(device=device, unlearn_epochs=BASE.NPO_EPOCHS, unlearn_method="npo"),
                )
            else:
                if events[-1] != "measure_post_npo":
                    raise RuntimeError("POST-NPO measurement must succeed before final FT.")
                with _record_final_ft_completion(
                    ruli_utils, work / stage / "completed_trainer_state.json",
                ):
                    model = ruli_utils.train_sft(model, retain_data, valid_data, tokenizer, 2)
        events.append(stage)
        if stage == "final_retain_ft":
            records["final_retain_ft"] = _verify_final_ft_epochs(work / stage)
        boundary = {"prefix": "pre_npo", "npo": "post_npo", "final_retain_ft": "post_ft"}.get(stage)
        if boundary:
            capture(boundary)
        cleanup[stage] = BASE._cleanup_successful_trainer_stage(work / stage, scratch_root)
    BASE._remove_empty_scratch_parents(work, scratch_root)
    if tuple(events) != STAGES:
        raise RuntimeError("Experiment 2D capture/measurement order changed.")
    records["stage_order"] = events
    return records, cleanup


def main():
    args = parse_args()
    manifest, manifest_metadata = REFERENCE._validate_manifest(args.manifest)
    condition_ids, validation = BASE._validate_experiment_2b_protocol(manifest)
    output_root = _output_root(args.output_root, args.seed)
    print("[VERIFY] Exact frozen manifest, cohorts, and 2B condition memberships passed.")
    if args.validate_manifest_only:
        return
    if not args.validate_only and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing nonempty Experiment 2D output: {output_root}")
    ruli_root = args.ruli_root.resolve()
    shadow_path, target_path = BASE._artifact_paths(args, ruli_root)
    artifacts = manifest["input_artifacts"]
    COMMON._verify_file_artifact(shadow_path, artifacts["shadow_artifact"], "Frozen 9-shadow artifact")
    COMMON._verify_target_dataset_storage(target_path, artifacts["target_dataset"])

    import numpy as np
    import torch
    from datasets import load_from_disk
    from torch.utils.data import ConcatDataset, Subset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.validate_only and str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable.")
    sources = PREVIOUS._source_metadata(ruli_root)
    utils = COMMON._load_ruli_utils(ruli_root / "text")
    loss_metadata = REFERENCE._validate_reference_loss_path(utils, torch)
    shadow, shadow_metadata = PREVIOUS._load_shadow(shadow_path, artifacts["shadow_artifact"], manifest, torch)
    ids = tuple(manifest["evaluation_ids"]["unlearn_ids"])
    _kde_references(shadow, ids, utils.gaussian_kde)
    REFERENCE._build_kde_references(shadow, list(ids) + manifest["evaluation_ids"]["out_ids"], utils.gaussian_kde)
    tokenizer = AutoTokenizer.from_pretrained(BASE.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    target, target_metadata, raw_tokens = REFERENCE._validate_target_dataset(
        target_path, artifacts["target_dataset"], manifest, tokenizer, load_from_disk,
    )
    tokens = _immutable_tokens(raw_tokens, ids)
    if any(i < 0 or i >= len(target) for values in condition_ids.values() for i in values):
        raise ValueError("A condition ID is outside the frozen target dataset.")
    with COMMON._working_directory(ruli_root / "text"):
        train, valid, _ = utils.load_data("WikiText103", SimpleNamespace(model_name=BASE.MODEL_NAME))
    attack = train.shuffle(seed=BASE.FROZEN_ARTIFACT_SEED).select(range(BASE.ATTACK_SIZE))
    background = COMMON._validate_background_dataset(train, attack, tokenizer, manifest)
    unlearn, targets, retain, initial = BASE._build_condition_datasets(
        target, attack, list(ids), condition_ids, Subset, ConcatDataset,
    )
    if args.validate_only:
        with tempfile.TemporaryDirectory(prefix="experiment_2d_initial_") as directory:
            digest, _, _ = BASE._create_and_verify_shared_initial(
                Path(directory) / "initial_shared", args.seed, AutoModelForCausalLM, tokenizer, torch, np,
            )
        print(f"[VERIFY] Full 2D preflight passed; identical initial SHA-256={digest}.")
        print("[VERIFY] No training or persistent experiment outputs.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints, scratch = _checkpoint_paths(output_root), output_root / "trainer_work"
    digest, config, preflight = BASE._create_and_verify_shared_initial(
        checkpoints["initial_shared"], args.seed, AutoModelForCausalLM, tokenizer, torch, np,
    )
    COMMON._configure_training_arguments_seed(utils, args.seed)

    def measure(model, condition, stage, expected_hash):
        return _measure_boundary(
            model, target, tokenizer, args.device, utils, ids, tokens, torch, np,
            expected_hash, condition, stage, args.seed,
            output_root / "measurements" / f"{condition}_{stage}.json",
        )

    records = {"initial_shared": {"path": str(checkpoints["initial_shared"]), "parameter_sha256": digest}}
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
            condition, checkpoints, scratch, utils, args.device, torch, measure,
        )
        records[condition]["starting_parameter_sha256"] = starts[condition]
        del model
        COMMON._cleanup_cuda(torch)
    BASE._assert_initial_parameter_identity(digest, starts, "Actual 2D branches")
    if scratch.exists():
        raise RuntimeError(f"Unexpected Trainer scratch remains: {scratch}")
    metadata = {
        "schema_version": 1, "experiment": "2D", "seed": args.seed,
        "manifest": manifest_metadata, "model_and_hyperparameters": HYPERPARAMETERS,
        "model_config": config, "upstream_ruli_source_files": sources,
        "training_implementation": "unchanged 2B upstream train_sft/train_prefix/unlearn_model helpers",
        "loss": loss_metadata, "ordered_target_dataset_ids": condition_ids,
        "ordered_unlearn_ids": list(ids), "token_sequences_sha256": _token_digest(tokens, ids),
        "background_dataset": background,
        "input_artifacts": {"shadow": shadow_metadata, "target_dataset": target_metadata},
        "dataset_sizes": {
            "condition_target": {c: len(targets[c]) for c in CONDITIONS},
            "initial_sft_and_prefix": {c: len(initial[c]) for c in CONDITIONS},
            "npo_retain": {c: len(retain[c]) for c in CONDITIONS},
            "final_retain": {c: len(retain[c]) for c in CONDITIONS}, "npo_forget": len(unlearn),
        },
        "checkpoints": records,
        "starting_parameter_identity": {"passed": True, "sha256": digest,
            "preflight_reload_sha256": preflight, "actual_training_branch_sha256": starts},
        "rng_policy": {"training_seed": args.seed, "data_seed": args.seed,
            "frozen_wikitext_selection_seed": 42, "reset_before_each_branch": True,
            "measurement_rng_and_modes_restored": True},
        "trainer_work_cleanup": cleanup,
        "validation_results": {**validation, **dict.fromkeys(REQUIRED_CHECKS, True)},
        "git": {"ruli": COMMON._git_metadata(ruli_root),
                "ruli_experiments": COMMON._git_metadata(BASE.REPOSITORY_ROOT)},
        "software": COMMON._software_metadata(torch),
        "deviations_from_reference_ruli_behavior": [],
        "instrumentation": "Save and measure three boundaries; restore all RNG and module modes.",
    }
    COMMON._atomic_json(metadata, output_root / "run_metadata.json")
    print(f"[INFO] Experiment 2D training and boundary measurements complete: {output_root}")


if __name__ == "__main__":
    main()
