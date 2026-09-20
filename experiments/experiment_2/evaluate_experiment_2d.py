#!/usr/bin/env python3
"""Three-stage raw-loss DiD, with ordinary final fixed-shadow RULI diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import tempfile
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "_2d_runner", Path(__file__).resolve().parent / "run_experiment_2d.py",
)
assert _SPEC is not None and _SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(RUNNER)
PREVIOUS, BASE, COMMON, REFERENCE = RUNNER.PREVIOUS, RUNNER.BASE, RUNNER.COMMON, RUNNER.REFERENCE


CONDITIONS = RUNNER.CONDITIONS
CHECKS = PREVIOUS._load_module("evaluate_experiment_2c.py", "_2c_checks_for_2d")
_mapping, _equal = COMMON._mapping, CHECKS._require_equal
OUTPUT_FILENAMES = (
    "per_sample_three_stage.csv", "primary_contrast.csv", "evaluation_summary.json",
    "post_ft_ruli_per_sample.csv",
)
KDE_NOTE = (
    "Pre-NPO uses original-state shadows; post-NPO and post-FT use unlearned-state "
    "shadows. Direct pre/post KDE subtraction is not a stage-localization estimand. "
    "The fixed unlearned shadows include final retain FT; immediate post-NPO target "
    "models precede that stage. All KDE outcomes are secondary diagnostics."
)
ESTIMANDS = {
    "primary_cohort": "28 frozen supported S samples; LOW versus PLACEBO",
    "delta_npo": "post_npo_loss - pre_loss",
    "delta_ft": "post_ft_loss - post_npo_loss",
    "delta_total": "post_ft_loss - pre_loss",
    "NPO_DiD": "LOW_delta_npo - PLACEBO_delta_npo",
    "FT_DiD": "LOW_delta_ft - PLACEBO_delta_ft",
    "TOTAL_DiD": "LOW_delta_total - PLACEBO_delta_total",
    "predicted_direction": "> 0 (strict; ties do not count)",
    "NPO_positive": "NPO increased forgotten-target loss more in LOW",
    "FT_positive": (
        "LOW loss changed more positively than PLACEBO during FT; when FT lowers "
        "loss, PLACEBO had a larger decrease / stronger relearning than LOW. "
        "Inspect absolute deltas before concluding either condition relearned."
    ),
    "TOTAL_positive": "larger net loss increase from pre-NPO to post-FT in LOW",
    "final_loss_gap": "LOW_post_ft_loss - PLACEBO_post_ft_loss; positive means stronger final suppression in LOW",
    "reference_condition": "HIGH minus PLACEBO repeats every contrast without replacing the primary comparison",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, choices=BASE.PREREGISTERED_TRAINING_SEEDS)
    parser.add_argument("--ruli-root", type=Path, default=BASE.DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=BASE.DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument("--experiment-output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validate-only", action="store_true", help=(
        "Check artifacts, metadata, boundary measurements, loss, KDEs, and checkpoint "
        "structure; loaded parameter hashes are checked during full evaluation."
    ))
    return parser.parse_args()


def _validate_training_run_metadata(root, seed, manifest):
    metadata = _mapping(json.loads((root / "run_metadata.json").read_text(encoding="utf-8")), "run metadata")
    _equal(metadata.get("experiment"), "2D", "experiment")
    _equal(metadata.get("seed"), seed, "seed")
    _equal(_mapping(metadata.get("manifest"), "manifest").get("declared_canonical_content_sha256"),
           manifest["manifest_hash"]["sha256"], "manifest hash")
    hyper = _mapping(metadata.get("model_and_hyperparameters"), "hyperparameters")
    for key, value in RUNNER.HYPERPARAMETERS.items():
        _equal(hyper.get(key), value, key)
    background = _mapping(metadata.get("background_dataset"), "background")
    for key, value in {"selection_seed": 42, "count": 15_000,
                       "membership_sha256": manifest["shared_wikitext_background"]["membership_sha256"]}.items():
        _equal(background.get(key), value, f"background {key}")
    _equal(metadata.get("ordered_unlearn_ids"), manifest["evaluation_ids"]["unlearn_ids"], "UNLEARN order")
    CHECKS._validate_hash(metadata.get("token_sequences_sha256"))
    ids = _mapping(metadata.get("ordered_target_dataset_ids"), "target IDs")
    sizes = _mapping(metadata.get("dataset_sizes"), "sizes")
    _equal(sizes.get("npo_forget"), 200, "forget count")
    for c in CONDITIONS:
        _equal(ids.get(c), manifest["conditions"][c]["ordered_target_dataset_ids"], f"{c} target IDs")
        for field, count in (("condition_target", 200), ("initial_sft_and_prefix", 15_400),
                             ("npo_retain", 15_200), ("final_retain", 15_200)):
            _equal(_mapping(sizes.get(field), field).get(c), count, f"{c} {field}")
    identity = _mapping(metadata.get("starting_parameter_identity"), "initial identity")
    _equal(identity.get("passed"), True, "initial identity")
    digest = identity.get("sha256")
    CHECKS._validate_hash(digest)
    for field in ("preflight_reload_sha256", "actual_training_branch_sha256"):
        BASE._assert_initial_parameter_identity(digest, _mapping(identity.get(field), field), field)
    records = _mapping(metadata.get("checkpoints"), "checkpoints")
    _equal(_mapping(records.get("initial_shared"), "initial_shared").get("parameter_sha256"), digest, "shared hash")
    for c in CONDITIONS:
        record = _mapping(records.get(c), c)
        _equal(record.get("starting_parameter_sha256"), digest, f"{c} start")
        _equal(record.get("stage_order"), list(RUNNER.STAGES), f"{c} stage order")
        ft = _mapping(record.get("final_retain_ft"), "final FT evidence")
        for field in ("requested_epochs", "completed_epochs"):
            _equal(ft.get(field), 2, f"{c} {field}")
        _equal(ft.get("upstream_best_model_selection_preserved"), True, "best-model selection")
        states = ft.get("trainer_states")
        if not isinstance(states, list) or not states:
            raise ValueError("Missing final FT Trainer state evidence.")
        for state in states:
            epoch = _mapping(state, "Trainer state").get("epoch")
            if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or not 0 < epoch <= 2:
                raise ValueError("Invalid completed FT epoch.")
            CHECKS._validate_hash(state.get("sha256"))
            completion = _mapping(state.get("train_end_state"), "train-end state")
            _equal(completion.get("epoch"), 2, "train-end epoch")
            _equal(completion.get("num_train_epochs"), 2, "Trainer total epochs")
            _equal(completion.get("global_step"), state.get("global_step"), "train-end steps")
            _equal(completion.get("best_model_checkpoint"), state.get("best_model_checkpoint"), "best checkpoint")
        _equal(max(s["epoch"] for s in states), 2, "actual completed FT epochs")
        for stage, capture in RUNNER.CAPTURE.items():
            checkpoint = _mapping(record.get(stage), f"{c} {stage}")
            _equal(checkpoint.get("capture"), capture, f"{c} {stage} capture")
            _equal(checkpoint.get("completed_final_retain_epochs"), 2 if stage == "post_ft" else 0, "FT boundary")
            CHECKS._validate_hash(checkpoint.get("parameter_sha256"))
            proof = _mapping(checkpoint.get("measurement"), "boundary measurement")
            _equal(proof.get("parameter_sha256"), checkpoint["parameter_sha256"], "measurement parameters")
            _equal(proof.get("token_sequences_sha256"), metadata["token_sequences_sha256"], "measurement tokens")
            _equal(proof.get("sample_count"), 200, "measurement count")
            CHECKS._validate_hash(proof.get("file_sha256"))
    rng = _mapping(metadata.get("rng_policy"), "RNG policy")
    for key, expected in {"training_seed": seed, "data_seed": seed,
                          "frozen_wikitext_selection_seed": 42, "reset_before_each_branch": True,
                          "measurement_rng_and_modes_restored": True}.items():
        _equal(rng.get(key), expected, key)
    validations = _mapping(metadata.get("validation_results"), "validation results")
    _, protocol = BASE._validate_experiment_2b_protocol(manifest)
    for key in (*RUNNER.REQUIRED_CHECKS, *protocol):
        _equal(validations.get(key), True, key)
    if any(v is not True for v in validations.values()):
        raise ValueError("A training validation failed.")
    return metadata


def _load_boundary_measurements(root, training, ids, token_digest):
    _equal(training["token_sequences_sha256"], token_digest, "training/evaluation tokens")
    result = {}
    for c in CONDITIONS:
        result[c] = {}
        for stage in RUNNER.CAPTURE:
            # Resolve within the supplied root, so artifact copies remain portable.
            path = root / "measurements" / f"{c}_{stage}.json"
            record = training["checkpoints"][c][stage]
            _equal(COMMON._sha256_file(path), record["measurement"]["file_sha256"], "boundary file hash")
            payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "boundary payload")
            for key, value in {"experiment": "2D", "seed": training["seed"], "condition": c,
                               "stage": stage, "parameter_sha256": record["parameter_sha256"],
                               "ordered_sample_ids": list(ids), "token_sequences_sha256": token_digest,
                               "parameters_unchanged": True, "rng_and_model_modes_restored": True}.items():
                _equal(payload.get(key), value, f"boundary {key}")
            losses = payload.get("losses")
            if not isinstance(losses, list) or len(losses) != len(ids) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in losses
            ):
                raise ValueError("Missing, nonfinite, or incomplete boundary losses.")
            result[c][stage] = losses
    return result


def _verify_boundary_losses(actual, saved, label):
    if len(actual) != len(saved) or any(
        not math.isfinite(a) or not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
        for a, b in zip(actual, saved, strict=True)
    ):
        raise ValueError(f"{label} reloaded losses differ from the immediate boundary measurement.")


def _paired_rows(measurements, ids, supported, negative, seed):
    if set(measurements) != set(CONDITIONS) or len(set(ids)) != len(ids):
        raise ValueError("Invalid condition or evaluation identities.")
    for c in CONDITIONS:
        if set(measurements[c]) != set(RUNNER.CAPTURE):
            raise ValueError(f"Missing or extra stage for {c}.")
        for stage, samples in measurements[c].items():
            if list(samples) != list(ids):
                raise ValueError(f"{c} {stage} sample identity/order mismatch.")
            for values in samples.values():
                if not all(math.isfinite(values[f]) for f in ("loss", "privacy_score", "privacy_log_odds")):
                    raise FloatingPointError("Nonfinite loss or KDE score.")
    rows = []
    for i in ids:
        row = {"sample_id": i, "seed": seed, "is_supported_S": i in supported, "is_negative_control": i in negative}
        for c in CONDITIONS:
            pre, npo, ft = (measurements[c][s][i]["loss"] for s in RUNNER.CAPTURE)
            row.update({f"{c}_pre_loss": pre, f"{c}_post_npo_loss": npo, f"{c}_post_ft_loss": ft,
                        f"{c}_delta_npo": npo - pre, f"{c}_delta_ft": ft - npo, f"{c}_delta_total": ft - pre})
            for stage in RUNNER.CAPTURE:
                for field in ("privacy_log_odds", "privacy_score"):
                    row[f"{c}_{stage}_{field}"] = measurements[c][stage][i][field]
        for c in ("LOW", "HIGH"):
            for stage in ("npo", "ft", "total"):
                row[f"{stage.upper()}_DiD_{c}_PLACEBO"] = row[f"{c}_delta_{stage}"] - row[f"PLACEBO_delta_{stage}"]
            row[f"post_ft_loss_{c}_minus_PLACEBO"] = row[f"{c}_post_ft_loss"] - row["PLACEBO_post_ft_loss"]
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError("Nonfinite derived contrast.")
        rows.append(row)
    return rows


def _cohort_summary(rows, ids):
    by_id = {row["sample_id"]: row for row in rows}
    selected = [by_id[i] for i in ids]

    def describe(field, direction=False):
        values = [r[field] for r in selected]
        result = {"mean": statistics.mean(values), "median": statistics.median(values)}
        if direction:
            n = sum(v > 0 for v in values)
            result.update(number_positive=n, fraction_in_predicted_direction=n / len(values), predicted_direction="> 0")
        return result

    return {"n": len(selected), "conditions": {
        c: {f: describe(f"{c}_{f}") for f in ("pre_loss", "post_npo_loss", "post_ft_loss", "delta_npo", "delta_ft", "delta_total")}
        for c in CONDITIONS
    }, "contrasts": {
        field: describe(field, True) for c in ("LOW", "HIGH")
        for field in (*[f"{s}_DiD_{c}_PLACEBO" for s in ("NPO", "FT", "TOTAL")], f"post_ft_loss_{c}_minus_PLACEBO")
    }}


def _write_outputs(output_dir, rows, primary, summary, ruli_rows):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="write_2d_", dir=output_dir) as directory:
        temp = Path(directory)
        for name, data in ((OUTPUT_FILENAMES[0], rows), (OUTPUT_FILENAMES[1], primary), (OUTPUT_FILENAMES[3], ruli_rows)):
            REFERENCE._write_csv(temp / name, data)
        (temp / OUTPUT_FILENAMES[2]).write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for name in OUTPUT_FILENAMES:
            (temp / name).replace(output_dir / name)


def main():
    args = parse_args()
    manifest, manifest_metadata = REFERENCE._validate_manifest(args.manifest)
    BASE._validate_experiment_2b_protocol(manifest)
    root = RUNNER._output_root(args.experiment_output, args.seed)
    training = _validate_training_run_metadata(root, args.seed, manifest)
    checkpoints = RUNNER._checkpoint_paths(root)
    checkpoint_metadata = {key: REFERENCE._checkpoint_metadata(path, key) for key, path in checkpoints.items()}
    sources = PREVIOUS._source_metadata(args.ruli_root.resolve())
    for name, record in sources.items():
        _equal(record["sha256"], training["upstream_ruli_source_files"][name]["sha256"], f"{name} source hash")
    shadow_path, target_path = BASE._artifact_paths(args, args.ruli_root.resolve())
    artifacts = manifest["input_artifacts"]
    COMMON._verify_file_artifact(shadow_path, artifacts["shadow_artifact"], "Frozen 9-shadow artifact")
    COMMON._verify_target_dataset_storage(target_path, artifacts["target_dataset"])

    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    shadow, shadow_metadata = PREVIOUS._load_shadow(shadow_path, artifacts["shadow_artifact"], manifest, torch)
    utils = COMMON._load_ruli_utils(args.ruli_root.resolve() / "text")
    loss_metadata = REFERENCE._validate_reference_loss_path(utils, torch)
    _equal(loss_metadata, training.get("loss"), "validated loss implementation")
    tokenizer = AutoTokenizer.from_pretrained(checkpoints["initial_shared"])
    dataset, target_metadata, raw_tokens = REFERENCE._validate_target_dataset(
        target_path, artifacts["target_dataset"], manifest, tokenizer, load_from_disk,
    )
    ids = tuple(manifest["evaluation_ids"]["unlearn_ids"])
    out_ids = manifest["evaluation_ids"]["out_ids"]
    tokens = RUNNER._immutable_tokens(raw_tokens, list(ids) + out_ids)
    token_digest = RUNNER._token_digest(tokens, ids)
    boundary = _load_boundary_measurements(root, training, ids, token_digest)
    kdes = RUNNER._kde_references(shadow, ids, utils.gaussian_kde)
    final_kdes = REFERENCE._build_kde_references(shadow, list(ids) + out_ids, utils.gaussian_kde)
    if args.validate_only:
        print("[VERIFY] 2D metadata, boundary records, frozen inputs, loss, KDEs, and ten checkpoint structures passed.")
        print("[VERIFY] No inference or output writes; loaded parameter hashes are checked during full evaluation.")
        return
    output_dir = root / "evaluation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty evaluation output: {output_dir}")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable.")
    device = torch.device(args.device)
    initial = AutoModelForCausalLM.from_pretrained(checkpoints["initial_shared"])
    CHECKS._verify_model_hash(initial, training["checkpoints"]["initial_shared"]["parameter_sha256"], "initial_shared", torch)
    del initial
    supported, negative = manifest["sets"]["S_sample_ids"], manifest["sets"]["negative_control_sample_ids"]
    measurements, aggregate, ruli_rows = {}, {}, []
    for c in CONDITIONS:
        measurements[c] = {}
        for stage in RUNNER.CAPTURE:
            label = f"{c}_{stage}"
            print(f"[INFO] Evaluating {label}.")
            REFERENCE._reset_determinism(torch, args.seed)
            model = AutoModelForCausalLM.from_pretrained(checkpoints[label])
            CHECKS._verify_model_hash(model, training["checkpoints"][c][stage]["parameter_sha256"], label, torch)
            model = model.to(device)
            if stage == "post_ft":
                # Exact helper used by 2B: UNLEARN + OUT privacy, checked against upstream aggregate metrics.
                final_rows, aggregate[c] = REFERENCE._evaluate_condition(
                    c, model, dataset, tokenizer, device, utils, shadow, final_kdes,
                    list(ids), out_ids, tokens, set(supported), set(negative), None,
                )
                ruli_rows.extend(final_rows)
                unlearn_rows = [r for r in final_rows if r["split"] == "unlearn"]
                _equal([r["sample_id"] for r in unlearn_rows], list(ids), "final RULI UNLEARN order")
                losses = [r["observed_loss"] for r in unlearn_rows]
            else:
                losses = REFERENCE._run_reference_inference(model, dataset, tokenizer, device, utils, ids, tokens)
            _verify_boundary_losses(losses, boundary[c][stage], label)
            values = {}
            for i, loss in zip(ids, losses, strict=True):
                log_odds, score = REFERENCE._score_kde(loss, kdes[stage][i], stage, c, i)
                values[i] = {"loss": loss, "privacy_log_odds": log_odds, "privacy_score": score}
            measurements[c][stage] = values
            del model
            COMMON._cleanup_cuda(torch)
    REFERENCE._validate_condition_row_alignment(ruli_rows, list(ids), out_ids)
    rows = _paired_rows(measurements, ids, set(supported), set(negative), args.seed)
    by_id = {r["sample_id"]: r for r in rows}
    primary = [by_id[i] for i in supported]
    cohorts = {"supported_S": supported, "negative_controls": negative, "all_UNLEARN": ids}
    summary = {
        "schema_version": 1, "experiment": "2D", "seed": args.seed,
        "single_seed_diagnostic_only": True, "primary_estimands": ESTIMANDS,
        "manifest": manifest_metadata, "shadow_artifact": shadow_metadata,
        "target_dataset": target_metadata, "checkpoints": checkpoint_metadata,
        "training_run_metadata": {"path": str(root / "run_metadata.json"), "sha256": COMMON._sha256_file(root / "run_metadata.json")},
        "sample_counts": {"UNLEARN": len(ids), "supported_S": len(supported), "negative_controls": len(negative),
                          "per_sample_rows": len(rows), "primary_contrast_rows": len(primary), "post_ft_ruli_rows": len(ruli_rows)},
        "scoring": {"loss": loss_metadata, "privacy_note": KDE_NOTE,
                    "privacy_log_odds": "log KDE_unlearn(loss) - log KDE_out(loss)",
                    "privacy_score": "p_unlearn / (p_unlearn + p_out + 1e-12)",
                    "kde_bandwidth": "unchanged scipy gaussian_kde default (Scott)",
                    "post_ft_2b_connection": "identical _evaluate_condition privacy path, fixed shadows and UNLEARN/OUT IDs; efficacy is not requested"},
        "primary_descriptive_results": {name: _cohort_summary(rows, cohort) for name, cohort in cohorts.items()},
        "secondary_post_ft_ruli": {"aggregate_reference_metrics": aggregate,
            "cohorts": {name: REFERENCE._cohort_summary(ruli_rows, cohort) for name, cohort in cohorts.items()}},
        "alignment": {"ordered_unlearn_ids": list(ids), "token_sequences_sha256": token_digest,
                      "same_immutable_tokens_all_nine_passes": True,
                      "boundary_remeasurement_tolerance": {"rel_tol": 1e-5, "abs_tol": 1e-5}},
        "validation_status": {"passed": True, **dict.fromkeys(RUNNER.REQUIRED_CHECKS, "passed"),
            "all_ten_loaded_parameter_hashes": "passed", "source_hashes_unchanged": "passed",
            "boundary_measurement_hashes_and_reloaded_losses": "passed",
            "condition_sample_order_and_identity": "passed",
            "post_ft_scores_reproduce_upstream_metrics": "passed", "nonfinite_losses_or_kde_scores": 0},
        "package_versions": REFERENCE._package_versions(), "device": str(device),
        "deviations_from_reference_ruli_behavior": [],
        "additional_diagnostics": "Raw-loss stage DiD; original-state KDE before NPO; no cross-stage KDE subtraction or five-seed statistics.",
    }
    _write_outputs(output_dir, rows, primary, summary, ruli_rows)
    print(f"[INFO] Wrote 200 three-stage rows, 28 primary contrasts, and secondary final RULI diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
