#!/usr/bin/env python3
"""Evaluate the frozen seed-42 Experiment 2A checkpoints with fixed-shadow RULI.

The evaluator imports the upstream RULI evaluator at runtime and directly reuses
its last-seven-token loss path.  It adds identifier-aligned, per-sample KDE
exports and the preregistered LOW-minus-PLACEBO contrast without modifying the
upstream repository or the frozen intervention manifest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import platform
import random
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RULI_ROOT = REPOSITORY_ROOT.parent / "Ruli"
DEFAULT_MANIFEST = SCRIPT_DIR / "results" / "intervention_manifest.json"
DEFAULT_EXPERIMENT_OUTPUT = (
    SCRIPT_DIR / "results" / "experiment_2a" / "seed_42"
)

SEED = 42
MODEL_NAME = "gpt2"
CONDITIONS = ("HIGH", "LOW", "PLACEBO")
EVALUATION_SPLITS = ("unlearn", "out")
EXPECTED_SPLIT_COUNT = 200
EXPECTED_S_COUNT = 28
EXPECTED_NEGATIVE_CONTROL_COUNT = 121
EXPECTED_SHADOW_COUNT = 9
FROZEN_MANIFEST_CONTENT_SHA256 = (
    "750c4cf9bc470091a05ff9e10fcf8f8cf6914f51a8a733b2e1528470ea02bf3b"
)
FROZEN_TARGET_DATASET_FINGERPRINT = "d4fe55339dd51c18"
OUTPUT_FILENAMES = (
    "per_sample_scores.csv",
    "primary_contrast.csv",
    "evaluation_summary.json",
)


def _load_training_runner() -> Any:
    path = SCRIPT_DIR / "run_experiment_2a.py"
    spec = importlib.util.spec_from_file_location("_experiment_2a_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Experiment 2A runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING_RUNNER = _load_training_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen Experiment 2A seed-42 HIGH, LOW, and PLACEBO "
            "checkpoints with exact per-sample fixed-shadow RULI scoring."
        )
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ruli-root", type=Path, default=DEFAULT_RULI_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--shadow-path", type=Path)
    parser.add_argument("--target-data-path", type=Path)
    parser.add_argument(
        "--experiment-output",
        type=Path,
        default=DEFAULT_EXPERIMENT_OUTPUT,
        help=(
            "Seed output directory containing post_npo_pre_final_ft and the "
            "three final checkpoints. Results are written below evaluation/."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--original-checkpoint",
        type=Path,
        help=(
            "Optional original pre-unlearning target-model checkpoint. When "
            "provided, exact reference efficacy scores are also computed for "
            "OUT rows and aggregate efficacy metrics. The post-NPO checkpoint "
            "is never substituted for this model."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the manifest, shadow artifact, target dataset, upstream "
            "loss path, and checkpoint structure without loading model weights "
            "or writing evaluation outputs."
        ),
    )
    return parser.parse_args()


def _integer_list(value: Any, description: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a JSON array.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{description} must contain only integer IDs.")
    return [int(item) for item in value]


def _validate_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, metadata = TRAINING_RUNNER._load_and_validate_manifest(path)
    declared_hash = manifest["manifest_hash"]["sha256"]
    if declared_hash != FROZEN_MANIFEST_CONTENT_SHA256:
        raise ValueError(
            "Frozen intervention manifest SHA-256 mismatch: expected "
            f"{FROZEN_MANIFEST_CONTENT_SHA256}, found {declared_hash}."
        )

    sets = manifest.get("sets")
    if not isinstance(sets, Mapping):
        raise ValueError("manifest.sets must be a JSON object.")
    supported_ids = _integer_list(sets.get("S_sample_ids"), "sets.S_sample_ids")
    negative_ids = _integer_list(
        sets.get("negative_control_sample_ids"),
        "sets.negative_control_sample_ids",
    )
    unlearn_ids = manifest["evaluation_ids"]["unlearn_ids"]
    if len(supported_ids) != EXPECTED_S_COUNT or len(set(supported_ids)) != 28:
        raise ValueError(
            f"Frozen supported set S must contain exactly 28 unique IDs; "
            f"found {len(supported_ids)} rows and {len(set(supported_ids))} unique."
        )
    if (
        len(negative_ids) != EXPECTED_NEGATIVE_CONTROL_COUNT
        or len(set(negative_ids)) != EXPECTED_NEGATIVE_CONTROL_COUNT
    ):
        raise ValueError(
            "Frozen negative-control set must contain exactly 121 unique IDs."
        )
    if not set(supported_ids) <= set(unlearn_ids):
        raise ValueError("Supported set S is not a subset of official UNLEARN IDs.")
    if not set(negative_ids) <= set(unlearn_ids):
        raise ValueError(
            "Negative-control set is not a subset of official UNLEARN IDs."
        )
    if set(supported_ids) & set(negative_ids):
        raise ValueError("Supported and negative-control cohorts overlap.")

    metadata = dict(metadata)
    metadata["frozen_content_sha256"] = FROZEN_MANIFEST_CONTENT_SHA256
    metadata["supported_count"] = len(supported_ids)
    metadata["negative_control_count"] = len(negative_ids)
    return manifest, metadata


def _checkpoint_paths(experiment_output: Path) -> dict[str, Path]:
    root = experiment_output.resolve()
    return {
        "post_npo_pre_final_ft": root / "post_npo_pre_final_ft",
        "HIGH": root / "HIGH_final",
        "LOW": root / "LOW_final",
        "PLACEBO": root / "PLACEBO_final",
    }


def _checkpoint_metadata(path: Path, name: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} checkpoint directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{name} checkpoint has no config.json: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} checkpoint has invalid config.json: {path}") from exc
    if config.get("model_type") != "gpt2":
        raise ValueError(
            f"{name} checkpoint model_type is {config.get('model_type')!r}, not 'gpt2'."
        )
    weight_files = sorted(
        file
        for pattern in (
            "model*.safetensors",
            "pytorch_model*.bin",
        )
        for file in path.glob(pattern)
        if file.is_file()
    )
    if not weight_files:
        raise FileNotFoundError(f"{name} checkpoint has no model weight files: {path}")
    return {
        "path": str(path),
        "config_sha256": TRAINING_RUNNER._sha256_file(config_path),
        "weight_files": [
            {"name": file.name, "bytes": file.stat().st_size}
            for file in weight_files
        ],
    }


def _normalize_shadow_mapping(
    value: Any, name: str
) -> dict[int, Sequence[float]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Shadow artifact field {name!r} is not a mapping.")
    normalized: dict[int, Sequence[float]] = {}
    for raw_key, observations in value.items():
        if isinstance(raw_key, bool):
            raise ValueError(f"Invalid boolean sample ID in shadow field {name!r}.")
        try:
            sample_id = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid sample ID {raw_key!r} in shadow field {name!r}."
            ) from exc
        if sample_id in normalized:
            raise ValueError(f"Duplicate normalized shadow sample ID {sample_id}.")
        normalized[sample_id] = observations
    return normalized


def _plain_shadow_observations(
    values: Any, field: str, sample_id: int
) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field}[{sample_id}] is not a numeric sequence.")
    observations = [float(value) for value in values]
    if len(observations) != EXPECTED_SHADOW_COUNT:
        raise ValueError(
            f"{field}[{sample_id}] contains {len(observations)} observations, "
            f"not the fixed {EXPECTED_SHADOW_COUNT}."
        )
    if not all(math.isfinite(value) for value in observations):
        raise ValueError(f"{field}[{sample_id}] contains NaN or Inf observations.")
    return observations


def _load_and_validate_shadow(
    shadow_path: Path,
    expected_artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    torch: Any,
) -> tuple[dict[str, dict[int, list[float]]], dict[str, Any]]:
    metadata = TRAINING_RUNNER._verify_file_artifact(
        shadow_path,
        expected_artifact,
        "Frozen 9-shadow artifact",
    )
    raw = torch.load(shadow_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("Shadow artifact root is not a mapping.")
    required_fields = (
        "in_original",
        "unlearn_unlearned",
        "out_unlearned",
        "out_original",
    )
    mappings = {
        field: _normalize_shadow_mapping(raw.get(field), field)
        for field in required_fields
    }
    actual_ids = sorted(mappings["in_original"])
    expected_ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if actual_ids != expected_ids:
        raise ValueError(
            "Shadow artifact IDs do not exactly match the frozen manifest order."
        )
    in_ids = actual_ids[:200]
    unlearn_ids = actual_ids[200:400]
    out_ids = actual_ids[400:600]
    if not (
        in_ids == manifest["evaluation_ids"]["in_ids"]
        and unlearn_ids == manifest["evaluation_ids"]["unlearn_ids"]
        and out_ids == manifest["evaluation_ids"]["out_ids"]
    ):
        raise ValueError(
            "First-200 IN, next-200 UNLEARN, and next-200 OUT partitions "
            "do not match the frozen manifest."
        )
    evaluation_ids = unlearn_ids + out_ids
    normalized: dict[str, dict[int, list[float]]] = {}
    for field, mapping in mappings.items():
        missing = [
            sample_id for sample_id in evaluation_ids if sample_id not in mapping
        ]
        if missing:
            raise ValueError(
                f"Shadow field {field!r} cannot align all evaluation IDs; "
                f"missing {missing[:10]}."
            )
        normalized[field] = {
            sample_id: _plain_shadow_observations(
                mapping[sample_id], field, sample_id
            )
            for sample_id in evaluation_ids
        }
    metadata = dict(metadata)
    metadata.update(
        {
            "shadow_count": EXPECTED_SHADOW_COUNT,
            "ordered_target_id_count": len(actual_ids),
            "partition_rule": (
                "sorted(shadow_results['in_original'].keys()); first 200 IN, "
                "next 200 UNLEARN, next 200 OUT"
            ),
        }
    )
    return normalized, metadata


def _token_ids(row: Mapping[str, Any], sample_id: int) -> list[int]:
    values = row.get("input_ids")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Target row {sample_id} has no integer input_ids sequence.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"Target row {sample_id} input_ids are not all integers.")
    result = [int(value) for value in values]
    if len(result) < 2:
        raise ValueError(
            f"Target row {sample_id} has fewer than two tokens; upstream RULI "
            "would silently drop it."
        )
    if len(result) < 8:
        raise ValueError(
            f"Target row {sample_id} has only {len(result)} tokens; the frozen "
            "evaluation requires seven valid next-token prediction positions."
        )
    return result


def _validate_target_dataset(
    target_data_path: Path,
    expected_artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    tokenizer: Any,
    load_from_disk: Callable[[str], Any],
) -> tuple[Any, dict[str, Any], dict[int, list[int]]]:
    metadata = TRAINING_RUNNER._verify_target_dataset_storage(
        target_data_path, expected_artifact
    )
    dataset = load_from_disk(str(target_data_path.resolve()))
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint != FROZEN_TARGET_DATASET_FINGERPRINT:
        raise ValueError(
            "Target dataset fingerprint mismatch: expected "
            f"{FROZEN_TARGET_DATASET_FINGERPRINT}, found {fingerprint}."
        )
    evaluation_ids = (
        manifest["evaluation_ids"]["unlearn_ids"]
        + manifest["evaluation_ids"]["out_ids"]
    )
    if min(evaluation_ids) < 0 or max(evaluation_ids) >= len(dataset):
        raise ValueError("An official evaluation sample ID is outside the dataset.")

    tokens: dict[int, list[int]] = {}
    for sample_id in evaluation_ids:
        row = dataset[sample_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"Target row {sample_id} is not a mapping.")
        for identifier_field in ("sample_id", "row_id"):
            if identifier_field in row and int(row[identifier_field]) != sample_id:
                raise ValueError(
                    f"Target row {sample_id} has mismatched {identifier_field}="
                    f"{row[identifier_field]!r}."
                )
        tokens[sample_id] = _token_ids(row, sample_id)

    supported_records = manifest.get("samples", {}).get("S")
    if not isinstance(supported_records, list) or len(supported_records) != 28:
        raise ValueError("Manifest samples.S does not contain exactly 28 rows.")
    expected_supported_order = manifest["sets"]["S_sample_ids"]
    actual_supported_order: list[int] = []
    for record in supported_records:
        if not isinstance(record, Mapping):
            raise ValueError("A manifest samples.S record is not an object.")
        sample_id = int(record.get("sample_id"))
        actual_supported_order.append(sample_id)
        if int(record.get("original_dataset_index")) != sample_id:
            raise ValueError(f"Supported sample {sample_id} dataset index mismatch.")
        if int(record.get("gpt2_token_count")) != len(tokens[sample_id]):
            raise ValueError(f"Supported sample {sample_id} token count mismatch.")
        decoded = tokenizer.decode(
            tokens[sample_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        text_hash = hashlib.sha256(decoded.encode("utf-8")).hexdigest()
        if text_hash != record.get("text_sha256"):
            raise ValueError(
                f"Supported sample {sample_id} decoded text hash mismatch."
            )
    if actual_supported_order != expected_supported_order:
        raise ValueError("Manifest samples.S order differs from sets.S_sample_ids.")

    metadata = dict(metadata)
    metadata.update(
        {
            "fingerprint": fingerprint,
            "row_count": len(dataset),
            "evaluation_token_sequences_validated": len(tokens),
            "supported_text_hashes_validated": len(supported_records),
        }
    )
    return dataset, metadata, tokens


def _function_ast(function: Callable[..., Any]) -> str:
    source = textwrap.dedent(inspect.getsource(function))
    return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)


def _validate_reference_loss_path(ruli_utils: Any, torch: Any) -> dict[str, Any]:
    mia_method = ruli_utils.MIAEvaluator._batch_inference
    efficacy_method = ruli_utils.EfficacyEvaluator._batch_inference
    if _function_ast(mia_method) != _function_ast(efficacy_method):
        raise ValueError(
            "MIAEvaluator and EfficacyEvaluator no longer share the identical "
            "reference text-loss implementation."
        )

    class _Output:
        def __init__(self, logits: Any):
            self.logits = logits

    class _DeterministicModel:
        def eval(self) -> Any:
            return self

        def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
            del attention_mask
            sequence_length = input_ids.shape[1]
            logits = torch.arange(
                sequence_length * 16,
                dtype=torch.float32,
                device=input_ids.device,
            ).reshape(1, sequence_length, 16)
            return _Output(logits)

    token_ids = list(range(10))
    evaluator = ruli_utils.MIAEvaluator(
        target_model=_DeterministicModel(),
        unlearned_model=_DeterministicModel(),
        target_dataset=None,
        tokenizer=None,
        device=torch.device("cpu"),
        args=SimpleNamespace(per_sample_output=None),
    )
    actual = mia_method(evaluator, evaluator.unlearned_model, [token_ids])
    logits = torch.arange(10 * 16, dtype=torch.float32).reshape(10, 16)
    expected = torch.nn.functional.cross_entropy(
        logits[:-1][torch.arange(2, 9)],
        torch.tensor(token_ids[1:])[torch.arange(2, 9)],
        reduction="mean",
    ).item()
    if len(actual) != 1 or not math.isclose(
        float(actual[0]), float(expected), rel_tol=0.0, abs_tol=1e-7
    ):
        raise ValueError(
            "Upstream RULI text loss does not equal mean next-token cross entropy "
            "over the final seven valid prediction positions."
        )
    source = textwrap.dedent(inspect.getsource(mia_method))
    return {
        "reused_function": "Ruli/text/utils.py:MIAEvaluator._batch_inference",
        "efficacy_function_ast_identical": True,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "behavioral_check": "passed",
        "definition": (
            "mean next-token cross entropy over min(7, sequence_length - 1) "
            "final valid prediction positions; exactly 7 for every frozen "
            "evaluation row"
        ),
    }


def _build_kde_references(
    shadow: Mapping[str, Mapping[int, Sequence[float]]],
    evaluation_ids: Sequence[int],
    gaussian_kde: Callable[..., Any],
) -> dict[str, dict[int, tuple[Any, Any]]]:
    definitions = {
        "privacy": ("unlearn_unlearned", "out_unlearned"),
        "efficacy": ("unlearn_unlearned", "out_original"),
    }
    result: dict[str, dict[int, tuple[Any, Any]]] = {}
    for outcome, (positive, negative) in definitions.items():
        result[outcome] = {}
        for sample_id in evaluation_ids:
            try:
                result[outcome][sample_id] = (
                    gaussian_kde(shadow[positive][sample_id]),
                    gaussian_kde(shadow[negative][sample_id]),
                )
            except Exception as exc:
                raise ValueError(
                    f"Could not fit reference {outcome} KDEs for sample "
                    f"{sample_id}: {exc}"
                ) from exc
    return result


def _score_kde(
    loss: float,
    kdes: tuple[Any, Any],
    outcome: str,
    condition: str,
    sample_id: int,
) -> tuple[float, float]:
    positive_kde, negative_kde = kdes
    positive_density = float(positive_kde.evaluate([loss])[0])
    negative_density = float(negative_kde.evaluate([loss])[0])
    positive_logpdf = float(positive_kde.logpdf([loss])[0])
    negative_logpdf = float(negative_kde.logpdf([loss])[0])
    bounded_score = positive_density / (
        positive_density + negative_density + 1e-12
    )
    log_odds = positive_logpdf - negative_logpdf
    values = {
        "loss": float(loss),
        "positive_density": positive_density,
        "negative_density": negative_density,
        "positive_logpdf": positive_logpdf,
        "negative_logpdf": negative_logpdf,
        "bounded_score": bounded_score,
        "log_odds": log_odds,
    }
    nonfinite = [name for name, value in values.items() if not math.isfinite(value)]
    if nonfinite:
        raise FloatingPointError(
            f"NaN/Inf {outcome} KDE result for condition={condition}, "
            f"sample_id={sample_id}: nonfinite fields={nonfinite}, values={values}."
        )
    if not 0.0 <= bounded_score <= 1.0:
        raise FloatingPointError(
            f"Out-of-range bounded {outcome} score for condition={condition}, "
            f"sample_id={sample_id}: {bounded_score}."
        )
    return log_odds, bounded_score


def _aggregate_from_rows(
    rows: Sequence[Mapping[str, Any]], score_field: str, ruli_utils: Any
) -> dict[str, float | int]:
    labels = [1 if row["split"] == "unlearn" else 0 for row in rows]
    scores = [float(row[score_field]) for row in rows]
    fpr, tpr, _ = ruli_utils.roc_curve(labels, scores)
    return {
        "AUC": float(ruli_utils.auc(fpr, tpr)),
        "ACC": float(
            ruli_utils.accuracy_score(labels, [score > 0.5 for score in scores])
        ),
        "TPR@1%FPR": float(
            tpr[ruli_utils.np.searchsorted(fpr, 0.01, side="right") - 1]
            if ruli_utils.np.any(fpr <= 0.01)
            else 0.0
        ),
        "TPR@5%FPR": float(
            tpr[ruli_utils.np.searchsorted(fpr, 0.05, side="right") - 1]
            if ruli_utils.np.any(fpr <= 0.05)
            else 0.0
        ),
        "Total": len(labels),
    }


def _assert_metrics_match(
    reference: Mapping[str, Any],
    per_sample: Mapping[str, Any],
    condition: str,
    outcome: str,
) -> None:
    for field in ("AUC", "ACC", "TPR@1%FPR", "TPR@5%FPR"):
        if not math.isclose(
            float(reference[field]),
            float(per_sample[field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Per-sample {condition} {outcome} scores do not reproduce "
                f"upstream {field}: per-sample={per_sample[field]}, "
                f"upstream={reference[field]}."
            )
    if int(reference["Total"]) != int(per_sample["Total"]):
        raise ValueError(
            f"Per-sample {condition} {outcome} Total does not reproduce upstream."
        )


def _run_reference_inference(
    model: Any,
    dataset: Any,
    tokenizer: Any,
    device: Any,
    ruli_utils: Any,
    sample_ids: Sequence[int],
    expected_tokens: Mapping[int, Sequence[int]],
) -> list[float]:
    token_lists = [expected_tokens[sample_id] for sample_id in sample_ids]
    evaluator = ruli_utils.MIAEvaluator(
        target_model=model,
        unlearned_model=model,
        target_dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        args=SimpleNamespace(per_sample_output=None),
    )
    losses = evaluator._batch_inference(model, token_lists)
    if len(losses) != len(sample_ids):
        raise ValueError(
            "Upstream RULI inference reordered or dropped samples: requested "
            f"{len(sample_ids)}, returned {len(losses)}."
        )
    plain = [float(loss) for loss in losses]
    if not all(math.isfinite(loss) for loss in plain):
        raise FloatingPointError("Upstream RULI inference produced NaN/Inf losses.")
    return plain


def _evaluate_condition(
    condition: str,
    model: Any,
    dataset: Any,
    tokenizer: Any,
    device: Any,
    ruli_utils: Any,
    shadow: Mapping[str, Mapping[int, Sequence[float]]],
    kdes: Mapping[str, Mapping[int, tuple[Any, Any]]],
    unlearn_ids: Sequence[int],
    out_ids: Sequence[int],
    tokens: Mapping[int, Sequence[int]],
    supported_ids: set[int],
    negative_ids: set[int],
    original_out_losses: Mapping[int, float] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_ids = list(unlearn_ids) + list(out_ids)
    losses = _run_reference_inference(
        model,
        dataset,
        tokenizer,
        device,
        ruli_utils,
        ordered_ids,
        tokens,
    )
    loss_by_id = dict(zip(ordered_ids, losses, strict=True))
    rows: list[dict[str, Any]] = []
    for split, ids in (("unlearn", unlearn_ids), ("out", out_ids)):
        for sample_id in ids:
            observed_loss = loss_by_id[sample_id]
            privacy_log_odds, privacy_score = _score_kde(
                observed_loss,
                kdes["privacy"][sample_id],
                "privacy",
                condition,
                sample_id,
            )
            efficacy_loss = (
                observed_loss
                if split == "unlearn"
                else (
                    None
                    if original_out_losses is None
                    else original_out_losses[sample_id]
                )
            )
            efficacy_log_odds: float | None = None
            efficacy_score: float | None = None
            if efficacy_loss is not None:
                efficacy_log_odds, efficacy_score = _score_kde(
                    efficacy_loss,
                    kdes["efficacy"][sample_id],
                    "efficacy",
                    condition,
                    sample_id,
                )
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "is_supported_S": int(sample_id in supported_ids),
                    "is_negative_control": int(sample_id in negative_ids),
                    "condition": condition,
                    "observed_loss": observed_loss,
                    "privacy_log_odds": privacy_log_odds,
                    "privacy_score": privacy_score,
                    "efficacy_observed_loss": efficacy_loss,
                    "efficacy_log_odds": efficacy_log_odds,
                    "efficacy_score": efficacy_score,
                }
            )

    privacy_reference = ruli_utils.MIAEvaluator(
        target_model=model,
        unlearned_model=model,
        target_dataset=dataset,
        tokenizer=tokenizer,
        device=device,
        args=SimpleNamespace(per_sample_output=None),
    ).evaluate_with_kde(
        unlearn_losses=[loss_by_id[sample_id] for sample_id in unlearn_ids],
        out_losses=[loss_by_id[sample_id] for sample_id in out_ids],
        unlearn_ids=list(unlearn_ids),
        out_ids=list(out_ids),
        shadow_in=shadow["unlearn_unlearned"],
        shadow_out=shadow["out_unlearned"],
    )
    privacy_metrics = _aggregate_from_rows(rows, "privacy_score", ruli_utils)
    _assert_metrics_match(
        privacy_reference, privacy_metrics, condition, "privacy"
    )
    metrics: dict[str, Any] = {"privacy": privacy_metrics, "efficacy": None}

    if original_out_losses is not None:
        efficacy_reference = ruli_utils.MIAEvaluator(
            target_model=model,
            unlearned_model=model,
            target_dataset=dataset,
            tokenizer=tokenizer,
            device=device,
            args=SimpleNamespace(per_sample_output=None),
        ).evaluate_with_kde(
            unlearn_losses=[loss_by_id[sample_id] for sample_id in unlearn_ids],
            out_losses=[original_out_losses[sample_id] for sample_id in out_ids],
            unlearn_ids=list(unlearn_ids),
            out_ids=list(out_ids),
            shadow_in=shadow["unlearn_unlearned"],
            shadow_out=shadow["out_original"],
        )
        efficacy_metrics = _aggregate_from_rows(rows, "efficacy_score", ruli_utils)
        _assert_metrics_match(
            efficacy_reference, efficacy_metrics, condition, "efficacy"
        )
        metrics["efficacy"] = efficacy_metrics
    return rows, metrics


def _validate_condition_row_alignment(
    rows: Sequence[Mapping[str, Any]],
    unlearn_ids: Sequence[int],
    out_ids: Sequence[int],
) -> None:
    expected = [
        (condition, split, sample_id)
        for condition in CONDITIONS
        for split, ids in (("unlearn", unlearn_ids), ("out", out_ids))
        for sample_id in ids
    ]
    actual = [
        (str(row["condition"]), str(row["split"]), int(row["sample_id"]))
        for row in rows
    ]
    if actual != expected:
        raise ValueError(
            "Condition evaluation sample IDs were reordered, dropped, or duplicated."
        )


def _contrast_rows(
    rows: Sequence[Mapping[str, Any]], supported_ids: Sequence[int]
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["condition"]), int(row["sample_id"])): float(
            row["privacy_log_odds"]
        )
        for row in rows
        if row["split"] == "unlearn"
    }
    result: list[dict[str, Any]] = []
    for sample_id in supported_ids:
        high = by_key[("HIGH", sample_id)]
        low = by_key[("LOW", sample_id)]
        placebo = by_key[("PLACEBO", sample_id)]
        result.append(
            {
                "sample_id": sample_id,
                "privacy_log_odds_HIGH": high,
                "privacy_log_odds_LOW": low,
                "privacy_log_odds_PLACEBO": placebo,
                "LOW_minus_PLACEBO": low - placebo,
            }
        )
    if len(result) != EXPECTED_S_COUNT:
        raise ValueError("Primary contrast does not contain exactly 28 rows.")
    return result


def _median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty cohort.")
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _cohort_summary(
    rows: Sequence[Mapping[str, Any]], sample_ids: Sequence[int]
) -> dict[str, Any]:
    ordered_ids = list(sample_ids)
    if len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("A summary cohort contains duplicate IDs.")
    by_key = {
        (str(row["condition"]), int(row["sample_id"])): float(
            row["privacy_log_odds"]
        )
        for row in rows
        if row["split"] == "unlearn" and int(row["sample_id"]) in set(ordered_ids)
    }
    expected_keys = {
        (condition, sample_id)
        for condition in CONDITIONS
        for sample_id in ordered_ids
    }
    if set(by_key) != expected_keys:
        raise ValueError("Cohort summary cannot align every condition and sample ID.")
    condition_stats: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        values = [by_key[(condition, sample_id)] for sample_id in ordered_ids]
        condition_stats[condition] = {
            "mean_privacy_log_odds": sum(values) / len(values),
            "median_privacy_log_odds": _median(values),
        }
    differences = [
        by_key[("LOW", sample_id)] - by_key[("PLACEBO", sample_id)]
        for sample_id in ordered_ids
    ]
    negative_count = sum(value < 0.0 for value in differences)
    return {
        "sample_count": len(ordered_ids),
        "condition_privacy_log_odds": condition_stats,
        "LOW_minus_PLACEBO": {
            "mean": sum(differences) / len(differences),
            "median": _median(differences),
            "count_below_zero": negative_count,
            "fraction_below_zero": negative_count / len(differences),
        },
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in (
        "datasets",
        "numpy",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_outputs(
    output_dir: Path,
    per_sample_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {name: output_dir / name for name in OUTPUT_FILENAMES}
    occupied = [path for path in final_paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs: "
            + ", ".join(str(path) for path in occupied)
        )
    temporary = {
        name: path.with_name(path.name + ".tmp")
        for name, path in final_paths.items()
    }
    try:
        _write_csv(temporary["per_sample_scores.csv"], per_sample_rows)
        _write_csv(temporary["primary_contrast.csv"], contrast_rows)
        temporary["evaluation_summary.json"].write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for name in OUTPUT_FILENAMES:
            temporary[name].replace(final_paths[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()


def _reset_determinism(torch: Any) -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()
    if args.seed != SEED:
        raise ValueError("Experiment 2A evaluation supports only frozen seed 42.")

    manifest, manifest_metadata = _validate_manifest(args.manifest)
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
    checkpoints = _checkpoint_paths(args.experiment_output)
    checkpoint_metadata = {
        name: _checkpoint_metadata(path, name)
        for name, path in checkpoints.items()
    }
    original_metadata = None
    if args.original_checkpoint is not None:
        original_metadata = _checkpoint_metadata(
            args.original_checkpoint, "original pre-unlearning"
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
    shadow, shadow_metadata = _load_and_validate_shadow(
        shadow_path,
        input_artifacts["shadow_artifact"],
        manifest,
        torch,
    )
    ruli_utils = TRAINING_RUNNER._load_ruli_utils(ruli_text_dir)
    loss_metadata = _validate_reference_loss_path(ruli_utils, torch)
    tokenizer = AutoTokenizer.from_pretrained(checkpoints["post_npo_pre_final_ft"])
    target_dataset, target_metadata, tokens = _validate_target_dataset(
        target_data_path,
        input_artifacts["target_dataset"],
        manifest,
        tokenizer,
        load_from_disk,
    )

    print(
        "[VERIFY] Frozen manifest, shadow artifact, target partitions, target "
        "rows, upstream loss path, and all four seed-42 checkpoints passed."
    )
    if args.validate_only:
        print(
            "[VERIFY] Validation-only mode loaded no model weights and wrote no "
            "files."
        )
        return

    output_dir = args.experiment_output.resolve() / "evaluation"
    occupied = [
        output_dir / name
        for name in OUTPUT_FILENAMES
        if (output_dir / name).exists()
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite existing evaluation outputs: "
            + ", ".join(str(path) for path in occupied)
        )

    _reset_determinism(torch)
    device = torch.device(args.device)
    unlearn_ids = manifest["evaluation_ids"]["unlearn_ids"]
    out_ids = manifest["evaluation_ids"]["out_ids"]
    evaluation_ids = unlearn_ids + out_ids
    supported_order = manifest["sets"]["S_sample_ids"]
    negative_order = manifest["sets"]["negative_control_sample_ids"]
    supported_ids = set(supported_order)
    negative_ids = set(negative_order)
    kdes = _build_kde_references(
        shadow,
        evaluation_ids,
        ruli_utils.gaussian_kde,
    )

    original_out_losses: dict[int, float] | None = None
    if args.original_checkpoint is not None:
        _reset_determinism(torch)
        original_model = AutoModelForCausalLM.from_pretrained(
            args.original_checkpoint.resolve()
        ).to(device)
        losses = _run_reference_inference(
            original_model,
            target_dataset,
            tokenizer,
            device,
            ruli_utils,
            out_ids,
            tokens,
        )
        original_out_losses = dict(zip(out_ids, losses, strict=True))
        del original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_sample_rows: list[dict[str, Any]] = []
    aggregate_metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        print(f"[INFO] Evaluating {condition} with upstream RULI text loss.")
        _reset_determinism(torch)
        model = AutoModelForCausalLM.from_pretrained(checkpoints[condition]).to(device)
        condition_rows, condition_metrics = _evaluate_condition(
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
            original_out_losses,
        )
        per_sample_rows.extend(condition_rows)
        aggregate_metrics[condition] = condition_metrics
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _validate_condition_row_alignment(per_sample_rows, unlearn_ids, out_ids)
    contrast = _contrast_rows(per_sample_rows, supported_order)
    primary_results = {
        "supported_S": _cohort_summary(per_sample_rows, supported_order),
        "all_UNLEARN": _cohort_summary(per_sample_rows, unlearn_ids),
        "negative_controls": _cohort_summary(per_sample_rows, negative_order),
    }
    efficacy_scope = (
        "UNLEARN and OUT; OUT losses from supplied original pre-unlearning checkpoint"
        if original_out_losses is not None
        else (
            "UNLEARN only; OUT fields intentionally blank because the reference "
            "requires an original pre-unlearning checkpoint"
        )
    )
    deviations = (
        []
        if original_out_losses is not None
        else [
            (
                "OUT efficacy scores and aggregate efficacy metrics are omitted: "
                "the Experiment 2A trainer did not save the original pre-unlearning "
                "model, and post_npo_pre_final_ft is not a valid substitute. "
                "All UNLEARN efficacy scores follow the reference path exactly."
            )
        ]
    )
    summary = {
        "schema_version": 1,
        "experiment": "2A",
        "seed": SEED,
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
        "original_checkpoint": original_metadata,
        "sample_counts": {
            "IN_partition": EXPECTED_SPLIT_COUNT,
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
            "efficacy_scope": efficacy_scope,
        },
        "primary_descriptive_results": primary_results,
        "aggregate_reference_metrics": aggregate_metrics,
        "package_versions": _package_versions(),
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
            "condition_sample_order_and_identity": "passed",
            "upstream_loss_behavior": "passed",
            "per_sample_bounded_scores_reproduce_upstream_metrics": "passed",
            "nonfinite_losses_or_kde_scores": 0,
        },
        "deviations_from_reference_behavior": deviations,
    }
    _write_outputs(output_dir, per_sample_rows, contrast, summary)
    print(f"[VERIFY] Wrote {len(per_sample_rows)} aligned per-sample rows.")
    print(f"[INFO] Evaluation outputs: {output_dir}")


if __name__ == "__main__":
    main()
