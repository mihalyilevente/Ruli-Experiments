"""Convert direct official-RULI sample capture into an analysis CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve


RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRIC_KEYS = ("AUC", "ACC", "TPR@1%FPR", "TPR@5%FPR", "Total")
CSV_FIELDS = (
    "sample_id",
    "text",
    "split",
    "privacy_observed_loss",
    "efficacy_observed_loss",
    "privacy_score",
    "efficacy_score",
    "privacy_label",
    "efficacy_label",
    "privacy_positive_shadow_condition",
    "privacy_negative_shadow_condition",
    "privacy_positive_shadow_distribution",
    "privacy_negative_shadow_distribution",
    "efficacy_positive_shadow_condition",
    "efficacy_negative_shadow_condition",
    "efficacy_positive_shadow_distribution",
    "efficacy_negative_shadow_distribution",
)


@dataclass(frozen=True)
class CapturedSample:
    sample_id: int
    observed_loss: float
    label: int
    likelihood_ratio: float
    positive_shadow_distribution: tuple[float, ...]
    negative_shadow_distribution: tuple[float, ...]


@dataclass(frozen=True)
class CapturedAttack:
    name: str
    positive_shadow_condition: str
    negative_shadow_condition: str
    metrics: dict[str, float | int]
    samples: tuple[CapturedSample, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-sample values captured directly during official RULI "
            "mia_inference.py execution. No model inference is performed."
        )
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        default=RESULTS_DIR / "official_ruli_samples.pth",
        help="File produced by mia_inference.py --sample_export_path.",
    )
    parser.add_argument("--target-data-path", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Tokenizer used only to decode target-dataset text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "ruli_scores.csv",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Verification JSON path (defaults to <output stem>.metrics.json).",
    )
    parser.add_argument("--expected-privacy-auc", type=float, default=0.8531)
    parser.add_argument("--expected-efficacy-auc", type=float, default=0.8589)
    parser.add_argument(
        "--verify-expected-aucs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require exported AUCs to match the expected 9-shadow values at "
            "four decimal places (default: enabled)."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {description}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {description}: {value!r}")
    return result


def _distribution(value: Any, description: str) -> tuple[float, ...]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected a sequence for {description}.")
    result = tuple(
        _finite_float(item, f"{description} value")
        for item in value
    )
    if not result:
        raise ValueError(f"Empty {description}.")
    return result


def _normalise_metrics(value: Any, attack_name: str) -> dict[str, float | int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Missing metric mapping for {attack_name}.")
    missing = set(METRIC_KEYS).difference(value)
    if missing:
        raise ValueError(
            f"{attack_name} metrics are missing: {', '.join(sorted(missing))}"
        )
    metrics: dict[str, float | int] = {}
    for key in METRIC_KEYS:
        if key == "Total":
            metrics[key] = int(value[key])
        else:
            metrics[key] = _finite_float(value[key], f"{attack_name} {key}")
    return metrics


def _normalise_samples(value: Any, attack_name: str) -> tuple[CapturedSample, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Missing sample sequence for {attack_name}.")
    samples: list[CapturedSample] = []
    seen_ids: set[int] = set()
    required = {
        "sample_id",
        "observed_loss",
        "label",
        "likelihood_ratio",
        "positive_shadow_distribution",
        "negative_shadow_distribution",
    }
    for index, raw_sample in enumerate(value):
        if not isinstance(raw_sample, Mapping):
            raise ValueError(f"{attack_name} sample {index} is not a mapping.")
        missing = required.difference(raw_sample)
        if missing:
            raise ValueError(
                f"{attack_name} sample {index} is missing: "
                f"{', '.join(sorted(missing))}"
            )
        sample_id = int(raw_sample["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate {attack_name} sample_id: {sample_id}")
        seen_ids.add(sample_id)
        label = int(raw_sample["label"])
        if label not in {0, 1}:
            raise ValueError(
                f"{attack_name} sample {sample_id} has invalid label {label}."
            )
        likelihood_ratio = _finite_float(
            raw_sample["likelihood_ratio"], "likelihood ratio"
        )
        if not 0.0 <= likelihood_ratio <= 1.0:
            raise ValueError(
                f"{attack_name} sample {sample_id} has likelihood ratio "
                f"{likelihood_ratio}, outside [0, 1]."
            )
        samples.append(
            CapturedSample(
                sample_id=sample_id,
                observed_loss=_finite_float(
                    raw_sample["observed_loss"], "observed loss"
                ),
                label=label,
                likelihood_ratio=likelihood_ratio,
                positive_shadow_distribution=_distribution(
                    raw_sample["positive_shadow_distribution"],
                    "positive shadow distribution",
                ),
                negative_shadow_distribution=_distribution(
                    raw_sample["negative_shadow_distribution"],
                    "negative shadow distribution",
                ),
            )
        )
    if not samples:
        raise ValueError(f"No captured samples for {attack_name}.")
    return tuple(samples)


def _load_capture(path: Path) -> tuple[CapturedAttack, CapturedAttack]:
    if not path.is_file():
        raise FileNotFoundError(f"Official RULI capture does not exist: {path}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or int(raw.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported or missing official capture schema version.")

    expected_conditions = {
        "privacy": ("unlearn_unlearned", "out_unlearned"),
        "efficacy": ("unlearn_unlearned", "out_original"),
    }
    attacks: list[CapturedAttack] = []
    for attack_name in ("privacy", "efficacy"):
        section = raw.get(attack_name)
        if not isinstance(section, Mapping):
            raise ValueError(f"Capture is missing the {attack_name} section.")
        positive_condition = str(section.get("positive_shadow_condition", ""))
        negative_condition = str(section.get("negative_shadow_condition", ""))
        if not positive_condition or not negative_condition:
            raise ValueError(f"Capture lacks condition names for {attack_name}.")
        if (positive_condition, negative_condition) != expected_conditions[attack_name]:
            raise ValueError(
                f"Unexpected {attack_name} shadow conditions: "
                f"{positive_condition!r} versus {negative_condition!r}."
            )
        attacks.append(
            CapturedAttack(
                name=attack_name,
                positive_shadow_condition=positive_condition,
                negative_shadow_condition=negative_condition,
                metrics=_normalise_metrics(section.get("metrics"), attack_name),
                samples=_normalise_samples(section.get("samples"), attack_name),
            )
        )
    return attacks[0], attacks[1]


def _metrics_from_arrays(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    if len(np.unique(labels)) != 2:
        raise ValueError("Both UNLEARN and OUT labels are required for metrics.")
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    auc_score = float(auc(false_positive_rate, true_positive_rate))
    tpr_at_1 = (
        true_positive_rate[
            np.searchsorted(false_positive_rate, 0.01, side="right") - 1
        ]
        if np.any(false_positive_rate <= 0.01)
        else 0.0
    )
    tpr_at_5 = (
        true_positive_rate[
            np.searchsorted(false_positive_rate, 0.05, side="right") - 1
        ]
        if np.any(false_positive_rate <= 0.05)
        else 0.0
    )
    return {
        "AUC": auc_score,
        "ACC": float(accuracy_score(labels, scores > 0.5)),
        "TPR@1%FPR": float(tpr_at_1),
        "TPR@5%FPR": float(tpr_at_5),
        "Total": int(len(labels)),
    }


def _metrics_from_samples(
    samples: Sequence[CapturedSample],
) -> dict[str, float | int]:
    labels = np.asarray([sample.label for sample in samples], dtype=np.int64)
    scores = np.asarray(
        [sample.likelihood_ratio for sample in samples], dtype=np.float64
    )
    return _metrics_from_arrays(labels, scores)


def _metrics_from_csv(path: Path, attack_name: str) -> dict[str, float | int]:
    label_field = f"{attack_name}_label"
    score_field = f"{attack_name}_score"
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = {label_field, score_field}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Exported CSV is missing metric fields: {', '.join(sorted(missing))}"
            )
        for row in reader:
            labels.append(int(row[label_field]))
            scores.append(_finite_float(row[score_field], score_field))
    return _metrics_from_arrays(
        np.asarray(labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
    )


def _verify_matches_official(
    attack: CapturedAttack, reconstructed: Mapping[str, float | int]
) -> None:
    for key in METRIC_KEYS:
        official_value = attack.metrics[key]
        reconstructed_value = reconstructed[key]
        if key == "Total":
            matches = int(official_value) == int(reconstructed_value)
        else:
            matches = math.isclose(
                float(official_value),
                float(reconstructed_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        if not matches:
            raise ValueError(
                f"Exported {attack.name} samples do not reproduce official {key}: "
                f"official={official_value}, reconstructed={reconstructed_value}."
            )


def _verify_expected_auc(
    attack_name: str, actual_auc: float, expected_auc: float
) -> None:
    actual_text = f"{actual_auc:.4f}"
    expected_text = f"{expected_auc:.4f}"
    if actual_text != expected_text:
        raise ValueError(
            f"Expected {attack_name} AUC {expected_text}, captured {actual_text}. "
            "Use the official 9-shadow run or disable the expected-AUC check for "
            "a non-reference smoke run."
        )


def _verify_reference_shape(attack: CapturedAttack) -> None:
    label_counts = {
        label: sum(sample.label == label for sample in attack.samples)
        for label in (0, 1)
    }
    if label_counts != {0: 200, 1: 200}:
        raise ValueError(
            f"Expected 200 UNLEARN and 200 OUT {attack.name} samples; "
            f"captured labels are {label_counts}."
        )


def _samples_by_id(attack: CapturedAttack) -> dict[int, CapturedSample]:
    return {sample.sample_id: sample for sample in attack.samples}


def _json_distribution(values: Sequence[float]) -> str:
    return json.dumps(list(values), separators=(",", ":"), allow_nan=False)


def _build_rows(
    privacy: CapturedAttack,
    efficacy: CapturedAttack,
    target_dataset: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    efficacy_by_id = _samples_by_id(efficacy)
    privacy_ids = {sample.sample_id for sample in privacy.samples}
    if privacy_ids != set(efficacy_by_id):
        raise ValueError("Privacy and efficacy captures contain different sample IDs.")

    rows: list[dict[str, Any]] = []
    for privacy_sample in privacy.samples:
        efficacy_sample = efficacy_by_id[privacy_sample.sample_id]
        if privacy_sample.label != efficacy_sample.label:
            raise ValueError(
                f"Privacy/efficacy label mismatch for sample "
                f"{privacy_sample.sample_id}."
            )
        sample_id = privacy_sample.sample_id
        if sample_id < 0 or sample_id >= len(target_dataset):
            raise IndexError(
                f"Sample ID {sample_id} is outside target dataset size "
                f"{len(target_dataset)}."
            )
        input_ids = target_dataset[sample_id]["input_ids"]
        rows.append(
            {
                "sample_id": sample_id,
                "text": tokenizer.decode(input_ids, skip_special_tokens=True),
                "split": "unlearn" if privacy_sample.label == 1 else "out",
                "privacy_observed_loss": privacy_sample.observed_loss,
                "efficacy_observed_loss": efficacy_sample.observed_loss,
                "privacy_score": privacy_sample.likelihood_ratio,
                "efficacy_score": efficacy_sample.likelihood_ratio,
                "privacy_label": privacy_sample.label,
                "efficacy_label": efficacy_sample.label,
                "privacy_positive_shadow_condition": (
                    privacy.positive_shadow_condition
                ),
                "privacy_negative_shadow_condition": (
                    privacy.negative_shadow_condition
                ),
                "privacy_positive_shadow_distribution": _json_distribution(
                    privacy_sample.positive_shadow_distribution
                ),
                "privacy_negative_shadow_distribution": _json_distribution(
                    privacy_sample.negative_shadow_distribution
                ),
                "efficacy_positive_shadow_condition": (
                    efficacy.positive_shadow_condition
                ),
                "efficacy_negative_shadow_condition": (
                    efficacy.negative_shadow_condition
                ),
                "efficacy_positive_shadow_distribution": _json_distribution(
                    efficacy_sample.positive_shadow_distribution
                ),
                "efficacy_negative_shadow_distribution": _json_distribution(
                    efficacy_sample.negative_shadow_distribution
                ),
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    resolved_paths = {
        args.capture_path.resolve(),
        args.target_data_path.resolve(),
        args.output.resolve(),
        metrics_output.resolve(),
    }
    if len(resolved_paths) != 4:
        raise ValueError("Capture, dataset, CSV, and metric paths must be different.")

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    privacy, efficacy = _load_capture(args.capture_path)
    captured_metrics = {
        "privacy": _metrics_from_samples(privacy.samples),
        "efficacy": _metrics_from_samples(efficacy.samples),
    }
    _verify_matches_official(privacy, captured_metrics["privacy"])
    _verify_matches_official(efficacy, captured_metrics["efficacy"])
    if args.verify_expected_aucs:
        _verify_reference_shape(privacy)
        _verify_reference_shape(efficacy)
        _verify_expected_auc(
            "privacy",
            float(captured_metrics["privacy"]["AUC"]),
            args.expected_privacy_auc,
        )
        _verify_expected_auc(
            "efficacy",
            float(captured_metrics["efficacy"]["AUC"]),
            args.expected_efficacy_auc,
        )

    print("[INFO] Loading target dataset only to decode sample text...")
    target_dataset = load_from_disk(str(args.target_data_path))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = _build_rows(privacy, efficacy, target_dataset, tokenizer)
    _write_csv(rows, args.output)
    exported_metrics = {
        "privacy": _metrics_from_csv(args.output, "privacy"),
        "efficacy": _metrics_from_csv(args.output, "efficacy"),
    }
    _verify_matches_official(privacy, exported_metrics["privacy"])
    _verify_matches_official(efficacy, exported_metrics["efficacy"])
    if args.verify_expected_aucs:
        _verify_expected_auc(
            "privacy",
            float(exported_metrics["privacy"]["AUC"]),
            args.expected_privacy_auc,
        )
        _verify_expected_auc(
            "efficacy",
            float(exported_metrics["efficacy"]["AUC"]),
            args.expected_efficacy_auc,
        )

    verification = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "direct official mia_inference.py KDE-loop capture",
        "capture_path": str(args.capture_path.resolve()),
        "capture_sha256": _sha256_file(args.capture_path),
        "target_data_path": str(args.target_data_path.resolve()),
        "tokenizer": args.tokenizer,
        "sample_count": len(rows),
        "official_metrics": {
            "privacy": privacy.metrics,
            "efficacy": efficacy.metrics,
        },
        "metrics_reconstructed_from_exported_csv": exported_metrics,
        "official_metric_match": True,
        "expected_auc_check_enabled": args.verify_expected_aucs,
        "expected_auc_four_decimals": {
            "privacy": args.expected_privacy_auc,
            "efficacy": args.expected_efficacy_auc,
        },
        "expected_auc_match": True if args.verify_expected_aucs else None,
        "shadow_conditions": {
            "privacy": {
                "positive": privacy.positive_shadow_condition,
                "negative": privacy.negative_shadow_condition,
            },
            "efficacy": {
                "positive": efficacy.positive_shadow_condition,
                "negative": efficacy.negative_shadow_condition,
            },
        },
        "output_csv": str(args.output.resolve()),
        "output_csv_sha256": _sha256_file(args.output),
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as output_file:
        json.dump(verification, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(
        f"[INFO] Wrote {len(rows)} directly captured samples to "
        f"{args.output.resolve()}"
    )
    print(
        "[VERIFY] privacy AUC="
        f"{float(exported_metrics['privacy']['AUC']):.4f}; "
        "efficacy AUC="
        f"{float(exported_metrics['efficacy']['AUC']):.4f}"
    )
    print(f"[INFO] Wrote metric verification to {metrics_output.resolve()}")


if __name__ == "__main__":
    main()
