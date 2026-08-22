"""Flatten the JSON captured by official RULI into the Experiment 1 CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_curve


RESULTS_DIR = Path(__file__).resolve().parent / "results"
ATTACK_FIELDS = {
    "privacy": (
        "privacy_kde_likelihood_ratio_score",
        "privacy_observed_target_loss",
    ),
    "efficacy": (
        "efficacy_kde_likelihood_ratio_score",
        "efficacy_observed_target_loss",
    ),
}
CSV_FIELDS = (
    "sample_id",
    "text",
    "token_ids",
    "split",
    "label",
    "privacy_observed_loss",
    "efficacy_observed_loss",
    "privacy_score",
    "efficacy_score",
    "privacy_label",
    "efficacy_label",
    "unlearn_unlearned_shadow_observations",
    "out_unlearned_shadow_observations",
    "out_original_shadow_observations",
)
EXPECTED_CONDITIONS = {
    "privacy": ["unlearn_unlearned", "out_unlearned"],
    "efficacy": ["unlearn_unlearned", "out_original"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten values captured during official RULI mia_inference.py "
            "execution. This command performs no model inference or KDE."
        )
    )
    parser.add_argument(
        "--capture-path",
        type=Path,
        default=RESULTS_DIR / "official_ruli_samples.json",
        help="JSON produced by mia_inference.py --per_sample_output.",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "ruli_scores.csv"
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Defaults to <output stem>.metrics.json.",
    )
    parser.add_argument("--expected-privacy-auc", type=float, default=0.8531)
    parser.add_argument("--expected-privacy-acc", type=float, default=0.7700)
    parser.add_argument("--expected-efficacy-auc", type=float, default=0.8589)
    parser.add_argument("--expected-efficacy-acc", type=float, default=0.7925)
    parser.add_argument(
        "--verify-reference-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the four exported metrics to match the official 9-shadow "
            "reference at four decimal places (default: enabled)."
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
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {description}: {value!r}")
    return result


def _float_list(value: Any, description: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected a sequence for {description}.")
    result = [_finite_float(item, description) for item in value]
    if not result:
        raise ValueError(f"Empty {description}.")
    return result


def _load_capture(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Official RULI capture does not exist: {path}")
    with path.open("r", encoding="utf-8") as input_file:
        capture = json.load(input_file)
    if not isinstance(capture, Mapping) or capture.get("schema_version") != 1:
        raise ValueError("Unsupported or missing official capture schema version.")
    if capture.get("source") != "official text/mia_inference.py evaluator execution":
        raise ValueError("Capture was not produced by the official evaluator hook.")
    if capture.get("shadow_conditions") != EXPECTED_CONDITIONS:
        raise ValueError("Capture has unexpected privacy or efficacy KDE conditions.")
    if capture.get("sanity_check") != "passed":
        raise ValueError("Official post-export sanity check is absent or failed.")
    rows = capture.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Capture contains no per-sample rows.")
    return dict(capture)


def _normalise_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    required = {
        "sample_id",
        "split",
        "label",
        "text",
        "token_ids",
        "privacy_observed_target_loss",
        "efficacy_observed_target_loss",
        "privacy_kde_likelihood_ratio_score",
        "efficacy_kde_likelihood_ratio_score",
        "shadow_observations",
    }
    normalised: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row_number, raw_row in enumerate(rows, start=1):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"Capture row {row_number} is not an object.")
        missing = required.difference(raw_row)
        if missing:
            raise ValueError(
                f"Capture row {row_number} is missing: {', '.join(sorted(missing))}"
            )
        sample_id = int(raw_row["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id {sample_id}.")
        seen_ids.add(sample_id)
        label = int(raw_row["label"])
        split = str(raw_row["split"])
        if (split, label) not in {("unlearn", 1), ("out", 0)}:
            raise ValueError(
                f"Sample {sample_id} has inconsistent split/label {split!r}/{label}."
            )
        token_ids = raw_row["token_ids"]
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"Sample {sample_id} has no token IDs.")
        shadows = raw_row["shadow_observations"]
        if not isinstance(shadows, Mapping) or set(shadows) != {
            "unlearn_unlearned",
            "out_unlearned",
            "out_original",
        }:
            raise ValueError(f"Sample {sample_id} has incomplete shadow observations.")

        row = dict(raw_row)
        row["sample_id"] = sample_id
        row["label"] = label
        row["token_ids"] = [int(token_id) for token_id in token_ids]
        for score_field, loss_field in ATTACK_FIELDS.values():
            row[score_field] = _finite_float(row[score_field], score_field)
            row[loss_field] = _finite_float(row[loss_field], loss_field)
        row["shadow_observations"] = {
            name: _float_list(values, f"sample {sample_id} {name}")
            for name, values in shadows.items()
        }
        normalised.append(row)
    return normalised


def _metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels_array)) != 2:
        raise ValueError("Both unlearn and out rows are required.")
    fpr, tpr, _ = roc_curve(labels_array, scores_array)
    return {
        "AUC": float(auc(fpr, tpr)),
        "ACC": float(accuracy_score(labels_array, scores_array > 0.5)),
        "Total": int(len(labels_array)),
    }


def _metrics_from_rows(
    rows: Sequence[Mapping[str, Any]], score_field: str
) -> dict[str, float | int]:
    return _metrics(
        [int(row["label"]) for row in rows],
        [float(row[score_field]) for row in rows],
    )


def _verify_official_metrics(
    official: Mapping[str, Any], recomputed: Mapping[str, Mapping[str, Any]]
) -> None:
    for attack_name in ATTACK_FIELDS:
        if attack_name not in official:
            raise ValueError(f"Missing official {attack_name} metrics.")
        for metric_name in ("AUC", "ACC"):
            official_value = float(official[attack_name][metric_name])
            actual_value = float(recomputed[attack_name][metric_name])
            if not math.isclose(
                official_value, actual_value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"Rows do not reproduce official {attack_name} {metric_name}: "
                    f"official={official_value}, rows={actual_value}."
                )
        if int(official[attack_name]["Total"]) != recomputed[attack_name]["Total"]:
            raise ValueError(f"Rows do not reproduce official {attack_name} Total.")


def _verify_reference_metrics(
    recomputed: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Mapping[str, float]]
) -> None:
    for attack_name in ATTACK_FIELDS:
        for metric_name in ("AUC", "ACC"):
            actual_text = f"{float(recomputed[attack_name][metric_name]):.4f}"
            expected_text = f"{expected[attack_name][metric_name]:.4f}"
            if actual_text != expected_text:
                raise ValueError(
                    f"Expected official 9-shadow {attack_name} {metric_name} "
                    f"{expected_text}, exported rows give {actual_text}. Investigate "
                    "the run inputs and capture; the attack values were not adjusted."
                )


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        shadows = row["shadow_observations"]
        result.append(
            {
                "sample_id": row["sample_id"],
                "text": row["text"],
                "token_ids": json.dumps(row["token_ids"], separators=(",", ":")),
                "split": row["split"],
                "label": row["label"],
                "privacy_observed_loss": row["privacy_observed_target_loss"],
                "efficacy_observed_loss": row["efficacy_observed_target_loss"],
                "privacy_score": row["privacy_kde_likelihood_ratio_score"],
                "efficacy_score": row["efficacy_kde_likelihood_ratio_score"],
                "privacy_label": row["label"],
                "efficacy_label": row["label"],
                "unlearn_unlearned_shadow_observations": json.dumps(
                    shadows["unlearn_unlearned"], separators=(",", ":")
                ),
                "out_unlearned_shadow_observations": json.dumps(
                    shadows["out_unlearned"], separators=(",", ":")
                ),
                "out_original_shadow_observations": json.dumps(
                    shadows["out_original"], separators=(",", ":")
                ),
            }
        )
    return result


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_from_csv(path: Path, attack_name: str) -> dict[str, float | int]:
    labels: list[int] = []
    scores: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for row in csv.DictReader(input_file):
            labels.append(int(row["label"]))
            scores.append(float(row[f"{attack_name}_score"]))
    return _metrics(labels, scores)


def main() -> None:
    args = parse_args()
    metrics_output = args.metrics_output or args.output.with_suffix(".metrics.json")
    if len({args.capture_path.resolve(), args.output.resolve(), metrics_output.resolve()}) != 3:
        raise ValueError("Capture, CSV, and metrics paths must be different.")

    capture = _load_capture(args.capture_path)
    rows = _normalise_rows(capture["rows"])
    direct_metrics = {
        attack_name: _metrics_from_rows(rows, fields[0])
        for attack_name, fields in ATTACK_FIELDS.items()
    }
    _verify_official_metrics(capture["official_metrics"], direct_metrics)

    csv_rows = _csv_rows(rows)
    _write_csv(csv_rows, args.output)
    exported_metrics = {
        attack_name: _metrics_from_csv(args.output, attack_name)
        for attack_name in ATTACK_FIELDS
    }
    _verify_official_metrics(capture["official_metrics"], exported_metrics)

    expected = {
        "privacy": {"AUC": args.expected_privacy_auc, "ACC": args.expected_privacy_acc},
        "efficacy": {"AUC": args.expected_efficacy_auc, "ACC": args.expected_efficacy_acc},
    }
    if args.verify_reference_metrics:
        _verify_reference_metrics(exported_metrics, expected)

    verification = {
        "schema_version": 1,
        "source": "direct official mia_inference.py per-sample JSON",
        "capture_path": str(args.capture_path.resolve()),
        "capture_sha256": _sha256_file(args.capture_path),
        "sample_count": len(rows),
        "official_metrics": capture["official_metrics"],
        "metrics_recomputed_from_exported_csv": exported_metrics,
        "official_metric_match": True,
        "reference_metric_check_enabled": args.verify_reference_metrics,
        "expected_9_shadow_metrics": expected,
        "reference_metric_match": True if args.verify_reference_metrics else None,
        "output_csv": str(args.output.resolve()),
        "output_csv_sha256": _sha256_file(args.output),
    }
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with metrics_output.open("w", encoding="utf-8") as output_file:
        json.dump(verification, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(f"[INFO] Wrote {len(rows)} captured rows to {args.output.resolve()}")
    print(
        "[VERIFY] Privacy AUC="
        f"{exported_metrics['privacy']['AUC']:.4f}, "
        f"ACC={exported_metrics['privacy']['ACC']:.4f}; "
        "Efficacy AUC="
        f"{exported_metrics['efficacy']['AUC']:.4f}, "
        f"ACC={exported_metrics['efficacy']['ACC']:.4f}"
    )
    print(f"[INFO] Wrote verification to {metrics_output.resolve()}")


if __name__ == "__main__":
    main()
