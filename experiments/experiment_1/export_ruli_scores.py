"""Export sample-level RULI scores and losses for Experiment 1.

This script is deliberately separate from the official text attack implementation. It
loads already-trained target checkpoints and an existing shadow-output file; it never
trains or modifies a model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


SHADOW_KEYS = (
    "in_original",
    "out_original",
    "unlearn_original",
    "in_unlearned",
    "out_unlearned",
    "unlearn_unlearned",
)

CSV_FIELDS = (
    "sample_id",
    "text",
    "original_loss",
    "unlearned_loss",
    "out_shadow_mean",
    "unlearn_shadow_mean",
    "privacy_score",
    "efficacy_score",
    "privacy_label",
    "efficacy_label",
    "loss_change",
    "split",
    "efficacy_out_shadow_mean",
    "out_shadow_count",
    "unlearn_shadow_count",
    "efficacy_out_shadow_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-sample losses and KDE RULI scores from saved text-model "
            "checkpoints and shadow outputs."
        )
    )
    parser.add_argument("--shadow-path", type=Path, required=True)
    parser.add_argument("--target-data-path", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--unlearned-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "ruli_scores.csv",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name/path (defaults to --original-checkpoint).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--unlearn-start",
        type=int,
        default=200,
        help="Start offset in the sorted shadow sample IDs.",
    )
    parser.add_argument("--unlearn-count", type=int, default=200)
    parser.add_argument(
        "--out-start",
        type=int,
        default=400,
        help="Start offset in the sorted shadow sample IDs.",
    )
    parser.add_argument("--out-count", type=int, default=200)
    parser.add_argument(
        "--kde-error",
        choices=("raise", "nan"),
        default="raise",
        help=(
            "How to handle a per-sample KDE with too few or singular shadow "
            "observations. 'nan' is useful for inspecting incomplete smoke runs."
        ),
    )
    return parser.parse_args()


def _normalise_shadow_results(raw: Mapping[str, Any]) -> dict[str, dict[int, list[float]]]:
    missing = [key for key in SHADOW_KEYS if key not in raw]
    if missing:
        raise ValueError(f"Shadow file is missing keys: {', '.join(missing)}")

    normalised: dict[str, dict[int, list[float]]] = {}
    for condition in SHADOW_KEYS:
        if not isinstance(raw[condition], Mapping):
            raise TypeError(f"Shadow condition {condition!r} is not a mapping.")
        condition_values: dict[int, list[float]] = {}
        for sample_id, values in raw[condition].items():
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().reshape(-1).tolist()
            condition_values[int(sample_id)] = [float(value) for value in values]
        normalised[condition] = condition_values
    return normalised


def _select_evaluation_ids(
    all_ids: Sequence[int], start: int, count: int, name: str
) -> list[int]:
    if start < 0 or count <= 0:
        raise ValueError(f"{name} start must be >= 0 and count must be > 0.")
    selected = list(all_ids[start : start + count])
    if len(selected) != count:
        raise ValueError(
            f"Requested {count} {name} IDs at offset {start}, but only "
            f"{len(selected)} are available (total shadow IDs: {len(all_ids)})."
        )
    return selected


def _validate_ids(
    sample_ids: Iterable[int], dataset_size: int, shadows: Mapping[str, Mapping[int, Any]]
) -> None:
    for sample_id in sample_ids:
        if sample_id < 0 or sample_id >= dataset_size:
            raise IndexError(
                f"Sample ID {sample_id} is outside the target dataset of size "
                f"{dataset_size}. Check that --shadow-path and --target-data-path "
                "come from the same run."
            )
        missing = [key for key in SHADOW_KEYS if sample_id not in shadows[key]]
        if missing:
            raise ValueError(
                f"Sample ID {sample_id} is absent from shadow conditions: "
                f"{', '.join(missing)}"
            )


def _compute_losses(
    model: torch.nn.Module,
    samples: Sequence[tuple[int, Sequence[int]]],
    device: torch.device,
    batch_size: int,
    pad_token_id: int,
) -> dict[int, float]:
    """Match the official RULI last-7-next-token cross-entropy calculation."""
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    model.eval()
    losses: dict[int, float] = {}
    total_batches = math.ceil(len(samples) / batch_size)

    for batch_number, offset in enumerate(range(0, len(samples), batch_size), start=1):
        batch = samples[offset : offset + batch_size]
        lengths = [len(input_ids) for _, input_ids in batch]
        invalid = [sample_id for (sample_id, _), length in zip(batch, lengths) if length < 2]
        if invalid:
            raise ValueError(f"Samples have fewer than two tokens: {invalid}")

        max_length = max(lengths)
        input_tensor = torch.full(
            (len(batch), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_tensor)
        for row, (_, input_ids) in enumerate(batch):
            length = len(input_ids)
            input_tensor[row, :length] = torch.as_tensor(
                input_ids, dtype=torch.long, device=device
            )
            attention_mask[row, :length] = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_tensor, attention_mask=attention_mask
            ).logits

        for row, ((sample_id, _), sequence_length) in enumerate(zip(batch, lengths)):
            ngram_window = min(7, sequence_length - 1)
            start_index = max(sequence_length - ngram_window - 1, 0)
            target_positions = torch.arange(
                start_index, sequence_length - 1, device=device
            )
            selected_logits = logits[row, target_positions, :]
            selected_labels = input_tensor[row, target_positions + 1]
            loss = torch.nn.functional.cross_entropy(
                selected_logits, selected_labels, reduction="mean"
            )
            losses[sample_id] = float(loss.item())

        print(
            f"[INFO] Inference batch {batch_number}/{total_batches}",
            flush=True,
        )

    return losses


def _kde_score(
    observed_loss: float,
    positive_observations: Sequence[float],
    negative_observations: Sequence[float],
    *,
    sample_id: int,
    score_name: str,
    error_mode: str,
) -> float:
    """Return the same p_pos / (p_pos + p_neg + 1e-12) score as RULI."""
    try:
        if len(positive_observations) < 2 or len(negative_observations) < 2:
            raise ValueError(
                "at least two observations per condition are required; got "
                f"{len(positive_observations)} and {len(negative_observations)}"
            )
        positive_kde = gaussian_kde(np.asarray(positive_observations, dtype=float))
        negative_kde = gaussian_kde(np.asarray(negative_observations, dtype=float))
        p_positive = float(positive_kde.evaluate([observed_loss])[0])
        p_negative = float(negative_kde.evaluate([observed_loss])[0])
        return p_positive / (p_positive + p_negative + 1e-12)
    except (ValueError, np.linalg.LinAlgError) as exc:
        message = f"Cannot compute {score_name} KDE for sample {sample_id}: {exc}"
        if error_mode == "nan":
            print(f"[WARNING] {message}")
            return float("nan")
        raise ValueError(message) from exc


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float))) if values else float("nan")


def _build_rows(
    sample_ids: Sequence[int],
    unlearn_ids: set[int],
    dataset: Any,
    tokenizer: Any,
    original_losses: Mapping[int, float],
    unlearned_losses: Mapping[int, float],
    shadows: Mapping[str, Mapping[int, Sequence[float]]],
    kde_error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        is_unlearn = sample_id in unlearn_ids
        label = int(is_unlearn)
        original_loss = original_losses[sample_id]
        unlearned_loss = unlearned_losses[sample_id]
        unlearn_shadows = shadows["unlearn_unlearned"][sample_id]
        privacy_out_shadows = shadows["out_unlearned"][sample_id]
        efficacy_out_shadows = shadows["out_original"][sample_id]

        # These observed losses mirror MIAEvaluator.run and EfficacyEvaluator.run.
        privacy_observed_loss = unlearned_loss
        efficacy_observed_loss = unlearned_loss if is_unlearn else original_loss

        privacy_score = _kde_score(
            privacy_observed_loss,
            unlearn_shadows,
            privacy_out_shadows,
            sample_id=sample_id,
            score_name="privacy",
            error_mode=kde_error,
        )
        efficacy_score = _kde_score(
            efficacy_observed_loss,
            unlearn_shadows,
            efficacy_out_shadows,
            sample_id=sample_id,
            score_name="efficacy",
            error_mode=kde_error,
        )

        input_ids = dataset[sample_id]["input_ids"]
        rows.append(
            {
                "sample_id": sample_id,
                "text": tokenizer.decode(input_ids, skip_special_tokens=True),
                "original_loss": original_loss,
                "unlearned_loss": unlearned_loss,
                # Backward-compatible requested name: privacy OUT-after-unlearning.
                "out_shadow_mean": _mean(privacy_out_shadows),
                "unlearn_shadow_mean": _mean(unlearn_shadows),
                "privacy_score": privacy_score,
                "efficacy_score": efficacy_score,
                "privacy_label": label,
                "efficacy_label": label,
                "loss_change": unlearned_loss - original_loss,
                "split": "unlearn" if is_unlearn else "out",
                "efficacy_out_shadow_mean": _mean(efficacy_out_shadows),
                "out_shadow_count": len(privacy_out_shadows),
                "unlearn_shadow_count": len(unlearn_shadows),
                "efficacy_out_shadow_count": len(efficacy_out_shadows),
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_metrics(rows: Sequence[Mapping[str, Any]], score_name: str) -> None:
    valid_rows = [row for row in rows if not math.isnan(float(row[score_name]))]
    if len(valid_rows) != len(rows):
        print(
            f"[WARNING] {score_name}: omitted {len(rows) - len(valid_rows)} rows "
            "with NaN scores from the verification metrics."
        )
    if not valid_rows or len({int(row["privacy_label"]) for row in valid_rows}) < 2:
        print(f"[WARNING] {score_name}: insufficient valid rows to calculate metrics.")
        return

    labels = np.asarray([int(row["privacy_label"]) for row in valid_rows])
    scores = np.asarray([float(row[score_name]) for row in valid_rows])
    auc_value = roc_auc_score(labels, scores)
    accuracy = accuracy_score(labels, scores > 0.5)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)

    def tpr_at_fpr(target_fpr: float) -> float:
        eligible = np.flatnonzero(false_positive_rate <= target_fpr)
        return float(true_positive_rate[eligible[-1]]) if len(eligible) else 0.0

    print(
        f"[VERIFY] {score_name}: AUC={auc_value:.4f}, ACC={accuracy:.4f}, "
        f"TPR@1%FPR={tpr_at_fpr(0.01):.4f}, "
        f"TPR@5%FPR={tpr_at_fpr(0.05):.4f}"
    )


def main() -> None:
    args = parse_args()
    if args.unlearn_count <= 0 or args.out_count <= 0:
        raise ValueError("--unlearn-count and --out-count must be greater than zero.")

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[INFO] Loading target dataset and shadow outputs...")
    dataset = load_from_disk(str(args.target_data_path))
    raw_shadows = torch.load(
        args.shadow_path, map_location="cpu", weights_only=False
    )
    shadows = _normalise_shadow_results(raw_shadows)

    all_ids = sorted(shadows["in_original"])
    unlearn_ids = _select_evaluation_ids(
        all_ids, args.unlearn_start, args.unlearn_count, "UNLEARN"
    )
    out_ids = _select_evaluation_ids(all_ids, args.out_start, args.out_count, "OUT")
    if set(unlearn_ids) & set(out_ids):
        raise ValueError("Selected UNLEARN and OUT ID ranges overlap.")
    evaluation_ids = unlearn_ids + out_ids
    _validate_ids(evaluation_ids, len(dataset), shadows)

    tokenizer_source = args.tokenizer or str(args.original_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    samples = [(sample_id, dataset[sample_id]["input_ids"]) for sample_id in evaluation_ids]
    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    print("[INFO] Loading original checkpoint and computing losses...")
    original_model = AutoModelForCausalLM.from_pretrained(
        str(args.original_checkpoint)
    ).to(device)
    original_losses = _compute_losses(
        original_model,
        samples,
        device,
        args.batch_size,
        tokenizer.pad_token_id,
    )
    del original_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[INFO] Loading unlearned checkpoint and computing losses...")
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        str(args.unlearned_checkpoint)
    ).to(device)
    unlearned_losses = _compute_losses(
        unlearned_model,
        samples,
        device,
        args.batch_size,
        tokenizer.pad_token_id,
    )
    del unlearned_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rows = _build_rows(
        evaluation_ids,
        set(unlearn_ids),
        dataset,
        tokenizer,
        original_losses,
        unlearned_losses,
        shadows,
        args.kde_error,
    )
    _write_csv(rows, args.output)
    print(f"[INFO] Wrote {len(rows)} rows to {args.output.resolve()}")
    _print_metrics(rows, "privacy_score")
    _print_metrics(rows, "efficacy_score")


if __name__ == "__main__":
    main()
