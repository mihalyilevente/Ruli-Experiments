"""Analyze source-specific semantic support against per-sample RULI scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


EXPECTED_UNLEARN_ROWS = 200
EXPECTED_HEADING_ROWS = 51
EXPECTED_NON_HEADING_ROWS = 149
RETAIN_SOURCES = ("target_in", "wikitext_attack")
TOP_KS = (5, 10, 25)
THRESHOLDS = (0.70, 0.75, 0.80)
OUTCOMES = ("privacy_score", "efficacy_score")
LENGTH_COVARIATES = ("gpt2_token_count", "word_count", "character_length")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _support_variables() -> tuple[str, ...]:
    variables: list[str] = []
    for source in RETAIN_SOURCES:
        variables.extend(
            (
                f"{source}_max_similarity",
                *(f"{source}_mean_top_{top_k}_similarity" for top_k in TOP_KS),
            )
        )
        for threshold in THRESHOLDS:
            suffix = f"{threshold:.2f}".replace(".", "_")
            variables.extend(
                (
                    f"{source}_neighbor_count_ge_{suffix}",
                    f"{source}_similarity_sum_ge_{suffix}",
                )
            )
    return tuple(variables)


SUPPORT_VARIABLES = _support_variables()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze source-specific retained semantic support for the primary "
            "non-heading UNLEARN subset using existing generated artifacts only."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "unlearn_semantic_support.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "semantic_support_analysis",
    )
    parser.add_argument(
        "--verify-reference-counts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require 200 total, 51 heading, and 149 non-heading rows.",
    )
    parser.add_argument(
        "--minimum-n",
        type=int,
        default=3,
        help="Minimum complete observations required for a correlation.",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_value(row: Mapping[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Semantic-support CSV does not exist: {path}")
    required = {
        "sample_id",
        "split",
        "privacy_score",
        "efficacy_score",
        "character_length",
        "word_count",
        "gpt2_token_count",
        "is_wikitext_heading",
        "exact_retained_text_duplicate",
        "target_in_exact_duplicate_count",
        "wikitext_attack_exact_duplicate_count",
        *SUPPORT_VARIABLES,
    }
    rows: list[dict[str, str]] = []
    sample_ids: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError("Semantic-support CSV has duplicate column names.")
        missing = required.difference(fields)
        if missing:
            raise ValueError(
                "Semantic-support CSV is missing: " + ", ".join(sorted(missing))
            )
        for source_row, row in enumerate(reader, start=2):
            try:
                sample_id = int(row["sample_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV row {source_row} has invalid sample_id."
                ) from exc
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            if row["split"].strip().lower() != "unlearn":
                raise ValueError(
                    f"Semantic-support row {sample_id} is not in the UNLEARN split."
                )
            for field in (
                *OUTCOMES,
                "character_length",
                "word_count",
                "is_wikitext_heading",
                "exact_retained_text_duplicate",
                "target_in_exact_duplicate_count",
                "wikitext_attack_exact_duplicate_count",
                *SUPPORT_VARIABLES,
            ):
                if _finite_value(row, field) is None:
                    raise ValueError(
                        f"Sample {sample_id} has a missing/non-finite {field}."
                    )
            for field in ("is_wikitext_heading", "exact_retained_text_duplicate"):
                if _finite_value(row, field) not in {0.0, 1.0}:
                    raise ValueError(f"Sample {sample_id} has non-binary {field}.")
            duplicate_total = sum(
                int(float(row[f"{source}_exact_duplicate_count"]))
                for source in RETAIN_SOURCES
            )
            if int(float(row["exact_retained_text_duplicate"])) != int(
                duplicate_total > 0
            ):
                raise ValueError(
                    f"Sample {sample_id} has inconsistent exact-duplicate fields."
                )
            rows.append(dict(row))
    if len(rows) != EXPECTED_UNLEARN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_UNLEARN_ROWS} UNLEARN rows; found {len(rows)}."
        )
    return sorted(rows, key=lambda row: int(row["sample_id"]))


def _complete_arrays(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> tuple[list[np.ndarray], int]:
    complete: list[list[float]] = []
    for row in rows:
        values = [_finite_value(row, field) for field in fields]
        if all(value is not None for value in values):
            complete.append([float(value) for value in values if value is not None])
    if not complete:
        return [np.asarray([], dtype=np.float64) for _ in fields], 0
    matrix = np.asarray(complete, dtype=np.float64)
    return [matrix[:, index] for index in range(len(fields))], len(complete)


def _spearman(
    x: np.ndarray, y: np.ndarray, minimum_n: int
) -> tuple[float | None, float | None, str]:
    if len(x) < minimum_n:
        return None, None, "insufficient_n"
    if np.ptp(x) == 0:
        return None, None, "constant_predictor"
    if np.ptp(y) == 0:
        return None, None, "constant_outcome"
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue), "ok"


def _partial_spearman(
    support: np.ndarray,
    outcome: np.ndarray,
    length: np.ndarray,
    minimum_n: int,
) -> tuple[float | None, float | None, str]:
    if len(support) < minimum_n:
        return None, None, "insufficient_n"
    if np.ptp(support) == 0:
        return None, None, "constant_predictor"
    if np.ptp(outcome) == 0:
        return None, None, "constant_outcome"
    if np.ptp(length) == 0:
        return None, None, "constant_length_covariate"
    ranked_support = rankdata(support, method="average")
    ranked_outcome = rankdata(outcome, method="average")
    ranked_length = rankdata(length, method="average")
    design = np.column_stack((np.ones(len(length)), ranked_length))
    support_residual = ranked_support - design @ np.linalg.lstsq(
        design, ranked_support, rcond=None
    )[0]
    outcome_residual = ranked_outcome - design @ np.linalg.lstsq(
        design, ranked_outcome, rcond=None
    )[0]
    if np.ptp(support_residual) == 0 or np.ptp(outcome_residual) == 0:
        return None, None, "constant_residual"
    result = pearsonr(support_residual, outcome_residual)
    return float(result.statistic), float(result.pvalue), "ok"


def _benjamini_hochberg(
    rows: Sequence[dict[str, Any]], p_field: str, q_field: str
) -> None:
    valid = [
        (index, float(row[p_field]))
        for index, row in enumerate(rows)
        if row.get(p_field) is not None
    ]
    ordered = sorted(valid, key=lambda item: item[1])
    running_minimum = 1.0
    adjusted: dict[int, float] = {}
    test_count = len(ordered)
    for reverse_index in range(test_count - 1, -1, -1):
        row_index, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running_minimum = min(running_minimum, p_value * test_count / rank)
        adjusted[row_index] = min(1.0, running_minimum)
    for row_index, value in adjusted.items():
        rows[row_index][q_field] = value


def _variable_parts(variable: str) -> tuple[str, str]:
    for source in RETAIN_SOURCES:
        prefix = f"{source}_"
        if variable.startswith(prefix):
            return source, variable.removeprefix(prefix)
    raise ValueError(f"Unrecognized support variable: {variable}")


def _primary_length_covariate(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...], bool]:
    gpt2_complete = all(
        _finite_value(row, "gpt2_token_count") is not None for row in rows
    )
    if gpt2_complete:
        return "gpt2_token_count", LENGTH_COVARIATES, True
    return "word_count", ("word_count", "character_length"), False


def _correlation_rows(
    rows: Sequence[Mapping[str, Any]],
    primary_length: str,
    minimum_n: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for variable in SUPPORT_VARIABLES:
        source, metric = _variable_parts(variable)
        for outcome in OUTCOMES:
            ordinary_arrays, ordinary_n = _complete_arrays(rows, (variable, outcome))
            rho, p_value, status = _spearman(
                ordinary_arrays[0], ordinary_arrays[1], minimum_n
            )
            partial_arrays, partial_n = _complete_arrays(
                rows, (variable, outcome, primary_length)
            )
            partial_rho, partial_p, partial_status = _partial_spearman(
                partial_arrays[0],
                partial_arrays[1],
                partial_arrays[2],
                minimum_n,
            )
            results.append(
                {
                    "analysis_subset": "non_heading_unlearn",
                    "retain_source": source,
                    "support_metric": metric,
                    "support_variable": variable,
                    "outcome": outcome,
                    "spearman_n": ordinary_n,
                    "spearman_rho": rho,
                    "spearman_p_value": p_value,
                    "spearman_q_value_bh": None,
                    "spearman_status": status,
                    "primary_length_covariate": primary_length,
                    "partial_spearman_n": partial_n,
                    "partial_spearman_rho": partial_rho,
                    "partial_spearman_p_value": partial_p,
                    "partial_spearman_q_value_bh": None,
                    "partial_spearman_status": partial_status,
                }
            )
    _benjamini_hochberg(results, "spearman_p_value", "spearman_q_value_bh")
    _benjamini_hochberg(
        results, "partial_spearman_p_value", "partial_spearman_q_value_bh"
    )
    return results


def _sensitivity_rows(
    rows: Sequence[Mapping[str, Any]],
    covariates: Sequence[str],
    primary_length: str,
    minimum_n: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for covariate in covariates:
        family: list[dict[str, Any]] = []
        for variable in SUPPORT_VARIABLES:
            source, metric = _variable_parts(variable)
            for outcome in OUTCOMES:
                arrays, n = _complete_arrays(rows, (variable, outcome, covariate))
                rho, p_value, status = _partial_spearman(
                    arrays[0], arrays[1], arrays[2], minimum_n
                )
                family.append(
                    {
                        "analysis_subset": "non_heading_unlearn",
                        "retain_source": source,
                        "support_metric": metric,
                        "support_variable": variable,
                        "outcome": outcome,
                        "length_covariate": covariate,
                        "is_primary_length_covariate": int(
                            covariate == primary_length
                        ),
                        "n": n,
                        "partial_spearman_rho": rho,
                        "partial_spearman_p_value": p_value,
                        "partial_spearman_q_value_bh": None,
                        "status": status,
                    }
                )
        _benjamini_hochberg(
            family, "partial_spearman_p_value", "partial_spearman_q_value_bh"
        )
        results.extend(family)
    return results


def _feature_summary_rows(
    populations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for subset, rows in populations.items():
        for variable in SUPPORT_VARIABLES:
            source, metric = _variable_parts(variable)
            values = np.asarray(
                [
                    value
                    for row in rows
                    if (value := _finite_value(row, variable)) is not None
                ],
                dtype=np.float64,
            )
            results.append(
                {
                    "analysis_subset": subset,
                    "retain_source": source,
                    "support_metric": metric,
                    "support_variable": variable,
                    "n": len(values),
                    "missing_n": len(rows) - len(values),
                    "mean": float(np.mean(values)) if len(values) else None,
                    "standard_deviation": (
                        float(np.std(values)) if len(values) else None
                    ),
                    "minimum": float(np.min(values)) if len(values) else None,
                    "q25": float(np.quantile(values, 0.25)) if len(values) else None,
                    "median": float(np.median(values)) if len(values) else None,
                    "q75": float(np.quantile(values, 0.75)) if len(values) else None,
                    "maximum": float(np.max(values)) if len(values) else None,
                }
            )
    return results


def _predictor_correlation_rows(
    rows: Sequence[Mapping[str, Any]], minimum_n: int
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for first, second in combinations(SUPPORT_VARIABLES, 2):
        arrays, n = _complete_arrays(rows, (first, second))
        rho, p_value, status = _spearman(arrays[0], arrays[1], minimum_n)
        results.append(
            {
                "analysis_subset": "non_heading_unlearn",
                "predictor_1": first,
                "predictor_2": second,
                "n": n,
                "spearman_rho": rho,
                "p_value": p_value,
                "status": status,
            }
        )
    return results


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.minimum_n < 3:
        raise ValueError("--minimum-n must be at least 3.")
    if args.output_dir.resolve() == args.input.resolve():
        raise ValueError("--output-dir must not overwrite the input CSV.")
    rows = _load_rows(args.input)
    heading_rows = [row for row in rows if int(row["is_wikitext_heading"]) == 1]
    primary_rows = [row for row in rows if int(row["is_wikitext_heading"]) == 0]
    if args.verify_reference_counts and (
        len(heading_rows) != EXPECTED_HEADING_ROWS
        or len(primary_rows) != EXPECTED_NON_HEADING_ROWS
    ):
        raise ValueError(
            "Reference subset counts differ: expected 51 heading and 149 "
            f"non-heading rows; found {len(heading_rows)} and {len(primary_rows)}."
        )
    primary_length, sensitivity_covariates, gpt2_available = (
        _primary_length_covariate(primary_rows)
    )

    correlations = _correlation_rows(
        primary_rows, primary_length, args.minimum_n
    )
    sensitivities = _sensitivity_rows(
        primary_rows, sensitivity_covariates, primary_length, args.minimum_n
    )
    feature_summaries = _feature_summary_rows(
        {"all_unlearn": rows, "non_heading_unlearn": primary_rows}
    )
    predictor_correlations = _predictor_correlation_rows(
        primary_rows, args.minimum_n
    )

    outputs = {
        "correlations": args.output_dir / "correlations.csv",
        "length_sensitivity": args.output_dir / "length_sensitivity.csv",
        "feature_summaries": args.output_dir / "feature_summaries.csv",
        "predictor_correlations": args.output_dir / "predictor_correlations.csv",
    }
    _write_csv(outputs["correlations"], correlations, tuple(correlations[0]))
    _write_csv(
        outputs["length_sensitivity"], sensitivities, tuple(sensitivities[0])
    )
    _write_csv(
        outputs["feature_summaries"], feature_summaries, tuple(feature_summaries[0])
    )
    _write_csv(
        outputs["predictor_correlations"],
        predictor_correlations,
        tuple(predictor_correlations[0]),
    )

    valid_predictor_pairs = [
        row for row in predictor_correlations if row["spearman_rho"] is not None
    ]
    highly_correlated = sorted(
        (
            row
            for row in valid_predictor_pairs
            if abs(float(row["spearman_rho"])) >= 0.9
        ),
        key=lambda row: abs(float(row["spearman_rho"])),
        reverse=True,
    )
    support_metadata_path = args.input.with_suffix(".metadata.json")
    sample_counts = {
        "all_unlearn": len(rows),
        "wikitext_heading": len(heading_rows),
        "non_heading_unlearn": len(primary_rows),
        "exact_retained_text_duplicate": sum(
            int(row["exact_retained_text_duplicate"]) for row in rows
        ),
        "non_heading_exact_retained_text_duplicate": sum(
            int(row["exact_retained_text_duplicate"]) for row in primary_rows
        ),
    }
    for source in RETAIN_SOURCES:
        count_field = f"{source}_exact_duplicate_count"
        sample_counts[f"unlearn_with_{source}_exact_duplicate"] = sum(
            int(row[count_field]) > 0 for row in rows
        )
        sample_counts[f"{source}_exact_duplicate_matches_total"] = sum(
            int(row[count_field]) for row in rows
        )
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": "source-specific semantic support; no graph metrics",
        "analysis_subset": {
            "name": "non_heading_unlearn",
            "exact_definition": (
                "rows in unlearn_semantic_support.csv with split == 'unlearn' "
                "and is_wikitext_heading == 0"
            ),
            "heading_definition": (
                "full text matches WikiText heading syntax with equal matching "
                "'=' levels and whitespace around nonempty heading text"
            ),
            "heading_strings_are_semantic_duplicates": False,
        },
        "sample_counts": sample_counts,
        "retained_source_counts": {
            "target_in": 200,
            "wikitext_attack": 15_000,
        },
        "selected_support_variables": list(SUPPORT_VARIABLES),
        "outcomes": list(OUTCOMES),
        "length_adjustment": {
            "primary_covariate": primary_length,
            "gpt2_token_count_complete_in_primary_subset": gpt2_available,
            "sensitivity_covariates": list(sensitivity_covariates),
            "partial_spearman_implementation": (
                "Rank support, outcome, and length separately using average ranks; "
                "OLS-residualize ranked support and ranked outcome separately "
                "against an intercept plus ranked length; apply Pearson "
                "correlation to the two residual vectors."
            ),
        },
        "multiple_testing": {
            "method": "Benjamini-Hochberg FDR",
            "ordinary_family": (
                "all selected support-variable x outcome ordinary Spearman tests "
                "in correlations.csv"
            ),
            "primary_partial_family": (
                "all selected support-variable x outcome primary-length partial "
                "Spearman tests in correlations.csv"
            ),
            "sensitivity_families": (
                "separate support-variable x outcome family for each length "
                "covariate in length_sensitivity.csv"
            ),
        },
        "predictor_redundancy_diagnostic": {
            "pair_count": len(predictor_correlations),
            "absolute_rho_ge_0_90_count": len(highly_correlated),
            "top_pairs_by_absolute_rho": highly_correlated[:20],
        },
        "input_artifacts": {
            "semantic_support_csv": {
                "path": str(args.input.resolve()),
                "sha256": _sha256_file(args.input),
                "bytes": args.input.stat().st_size,
            },
            "semantic_support_metadata": (
                {
                    "path": str(support_metadata_path.resolve()),
                    "sha256": _sha256_file(support_metadata_path),
                    "bytes": support_metadata_path.stat().st_size,
                }
                if support_metadata_path.is_file()
                else None
            ),
        },
        "library_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "rows": {
                    "correlations": len(correlations),
                    "length_sensitivity": len(sensitivities),
                    "feature_summaries": len(feature_summaries),
                    "predictor_correlations": len(predictor_correlations),
                }[name],
            }
            for name, path in outputs.items()
        },
    }
    summary_output = args.output_dir / "analysis_summary.json"
    with summary_output.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    print(
        f"[INFO] Analyzed {len(primary_rows)} non-heading UNLEARN samples "
        f"using {primary_length} as the primary length covariate."
    )
    print(
        f"[INFO] Wrote {len(correlations)} primary correlation rows and "
        f"{len(predictor_correlations)} predictor-pair diagnostics to "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
