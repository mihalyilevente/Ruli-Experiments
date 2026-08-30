import importlib.util
import json
import math
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "experiment_2"
    / "evaluate_experiment_2a.py"
)
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "experiment_2"
    / "results"
    / "intervention_manifest.json"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_experiment_2a", EVALUATOR_PATH
)
EVALUATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVALUATOR)


class _FakeKDE:
    def __init__(self, density, logpdf):
        self.density = density
        self.log_density = logpdf

    def evaluate(self, values):
        self._validate(values)
        return [self.density]

    def logpdf(self, values):
        self._validate(values)
        return [self.log_density]

    @staticmethod
    def _validate(values):
        if len(values) != 1:
            raise AssertionError("Expected one observation.")


class Experiment2AEvaluationTests(unittest.TestCase):
    def test_frozen_manifest_and_cohorts_pass(self):
        manifest, metadata = EVALUATOR._validate_manifest(MANIFEST_PATH)
        self.assertEqual(
            metadata["frozen_content_sha256"],
            EVALUATOR.FROZEN_MANIFEST_CONTENT_SHA256,
        )
        self.assertEqual(len(manifest["sets"]["S_sample_ids"]), 28)
        self.assertEqual(
            len(manifest["sets"]["negative_control_sample_ids"]), 121
        )

    def test_kde_score_uses_raw_logpdf_difference_and_reference_transform(self):
        log_odds, score = EVALUATOR._score_kde(
            1.25,
            (_FakeKDE(0.25, -2.0), _FakeKDE(0.75, -5.5)),
            "privacy",
            "LOW",
            201,
        )
        self.assertEqual(log_odds, 3.5)
        self.assertTrue(
            math.isclose(score, 0.25 / (1.0 + 1e-12), abs_tol=1e-15)
        )

    def test_nonfinite_kde_is_reported_and_rejected(self):
        with self.assertRaisesRegex(
            FloatingPointError, "condition=LOW, sample_id=201"
        ):
            EVALUATOR._score_kde(
                1.25,
                (_FakeKDE(0.0, -math.inf), _FakeKDE(0.75, -5.5)),
                "privacy",
                "LOW",
                201,
            )

    def test_nine_shadow_artifact_has_three_observations_per_condition(self):
        observations = EVALUATOR._plain_shadow_observations(
            [1.0, 2.0, 3.0], "in_original", 200
        )
        self.assertEqual(observations, [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(
            ValueError, "not the fixed 3 produced by 9 shadow models"
        ):
            EVALUATOR._plain_shadow_observations(
                [float(value) for value in range(9)], "in_original", 200
            )

    def test_primary_contrast_preserves_manifest_order(self):
        supported = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["sets"][
            "S_sample_ids"
        ]
        rows = []
        for condition in EVALUATOR.CONDITIONS:
            for index, sample_id in enumerate(supported):
                base = float(index)
                value = {
                    "HIGH": base + 3.0,
                    "LOW": base + 1.0,
                    "PLACEBO": base + 2.0,
                }[condition]
                rows.append(
                    {
                        "condition": condition,
                        "sample_id": sample_id,
                        "split": "unlearn",
                        "privacy_log_odds": value,
                    }
                )
        contrast = EVALUATOR._contrast_rows(rows, supported)
        self.assertEqual([row["sample_id"] for row in contrast], supported)
        self.assertEqual(contrast[0]["LOW_minus_PLACEBO"], -1.0)
        self.assertEqual(contrast[-1]["LOW_minus_PLACEBO"], -1.0)

    def test_condition_alignment_rejects_reordered_rows(self):
        unlearn_ids = [200, 201]
        out_ids = [400, 401]
        rows = [
            {"condition": condition, "split": split, "sample_id": sample_id}
            for condition in EVALUATOR.CONDITIONS
            for split, ids in (("unlearn", unlearn_ids), ("out", out_ids))
            for sample_id in ids
        ]
        EVALUATOR._validate_condition_row_alignment(rows, unlearn_ids, out_ids)
        rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaisesRegex(ValueError, "reordered, dropped, or duplicated"):
            EVALUATOR._validate_condition_row_alignment(
                rows, unlearn_ids, out_ids
            )

    def test_manifest_file_is_valid_json(self):
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["manifest_hash"]["sha256"],
            EVALUATOR.FROZEN_MANIFEST_CONTENT_SHA256,
        )


if __name__ == "__main__":
    unittest.main()
