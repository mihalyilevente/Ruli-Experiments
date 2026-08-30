import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    REPOSITORY_ROOT / "experiments" / "experiment_2" / "run_experiment_2a.py"
)
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "experiment_2"
    / "results"
    / "intervention_manifest.json"
)
SPEC = importlib.util.spec_from_file_location("run_experiment_2a", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)
ORCHESTRATOR_PATH = (
    REPOSITORY_ROOT / "experiments" / "experiment_2" / "run_remaining_seeds.py"
)
ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "run_remaining_seeds", ORCHESTRATOR_PATH
)
ORCHESTRATOR = importlib.util.module_from_spec(ORCHESTRATOR_SPEC)
assert ORCHESTRATOR_SPEC.loader is not None
ORCHESTRATOR_SPEC.loader.exec_module(ORCHESTRATOR)


def write_rehashed_manifest(payload, path):
    payload = copy.deepcopy(payload)
    payload.pop("manifest_hash", None)
    content_hash = RUNNER._canonical_sha256(payload)
    payload["manifest_hash"] = {
        "algorithm": "sha256",
        "scope": "test fixture",
        "sha256": content_hash,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class Experiment2AManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_frozen_manifest_passes(self):
        _, summary = RUNNER._load_and_validate_manifest(MANIFEST_PATH)
        self.assertEqual(summary["condition_target_counts"], {
            "HIGH": 200,
            "LOW": 200,
            "PLACEBO": 200,
        })
        self.assertEqual(summary["wikitext_count"], 15_000)

    def test_rejects_invalid_low_membership(self):
        payload = copy.deepcopy(self.manifest)
        payload["conditions"]["LOW"]["ordered_target_dataset_ids"][0] = 999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_rehashed_manifest(payload, path)
            with self.assertRaisesRegex(ValueError, "LOW is not exactly"):
                RUNNER._load_and_validate_manifest(path)

    def test_rejects_failed_protocol_validation(self):
        payload = copy.deepcopy(self.manifest)
        payload["validation"]["all_protocol_invariants_pass"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_rehashed_manifest(payload, path)
            with self.assertRaisesRegex(ValueError, "is not true"):
                RUNNER._load_and_validate_manifest(path)

    def test_rejects_a_different_internally_valid_manifest_hash(self):
        payload = copy.deepcopy(self.manifest)
        payload["created_at_utc"] = "test-only-change"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_rehashed_manifest(payload, path)
            with self.assertRaisesRegex(ValueError, "Frozen intervention manifest"):
                RUNNER._load_and_validate_manifest(path)

    def test_all_preregistered_training_seeds_are_supported(self):
        for seed in (42, 43, 44, 45, 46):
            RUNNER._validate_training_seed(seed)
            self.assertEqual(RUNNER._seed_output_root(seed).name, f"seed_{seed}")
        with self.assertRaisesRegex(ValueError, "must be preregistered"):
            RUNNER._validate_training_seed(47)

    def test_seed_output_path_must_match_requested_seed(self):
        matching = Path("results") / "experiment_2a" / "seed_43"
        self.assertEqual(
            RUNNER._validate_seed_output_path(matching, 43, "--output-root").name,
            "seed_43",
        )
        with self.assertRaisesRegex(ValueError, "seed_43"):
            RUNNER._validate_seed_output_path(
                Path("results") / "experiment_2a" / "seed_42",
                43,
                "--output-root",
            )

    def test_training_arguments_receive_model_and_data_seed(self):
        class FakeTrainingArguments:
            def __init__(self, **kwargs):
                self.seed = kwargs["seed"]
                self.data_seed = kwargs["data_seed"]

        class FakeRuliUtils:
            TrainingArguments = FakeTrainingArguments

        RUNNER._configure_training_arguments_seed(FakeRuliUtils, 46)
        training_args = FakeRuliUtils.TrainingArguments(output_dir="unused")
        self.assertEqual(training_args.seed, 46)
        self.assertEqual(training_args.data_seed, 46)

    def test_remaining_seed_orchestrator_can_never_run_seed_42(self):
        self.assertEqual(ORCHESTRATOR.SEEDS, (43, 44, 45, 46))
        self.assertNotIn(42, ORCHESTRATOR.SEEDS)


if __name__ == "__main__":
    unittest.main()
