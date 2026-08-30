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


if __name__ == "__main__":
    unittest.main()
