import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPOSITORY_ROOT / "experiments" / "experiment_2"
MANIFEST_PATH = SCRIPT_DIR / "results" / "intervention_manifest.json"


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module("run_experiment_2b.py", "run_experiment_2b")
EVALUATOR = load_module("evaluate_experiment_2b.py", "evaluate_experiment_2b")
ORCHESTRATOR = load_module(
    "run_experiment_2b_seeds.py", "run_experiment_2b_seeds"
)


class Experiment2BProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_frozen_protocol_and_condition_sets_pass(self):
        manifest, _ = RUNNER.COMMON._load_and_validate_manifest(MANIFEST_PATH)
        condition_ids, validation = RUNNER._validate_experiment_2b_protocol(
            manifest
        )
        self.assertTrue(all(validation.values()))
        self.assertEqual(condition_ids["HIGH"], list(range(200)))
        self.assertEqual(
            set(condition_ids["LOW"]),
            (set(range(200)) - set(manifest["sets"]["U_sample_ids"]))
            | set(manifest["sets"]["R_sample_ids"]),
        )
        self.assertEqual(
            set(condition_ids["PLACEBO"]),
            (set(range(200)) - set(manifest["sets"]["P_sample_ids"]))
            | set(manifest["sets"]["R_sample_ids"]),
        )

    def test_protocol_rejects_wrong_supported_count(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["sets"]["S_sample_ids"].pop()
        with self.assertRaisesRegex(ValueError, "28 unique supported"):
            RUNNER._validate_experiment_2b_protocol(manifest)

    def test_protocol_rejects_reserve_overlap_with_evaluation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["sets"]["R_sample_ids"][0] = 0
        with self.assertRaisesRegex(ValueError, "R overlaps official"):
            RUNNER._validate_experiment_2b_protocol(manifest)

    def test_initial_identity_accepts_only_all_three_shared_hashes(self):
        shared = "a" * 64
        hashes = {condition: shared for condition in RUNNER.CONDITION_NAMES}
        proof = RUNNER._assert_initial_parameter_identity(
            shared, hashes, "test preflight"
        )
        self.assertTrue(proof["passed"])
        hashes["LOW"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "LOW"):
            RUNNER._assert_initial_parameter_identity(
                shared, hashes, "test preflight"
            )

    def test_shared_initial_is_persisted_and_independently_reloaded(self):
        import numpy
        import torch

        class FakeConfig:
            @staticmethod
            def to_dict():
                return {"model_type": "fake-gpt2"}

        class FakeModel:
            def __init__(self, weight):
                self.weight = weight.clone()
                self.config = FakeConfig()

            def state_dict(self):
                return {"weight": self.weight}

            def save_pretrained(self, path, safe_serialization):
                self.assert_safe_serialization = safe_serialization
                Path(path).mkdir(parents=True)
                FakeAutoModel.saved_weight = self.weight.clone()

        class FakeAutoModel:
            saved_weight = None

            @classmethod
            def from_pretrained(cls, source):
                if source == RUNNER.MODEL_NAME:
                    return FakeModel(torch.tensor([1.0, 2.0]))
                if cls.saved_weight is None:
                    raise AssertionError("Initial state was not persisted.")
                return FakeModel(cls.saved_weight)

        class FakeTokenizer:
            @staticmethod
            def save_pretrained(path):
                (Path(path) / "tokenizer.json").write_text(
                    "{}", encoding="utf-8"
                )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initial_shared"
            shared, config, hashes = RUNNER._create_and_verify_shared_initial(
                checkpoint,
                42,
                FakeAutoModel,
                FakeTokenizer(),
                torch,
                numpy,
            )
        self.assertEqual(config, {"model_type": "fake-gpt2"})
        self.assertEqual(set(hashes), set(RUNNER.CONDITION_NAMES))
        self.assertTrue(all(value == shared for value in hashes.values()))

    def test_all_preregistered_seeds_and_output_paths(self):
        for seed in (42, 43, 44, 45, 46):
            RUNNER._validate_training_seed(seed)
            self.assertEqual(RUNNER._seed_output_root(seed).name, f"seed_{seed}")
        with self.assertRaisesRegex(ValueError, "must be preregistered"):
            RUNNER._validate_training_seed(41)

    def test_checkpoint_layout_has_shared_initial_pre_npo_and_final(self):
        paths = RUNNER._checkpoint_paths(Path("seed_42"))
        self.assertEqual(
            set(paths),
            {
                "initial_shared",
                "HIGH_pre_npo",
                "LOW_pre_npo",
                "PLACEBO_pre_npo",
                "HIGH_final",
                "LOW_final",
                "PLACEBO_final",
            },
        )
        evaluator_paths = EVALUATOR._checkpoint_paths(Path("seed_42"))
        self.assertEqual(
            {key: value.name for key, value in paths.items()},
            {key: value.name for key, value in evaluator_paths.items()},
        )

    def test_dataset_builder_injects_condition_sets_at_all_four_stages(self):
        class FakeSubset:
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

        class FakeConcat:
            def __init__(self, datasets):
                self.datasets = datasets

            def __len__(self):
                return sum(len(dataset) for dataset in self.datasets)

        target = list(range(700))
        background = list(range(15_000))
        condition_ids = {
            condition: self.manifest["conditions"][condition][
                "ordered_target_dataset_ids"
            ]
            for condition in RUNNER.CONDITION_NAMES
        }
        unlearn, targets, retains, initials = RUNNER._build_condition_datasets(
            target,
            background,
            self.manifest["evaluation_ids"]["unlearn_ids"],
            condition_ids,
            FakeSubset,
            FakeConcat,
        )
        for condition in RUNNER.CONDITION_NAMES:
            self.assertIs(initials[condition].datasets[0], targets[condition])
            self.assertIs(initials[condition].datasets[1], unlearn)
            self.assertIs(initials[condition].datasets[2], background)
            self.assertIs(retains[condition].datasets[0], targets[condition])
            self.assertIs(retains[condition].datasets[1], background)
            self.assertEqual(len(initials[condition]), 15_400)
            self.assertEqual(len(retains[condition]), 15_200)

    def test_orchestrator_includes_all_five_seeds(self):
        self.assertEqual(ORCHESTRATOR.SEEDS, (42, 43, 44, 45, 46))

    def test_successful_stage_cleanup_is_scoped_and_records_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "trainer_work"
            stage = scratch / "HIGH" / "initial_sft"
            stage.mkdir(parents=True)
            payload = b"trainer-state"
            (stage / "optimizer.pt").write_bytes(payload)
            record = RUNNER._cleanup_successful_trainer_stage(stage, scratch)
            self.assertFalse(stage.exists())
            self.assertTrue(record["removed"])
            self.assertEqual(record["removed_bytes"], len(payload))

            unexpected = scratch / "HIGH" / "unknown_stage"
            unexpected.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "unexpected Trainer stage"):
                RUNNER._cleanup_successful_trainer_stage(unexpected, scratch)
            self.assertTrue(unexpected.exists())

    def test_completed_seed_cleanup_preserves_durable_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_root = Path(directory) / "seed_42"
            for marker in ORCHESTRATOR.TRAINING_MARKERS:
                path = seed_root / marker
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("marker", encoding="utf-8")
            scratch_file = seed_root / "trainer_work" / "HIGH" / "optimizer.pt"
            scratch_file.parent.mkdir(parents=True)
            scratch_file.write_bytes(b"scratch")

            self.assertTrue(
                ORCHESTRATOR._cleanup_completed_trainer_work(seed_root)
            )
            self.assertFalse((seed_root / "trainer_work").exists())
            self.assertTrue(
                all(
                    (seed_root / marker).is_file()
                    for marker in ORCHESTRATOR.TRAINING_MARKERS
                )
            )

    def test_completed_seed_cleanup_rejects_partial_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_root = Path(directory) / "seed_43"
            scratch = seed_root / "trainer_work"
            scratch.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "markers are incomplete"):
                ORCHESTRATOR._cleanup_completed_trainer_work(seed_root)
            self.assertTrue(scratch.exists())

    def test_environment_pins_support_frozen_list_schema(self):
        setup = (REPOSITORY_ROOT / "scripts" / "setup_ruli_env.sh").read_text(
            encoding="utf-8"
        )
        pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('"datasets==5.0.1"', setup)
        self.assertIn('"pyarrow==21.0.0"', setup)
        self.assertNotIn('"datasets==2.21.0"', setup)
        self.assertNotIn('"pyarrow==17.0.0"', setup)
        self.assertIn('"datasets>=5.0.1"', pyproject)
        self.assertIn('"pyarrow>=21.0.0"', pyproject)

    def test_evaluator_accepts_divergent_post_npo_hashes(self):
        shared = "a" * 64
        manifest = self.manifest
        payload = {
            "experiment": "2B",
            "seed": 42,
            "manifest": {
                "declared_canonical_content_sha256": manifest["manifest_hash"][
                    "sha256"
                ]
            },
            "model_and_hyperparameters": {
                "model": "gpt2",
                "initial_sft_epochs": 5,
                "prefix_epochs": 1,
                "unlearn_method": "npo",
                "npo_epochs": 15,
                "final_retain_sft_epochs": 2,
                "attack_size": 15_000,
            },
            "background_dataset": {
                "selection_seed": 42,
                "count": 15_000,
                "membership_sha256": manifest["shared_wikitext_background"][
                    "membership_sha256"
                ],
            },
            "ordered_target_dataset_ids": {
                condition: manifest["conditions"][condition][
                    "ordered_target_dataset_ids"
                ]
                for condition in RUNNER.CONDITION_NAMES
            },
            "dataset_sizes": {
                "initial_sft_and_prefix": {
                    condition: 15_400 for condition in RUNNER.CONDITION_NAMES
                },
                "npo_forget": 200,
                "npo_retain": {
                    condition: 15_200 for condition in RUNNER.CONDITION_NAMES
                },
                "final_retain": {
                    condition: 15_200 for condition in RUNNER.CONDITION_NAMES
                },
            },
            "starting_parameter_identity": {
                "passed": True,
                "sha256": shared,
                "preflight_reload_sha256": {
                    condition: shared for condition in RUNNER.CONDITION_NAMES
                },
                "actual_training_branch_sha256": {
                    condition: shared for condition in RUNNER.CONDITION_NAMES
                },
            },
            "checkpoints": {
                "initial_shared": {"parameter_sha256": shared},
                **{
                    condition: {
                        "starting_parameter_sha256": shared,
                        "pre_npo": {"parameter_sha256": str(index) * 64},
                        "post_npo_parameter_sha256": str(index + 3) * 64,
                        "final": {"parameter_sha256": str(index + 6) * 64},
                    }
                    for index, condition in enumerate(RUNNER.CONDITION_NAMES, 1)
                },
            },
            "validation_results": {"passed": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_metadata.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            result = EVALUATOR._validate_training_run_metadata(
                root, 42, manifest
            )
        self.assertFalse(result["post_npo_parameter_equality_required"])


if __name__ == "__main__":
    unittest.main()
