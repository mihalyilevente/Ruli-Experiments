"""CPU regressions using synthetic models/measurements, never experiment results."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "experiment_2"
SPEC = importlib.util.spec_from_file_location("evaluate_2c", SCRIPTS / "evaluate_experiment_2c.py")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)
RUNNER = EVALUATOR.RUNNER
MANIFEST, _ = RUNNER.REFERENCE._validate_manifest(SCRIPTS / "results" / "intervention_manifest.json")


def measurements():
    ids = MANIFEST["evaluation_ids"]["unlearn_ids"]
    return {
        c: {stage: {i: {"loss": pre + (delta if stage == "post_npo" else 0),
                         "privacy_log_odds": 0.5, "privacy_score": 0.6}
                    for i in ids}
            for stage in RUNNER.CAPTURE}
        for c, pre, delta in (("HIGH", 2., 1.), ("LOW", 9., 4.), ("PLACEBO", 3., 2.))
    }


def metadata():
    shared = "a" * 64
    return {
        "experiment": "2C", "seed": 42,
        "manifest": {"declared_canonical_content_sha256": MANIFEST["manifest_hash"]["sha256"]},
        "model_and_hyperparameters": dict(RUNNER.HYPERPARAMETERS),
        "ordered_unlearn_ids": MANIFEST["evaluation_ids"]["unlearn_ids"],
        "ordered_target_dataset_ids": {
            c: MANIFEST["conditions"][c]["ordered_target_dataset_ids"] for c in RUNNER.CONDITIONS
        },
        "background_dataset": {"selection_seed": 42, "count": 15_000,
                               "membership_sha256": MANIFEST["shared_wikitext_background"]["membership_sha256"]},
        "dataset_sizes": {
            "npo_forget": 200,
            **{field: {c: size for c in RUNNER.CONDITIONS} for field, size in
               (("condition_target", 200), ("initial_sft_and_prefix", 15_400), ("npo_retain", 15_200))},
        },
        "starting_parameter_identity": {
            "passed": True, "sha256": shared,
            "preflight_reload_sha256": {c: shared for c in RUNNER.CONDITIONS},
            "actual_training_branch_sha256": {c: shared for c in RUNNER.CONDITIONS},
        },
        "checkpoints": {
            "initial_shared": {"parameter_sha256": shared},
            **{c: {"starting_parameter_sha256": shared, "stage_order": list(RUNNER.STAGES),
                   **{stage: {"capture": capture, "final_retain_ft_updates": 0,
                              "parameter_sha256": str(index + offset) * 64}
                      for offset, (stage, capture) in enumerate(RUNNER.CAPTURE.items())}}
               for index, c in enumerate(RUNNER.CONDITIONS, 1)},
        },
        "validation_results": {key: True for key in (
            "frozen_artifacts", "last_7_loss_behavior", "shared_initial_parameter_identity",
            "checkpoint_capture_order", "no_final_retain_ft")},
    }


class Experiment2CTests(unittest.TestCase):
    def test_frozen_manifest_memberships_and_output_isolation(self):
        _, checks = RUNNER.BASE._validate_experiment_2b_protocol(MANIFEST)
        self.assertTrue(all(checks.values()))
        self.assertEqual(len(RUNNER._checkpoint_paths(Path("seed_42"))), 7)
        with self.assertRaisesRegex(ValueError, "2A or 2B"):
            RUNNER._output_root(Path("results/experiment_2b/seed_42"), 42)

    def test_branch_captures_exact_boundaries_and_never_runs_final_ft(self):
        import torch

        class Model:
            def __init__(self):
                self.weight = torch.tensor([0.])

            def state_dict(self):
                return {"weight": self.weight}

        model = Model()
        initial, retain, unlearn, valid, tokenizer = (object() for _ in range(5))
        calls, saved = [], {}

        def sft(m, data, validation, tok, epochs):
            self.assertIs(data, initial)
            self.assertIs(validation, valid)
            self.assertIs(tok, tokenizer)
            self.assertEqual(epochs, 5)
            calls.append("sft")
            m.weight += 1
            return m

        def prefix(m, data, validation, tok, epochs):
            self.assertIs(data, initial)
            self.assertEqual(epochs, 1)
            calls.append("prefix")
            m.weight += 2
            return m

        def npo(m, forget, remain, validation, tok, args):
            self.assertIs(forget, unlearn)
            self.assertIs(remain, retain)
            self.assertEqual(args.unlearn_method, "npo")
            self.assertEqual(args.unlearn_epochs, 15)
            self.assertEqual(saved["LOW_pre_npo"], 3.)
            calls.append("npo")
            m.weight += 4
            return m

        def save(m, tok, path):
            saved[path.name] = m.weight.item()

        utils = SimpleNamespace(train_sft=sft, train_prefix=prefix, unlearn_model=npo)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(RUNNER.COMMON, "_save_checkpoint", side_effect=save):
                record, cleanup = RUNNER._run_branch(
                    model, tokenizer, initial, retain, unlearn, valid, "LOW",
                    RUNNER._checkpoint_paths(root), root / "trainer_work", utils, "cpu", torch,
                )
            self.assertFalse((root / "trainer_work").exists())
        self.assertEqual(calls, ["sft", "prefix", "npo"])
        self.assertEqual(saved, {"LOW_pre_npo": 3., "LOW_post_npo": 7.})
        self.assertEqual(record["stage_order"], list(RUNNER.STAGES))
        self.assertEqual(set(cleanup), {"initial_sft", "prefix", "npo"})
        self.assertNotEqual(record["pre_npo"]["parameter_sha256"], record["post_npo"]["parameter_sha256"])

    def test_paired_loss_did_and_cohort_outputs(self):
        ids = MANIFEST["evaluation_ids"]["unlearn_ids"]
        supported = MANIFEST["sets"]["S_sample_ids"]
        negative = MANIFEST["sets"]["negative_control_sample_ids"]
        rows, wide = EVALUATOR._paired_rows(measurements(), ids, supported, negative, 42)
        self.assertEqual(len(rows), 600)
        self.assertEqual(sum(r["is_supported_S"] for r in rows), 28 * 3)
        self.assertEqual(sum(r["is_negative_control"] for r in rows), 121 * 3)
        self.assertEqual(wide[0]["LOW_minus_PLACEBO_delta"], 2.)
        self.assertEqual(wide[0]["HIGH_minus_PLACEBO_delta"], -1.)
        supported_summary = EVALUATOR._cohort_summary(wide, supported)
        self.assertEqual(supported_summary["n"], 28)
        self.assertEqual(supported_summary["LOW_minus_PLACEBO_delta"], {
            "mean": 2., "median": 2., "number_positive": 28, "fraction_positive": 1.})
        self.assertEqual(EVALUATOR._cohort_summary(wide, negative)["n"], 121)
        by_id = {r["sample_id"]: r for r in wide}
        primary = [by_id[i] for i in supported]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation"
            EVALUATOR._write_outputs(output, rows, primary, supported_summary)
            self.assertEqual(set(p.name for p in output.iterdir()), set(EVALUATOR.OUTPUT_FILENAMES))
            import csv
            with (output / "primary_contrast.csv").open() as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual([int(r["sample_id"]) for r in exported], supported)
            with self.assertRaises(FileExistsError):
                EVALUATOR._write_outputs(output, rows, primary, supported_summary)

    def test_rejects_reordered_missing_and_nonfinite_measurements(self):
        ids = MANIFEST["evaluation_ids"]["unlearn_ids"]
        for mutation in ("reorder", "missing", "nonfinite"):
            values = measurements()
            samples = values["LOW"]["post_npo"]
            if mutation == "reorder":
                values["LOW"]["post_npo"] = dict(reversed(list(samples.items())))
            elif mutation == "missing":
                samples.pop(ids[0])
            else:
                samples[ids[0]]["loss"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, FloatingPointError)):
                EVALUATOR._paired_rows(values, ids, set(), set(), 42)

    def test_metadata_rejects_stage_ft_seed_membership_and_initial_mismatches(self):
        original = metadata()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def validate(payload):
                (root / "run_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
                return EVALUATOR._validate_training_run_metadata(root, 42, MANIFEST)

            validate(original)  # Divergent condition pre/post hashes are expected.
            for mutation in ("experiment", "seed", "epochs", "capture", "order", "identity", "ids", "background", "ft"):
                payload = copy.deepcopy(original)
                if mutation in ("experiment", "seed"):
                    payload[mutation] = "wrong"
                elif mutation == "epochs":
                    payload["model_and_hyperparameters"]["final_retain_sft_epochs"] = 2
                elif mutation == "capture":
                    payload["checkpoints"]["LOW"]["post_npo"]["capture"] = "after_final_ft"
                elif mutation == "order":
                    payload["checkpoints"]["LOW"]["stage_order"].reverse()
                elif mutation == "identity":
                    payload["starting_parameter_identity"]["actual_training_branch_sha256"]["LOW"] = "b" * 64
                elif mutation == "ids":
                    payload["ordered_unlearn_ids"].reverse()
                elif mutation == "background":
                    payload["background_dataset"]["selection_seed"] = 43
                else:
                    payload["checkpoints"]["LOW"]["post_npo"]["final_retain_ft_updates"] = 1
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                    validate(payload)

    def test_original_and_unlearned_kdes_use_correct_distributions(self):
        shadow = {name: {201: values} for name, values in (
            ("unlearn_original", [1., 2., 3.]), ("out_original", [4., 5., 6.]),
            ("unlearn_unlearned", [7., 8., 9.]), ("out_unlearned", [10., 11., 12.]))}
        kdes = RUNNER._kde_references(shadow, [201], tuple)
        self.assertEqual(kdes["pre_npo"][201], ((1., 2., 3.), (4., 5., 6.)))
        self.assertEqual(kdes["post_npo"][201], ((7., 8., 9.), (10., 11., 12.)))

    def test_missing_or_invalid_original_shadow_observations_fail(self):
        fake_torch = SimpleNamespace(load=lambda *args, **kwargs: {})
        with patch.object(RUNNER.REFERENCE, "_load_and_validate_shadow", return_value=({}, {})):
            with self.assertRaises(ValueError):
                RUNNER._load_shadow(Path("synthetic"), {}, MANIFEST, fake_torch)
            raw = {"unlearn_original": {i: [1., 2., 3.] for i in MANIFEST["evaluation_ids"]["unlearn_ids"]}}
            fake_torch.load = lambda *args, **kwargs: raw
            shadow, details = RUNNER._load_shadow(Path("synthetic"), {}, MANIFEST, fake_torch)
            self.assertEqual(details["original_unlearn_observations_validated"], 200)
            self.assertEqual(len(shadow["unlearn_original"]), 200)
            raw["unlearn_original"][200] = [1., 2.]
            with self.assertRaises(ValueError):
                RUNNER._load_shadow(Path("synthetic"), {}, MANIFEST, fake_torch)

    def test_loaded_checkpoint_hash_mismatch_fails(self):
        with patch.object(RUNNER.COMMON, "_parameter_sha256", return_value="a" * 64):
            EVALUATOR._verify_model_hash(None, "a" * 64, "test", None)
            with self.assertRaisesRegex(ValueError, "loaded parameter hash"):
                EVALUATOR._verify_model_hash(None, "b" * 64, "test", None)

    def test_evaluator_main_writes_all_outputs_with_six_aligned_passes(self):
        """Synthetic inference only; real frozen manifest, pairing, and output code."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        class Model:
            def __init__(self, path):
                self.label = Path(path).name

            def state_dict(self):
                return {"weight": torch.tensor([1.])}

            def to(self, device):
                return self

        ids = MANIFEST["evaluation_ids"]["unlearn_ids"]
        digest = RUNNER.COMMON._parameter_sha256(Model("synthetic"), torch)
        payload = metadata()
        payload["starting_parameter_identity"]["sha256"] = digest
        for field in ("preflight_reload_sha256", "actual_training_branch_sha256"):
            payload["starting_parameter_identity"][field] = {
                c: digest for c in RUNNER.CONDITIONS
            }
        payload["checkpoints"]["initial_shared"]["parameter_sha256"] = digest
        for condition in RUNNER.CONDITIONS:
            payload["checkpoints"][condition]["starting_parameter_sha256"] = digest
            for stage in RUNNER.CAPTURE:
                payload["checkpoints"][condition][stage]["parameter_sha256"] = digest
        sources = {"utils.py": {"sha256": "f" * 64}}
        payload["upstream_ruli_source_files"] = sources
        calls, token_objects = [], []

        def inference(model, dataset, tokenizer, device, utils, sample_ids, tokens):
            self.assertEqual(list(sample_ids), ids)
            self.assertTrue(all(isinstance(tokens[i], tuple) for i in ids))
            calls.append(model.label)
            token_objects.append(tokens)
            condition = model.label.split("_")[0]
            pre, delta = {"HIGH": (2., 1.), "LOW": (9., 4.), "PLACEBO": (3., 2.)}[condition]
            return [pre + (delta if "post_npo" in model.label else 0)] * len(ids)

        class KDE:
            def __init__(self, values):
                pass

            def evaluate(self, values):
                return [0.5]

            def logpdf(self, values):
                return [-0.7]

        shadow = {field: {i: [1., 2., 3.] for i in ids} for field in (
            "unlearn_original", "out_original", "unlearn_unlearned", "out_unlearned"
        )}
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "experiment_2c" / "seed_42"
            root.mkdir(parents=True)
            (root / "run_metadata.json").write_text(json.dumps(payload), encoding="utf-8")
            args = SimpleNamespace(
                seed=42, manifest=SCRIPTS / "results" / "intervention_manifest.json",
                experiment_output=root, ruli_root=Path(directory),
                shadow_path=None, target_data_path=None, device="cpu", validate_only=False,
            )
            patches = (
                (EVALUATOR, "parse_args", {"return_value": args}),
                (RUNNER, "_source_metadata", {"return_value": sources}),
                (RUNNER, "_load_shadow", {"return_value": (shadow, {})}),
                (RUNNER.COMMON, "_verify_file_artifact", {"return_value": {}}),
                (RUNNER.COMMON, "_verify_target_dataset_storage", {"return_value": {}}),
                (RUNNER.COMMON, "_load_ruli_utils", {"return_value": SimpleNamespace(gaussian_kde=KDE)}),
                (EVALUATOR.REFERENCE, "_checkpoint_metadata", {"return_value": {}}),
                (EVALUATOR.REFERENCE, "_validate_reference_loss_path", {"return_value": {"synthetic": True}}),
                (EVALUATOR.REFERENCE, "_validate_target_dataset", {"return_value": (
                    None, {"synthetic": True}, {i: list(range(10)) for i in ids})}),
                (EVALUATOR.REFERENCE, "_run_reference_inference", {"side_effect": inference}),
                (AutoModelForCausalLM, "from_pretrained", {"side_effect": Model}),
                (AutoTokenizer, "from_pretrained", {"return_value": None}),
            )
            for obj, name, kwargs in patches:
                stack.enter_context(patch.object(obj, name, **kwargs))
            EVALUATOR.main()
            summary = json.loads((root / "evaluation" / "evaluation_summary.json").read_text())
            self.assertEqual(summary["sample_counts"]["per_sample_rows"], 600)
            self.assertEqual(summary["sample_counts"]["primary_contrast_rows"], 28)
            self.assertEqual(summary["primary_descriptive_results"]["supported_S"][
                "LOW_minus_PLACEBO_delta"]["mean"], 2.)
            self.assertEqual(summary["alignment"]["ordered_unlearn_ids"], ids)
        self.assertEqual(calls, [f"{c}_{stage}" for c in RUNNER.CONDITIONS
                                 for stage in RUNNER.CAPTURE])
        self.assertTrue(all(tokens is token_objects[0] for tokens in token_objects))


if __name__ == "__main__":
    unittest.main()
