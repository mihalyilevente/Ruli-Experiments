"""CPU synthetic regressions; no training results or substituted frozen artifacts."""

import copy
import importlib.util
import json
import random
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "experiment_2"
SPEC = importlib.util.spec_from_file_location("eval_2d_test", SCRIPTS / "evaluate_experiment_2d.py")
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)
RUN = EVAL.RUNNER
MANIFEST, _ = RUN.REFERENCE._validate_manifest(SCRIPTS / "results" / "intervention_manifest.json")
IDS = MANIFEST["evaluation_ids"]["unlearn_ids"]
SUPPORTED = MANIFEST["sets"]["S_sample_ids"]
NEGATIVE = MANIFEST["sets"]["negative_control_sample_ids"]


class Model(torch.nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.]))
        self.child = torch.nn.Dropout()
        self.config = SimpleNamespace(to_dict=lambda: {"synthetic": True})
        self.label = Path(path).name if path else "synthetic"
        if path and (Path(path) / "synthetic.pt").exists():
            self.load_state_dict(torch.load(Path(path) / "synthetic.pt", weights_only=True))

    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True)
        torch.save(self.state_dict(), Path(path) / "synthetic.pt")


class Tokenizer:
    eos_token = "eos"

    def save_pretrained(self, path):
        pass


def measurements():
    return {c: {stage: {i: {"loss": values[n], "privacy_score": 0.6, "privacy_log_odds": 0.5}
                            for i in IDS}
                for n, stage in enumerate(RUN.CAPTURE)}
            for c, values in (("HIGH", (2., 5., 3.)), ("LOW", (9., 13., 12.)), ("PLACEBO", (3., 5., 1.)))}


def write_ft_state(epoch=2):
    path = Path("completed_trainer_state.json")
    path.write_text(json.dumps({"epoch": epoch, "num_train_epochs": 2, "global_step": 2,
                                "best_model_checkpoint": "checkpoint-1"}))


class Experiment2DTests(unittest.TestCase):
    def test_frozen_manifest_output_isolation_and_immutable_tokens(self):
        self.assertEqual(len(RUN._checkpoint_paths(Path("seed_42"))), 10)
        for experiment in ("2a", "2b", "2c"):
            with self.assertRaises(ValueError):
                RUN._output_root(Path(f"results/experiment_{experiment}/seed_42"), 42)
        with self.assertRaises(ValueError):
            RUN._output_root(Path("seed_43"), 42)
        tokens = RUN._immutable_tokens({2: [1, 2, 3], 1: [3, 2, 1]}, [2, 1])
        with self.assertRaises(TypeError):
            tokens[2] = (1,)
        with self.assertRaises(TypeError):
            tokens[2][0] = 7
        self.assertNotEqual(RUN._token_digest(tokens, [2, 1]), RUN._token_digest(tokens, [1, 2]))

    def test_three_stage_did_signs_and_cohorts(self):
        rows = EVAL._paired_rows(measurements(), IDS, SUPPORTED, NEGATIVE, 42)
        self.assertEqual(len(rows), 200)
        row = rows[0]
        self.assertEqual((row["NPO_DiD_LOW_PLACEBO"], row["FT_DiD_LOW_PLACEBO"], row["TOTAL_DiD_LOW_PLACEBO"]), (2, 3, 5))
        self.assertEqual(row["post_ft_loss_LOW_minus_PLACEBO"], 11)
        self.assertEqual((row["NPO_DiD_HIGH_PLACEBO"], row["FT_DiD_HIGH_PLACEBO"], row["TOTAL_DiD_HIGH_PLACEBO"]), (1, 2, 3))
        for ids, count in ((SUPPORTED, 28), (NEGATIVE, 121)):
            summary = EVAL._cohort_summary(rows, ids)
            self.assertEqual(summary["n"], count)
            self.assertEqual(summary["contrasts"]["FT_DiD_LOW_PLACEBO"]["mean"], 3)
            self.assertEqual(summary["contrasts"]["FT_DiD_LOW_PLACEBO"]["fraction_in_predicted_direction"], 1)
        # A zero or negative contrast must not count in the predicted direction.
        indexed = {r["sample_id"]: r for r in rows}
        indexed[SUPPORTED[0]]["FT_DiD_LOW_PLACEBO"] = 0
        indexed[SUPPORTED[1]]["FT_DiD_LOW_PLACEBO"] = -1
        self.assertEqual(EVAL._cohort_summary(rows, SUPPORTED)["contrasts"]["FT_DiD_LOW_PLACEBO"]["number_positive"], 26)

    def test_rejects_alignment_nonfinite_and_boundary_loss_changes(self):
        for mutation in ("missing", "extra", "reorder", "stage", "loss", "score", "overflow"):
            values = measurements()
            samples = values["LOW"]["post_ft"]
            if mutation == "missing":
                samples.pop(IDS[0])
            elif mutation == "extra":
                samples[-1] = samples[IDS[0]]
            elif mutation == "reorder":
                values["LOW"]["post_ft"] = dict(reversed(list(samples.items())))
            elif mutation == "stage":
                values["LOW"].pop("post_ft")
            elif mutation == "loss":
                samples[IDS[0]]["loss"] = float("nan")
            elif mutation == "score":
                samples[IDS[0]]["privacy_score"] = float("inf")
            else:
                samples[IDS[0]]["loss"] = 1e308
                values["LOW"]["post_npo"][IDS[0]]["loss"] = -1e308
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, FloatingPointError)):
                EVAL._paired_rows(values, IDS, SUPPORTED, NEGATIVE, 42)
        with self.assertRaises(ValueError):
            EVAL._verify_boundary_losses([3.], [1.], "test")
        EVAL._verify_boundary_losses([1.000001], [1.], "test")

    def test_kde_references_keep_original_and_unlearned_states_separate(self):
        shadow = {name: {201: values} for name, values in (
            ("unlearn_original", [1, 2, 3]), ("out_original", [4, 5, 6]),
            ("unlearn_unlearned", [7, 8, 9]), ("out_unlearned", [10, 11, 12]))}
        result = RUN._kde_references(shadow, [201], tuple)
        self.assertEqual(result["pre_npo"][201], ((1, 2, 3), (4, 5, 6)))
        self.assertEqual(result["post_ft"][201], ((7, 8, 9), (10, 11, 12)))
        self.assertIs(result["post_npo"], result["post_ft"])

    def test_actual_upstream_last_seven_loss_behavior(self):
        import ast

        source = SCRIPTS.parents[2] / "Ruli/text/utils.py"
        if not source.is_file():
            self.skipTest("Requires the sibling upstream Ruli loss implementation.")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name in ("MIAEvaluator", "EfficacyEvaluator")]
        namespace = {"torch": torch, "tqdm": lambda items, **kwargs: items}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
        proof = RUN.REFERENCE._validate_reference_loss_path(SimpleNamespace(
            MIAEvaluator=namespace["MIAEvaluator"], EfficacyEvaluator=namespace["EfficacyEvaluator"],
        ), torch)
        self.assertEqual(proof["behavioral_check"], "passed")
        self.assertTrue(proof["efficacy_function_ast_identical"])

    def test_measurement_restores_rng_modes_and_rejects_parameter_changes(self):
        model = Model()
        model.child.eval()  # Preserve mixed submodule modes, too.
        tokens = RUN._immutable_tokens({1: list(range(10))}, [1])
        RUN.COMMON._reset_rng(42, torch, np)
        expected_draws = (random.random(), np.random.random(), torch.rand(1))
        RUN.COMMON._reset_rng(42, torch, np)

        def inference(*args):
            model.eval()
            random.random(), np.random.random(), torch.rand(10)
            return [2.]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurement.json"
            digest = RUN.COMMON._parameter_sha256(model, torch)
            with patch.object(RUN.REFERENCE, "_run_reference_inference", side_effect=inference):
                proof = RUN._measure_boundary(model, None, None, "cpu", None, [1], tokens,
                                              torch, np, digest, "LOW", "pre_npo", 42, path)
            self.assertEqual(proof["file_sha256"], RUN.COMMON._sha256_file(path))
            self.assertTrue(model.training)
            self.assertFalse(model.child.training)
            self.assertEqual(random.random(), expected_draws[0])
            self.assertEqual(np.random.random(), expected_draws[1])
            self.assertTrue(torch.equal(torch.rand(1), expected_draws[2]))

            def mutate(*args):
                model.weight.data += 1
                return [2.]

            with patch.object(RUN.REFERENCE, "_run_reference_inference", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed checkpoint parameters"):
                    RUN._measure_boundary(model, None, None, "cpu", None, [1], tokens,
                                          torch, np, digest, "LOW", "post_npo", 42, path)

    def test_ft_epoch_evidence_fails_closed(self):
        for epoch in (None, 1, 1.5, 3, float("nan")):
            with self.subTest(epoch=epoch), tempfile.TemporaryDirectory() as directory:
                with RUN.COMMON._working_directory(Path(directory)):
                    write_ft_state(epoch)
                    with self.assertRaises(ValueError):
                        RUN._verify_final_ft_epochs(Path(directory))

    def test_real_cpu_trainer_completion_survives_best_checkpoint_pruning(self):
        from transformers import Trainer, TrainingArguments

        class Regression(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.]))

            def forward(self, input_ids, labels=None):
                logits = input_ids.float() * self.weight
                return {"loss": ((logits - labels) ** 2).mean(), "logits": logits}

        evaluations = []

        def metrics(prediction):
            evaluations.append(1)
            return {"fixture_metric": len(evaluations)}  # First epoch is best.

        data = [{"input_ids": torch.tensor([1.]), "labels": torch.tensor([0.])}]
        utils = SimpleNamespace(Trainer=Trainer)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            args = TrainingArguments(
                output_dir=str(work / "output_sft"), use_cpu=True,
                num_train_epochs=2, per_device_train_batch_size=1,
                evaluation_strategy="epoch", save_strategy="epoch",
                load_best_model_at_end=True, save_total_limit=1,
                metric_for_best_model="fixture_metric", greater_is_better=False,
                report_to="none", disable_tqdm=True,
            )
            with RUN._record_final_ft_completion(utils, work / "completed_trainer_state.json"):
                trainer = utils.Trainer(model=Regression(), args=args, train_dataset=data,
                                        eval_dataset=data, compute_metrics=metrics)
                trainer.train()
            proof = RUN._verify_final_ft_epochs(work)
            self.assertEqual(proof["completed_epochs"], 2)
            self.assertEqual(proof["trainer_states"][0]["global_step"], 2)
            self.assertTrue((work / "output_sft/checkpoint-1").exists())
            self.assertFalse((work / "output_sft/checkpoint-2").exists())
            self.assertIs(utils.Trainer, Trainer)

    def test_failed_post_npo_measurement_prevents_final_ft(self):
        calls = []

        def sft(model, *args):
            calls.append(args[-1])
            return model

        def measure(model, c, stage, digest):
            if stage == "post_npo":
                raise FloatingPointError("synthetic failed measurement")
            return {}

        utils = SimpleNamespace(train_sft=sft, train_prefix=lambda model, *a: model,
                                unlearn_model=lambda model, *a: model)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FloatingPointError):
                RUN._run_branch(Model(), Tokenizer(), None, None, None, None, "LOW",
                                RUN._checkpoint_paths(root), root / "trainer_work", utils, "cpu", torch, measure)
            self.assertEqual(calls, [5])
            self.assertTrue((root / "LOW_post_npo").exists())
            self.assertFalse((root / "LOW_post_ft").exists())
            self.assertTrue((root / "trainer_work/LOW/npo").exists())

    def test_cpu_synthetic_runner_evaluator_and_metadata_tampering(self):
        """Exercise both mains and real 2B privacy aggregation with tiny fake models."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.trainer_callback import TrainerState
        from scipy.stats import gaussian_kde
        from sklearn.metrics import accuracy_score, auc, roc_curve

        out_ids = MANIFEST["evaluation_ids"]["out_ids"]
        all_ids = IDS + out_ids
        raw_tokens = {i: list(range(10)) for i in all_ids}
        shadow = {field: {i: [1., 3., 6.] for i in all_ids} for field in (
            "unlearn_original", "out_original", "unlearn_unlearned", "out_unlearned")}
        sources = {"utils.py": {"sha256": "f" * 64}}
        background = {"selection_seed": 42, "count": 15_000,
                      "membership_sha256": MANIFEST["shared_wikitext_background"]["membership_sha256"]}
        events, evaluated_tokens, initial_draws = [], [], []

        # Actual upstream scoring method, loaded without importing unrelated RULI dependencies.
        import ast
        source_path = SCRIPTS.parents[2] / "Ruli" / "text" / "utils.py"
        if not source_path.is_file():
            self.skipTest("Requires the sibling upstream Ruli checkout for the actual KDE method.")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        mia = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MIAEvaluator")
        score_method = next(node for node in mia.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate_with_kde")
        namespace = {"np": np, "gaussian_kde": gaussian_kde, "roc_curve": roc_curve,
                     "auc": auc, "accuracy_score": accuracy_score}
        exec(compile(ast.Module(body=[score_method], type_ignores=[]), str(source_path), "exec"), namespace)

        class MIA:
            def __init__(self, **kwargs):
                self.args = kwargs["args"]

            evaluate_with_kde = namespace["evaluate_with_kde"]

        def inference(model, dataset, tokenizer, device, utils, ids, tokens):
            evaluated_tokens.append(tokens)
            self.assertTrue(all(isinstance(tokens[i], tuple) for i in ids))
            model.eval()
            random.random(), np.random.random(), torch.rand(2)
            return [float(model.weight.item()) + (i % 3) / 10 for i in ids]

        def sft(model, data, valid, tok, epochs):
            stage = "initial_sft" if epochs == 5 else "final_retain_ft"
            events.append((Path.cwd().parent.name, stage))
            self.assertEqual(len(data), 15_400 if epochs == 5 else 15_200)
            if epochs == 5:
                initial_draws.append((random.random(), np.random.random(), torch.rand(1).item()))
            else:
                c = Path.cwd().parent.name
                measurement_path = Path.cwd().parents[2] / "measurements" / f"{c}_post_npo.json"
                self.assertTrue(measurement_path.exists())
                trainer = utils.Trainer()
                trainer.callback.on_train_end(
                    SimpleNamespace(num_train_epochs=epochs),
                    TrainerState(epoch=2, num_train_epochs=2, global_step=2,
                                 best_model_checkpoint="checkpoint-1"), None,
                )
            model.weight.data += 1 if epochs == 5 else -0.5
            return model

        def prefix(model, data, valid, tok, epochs):
            self.assertEqual((len(data), epochs), (15_400, 1))
            model.weight.data += 1
            return model

        def npo(model, forget, retain, valid, tok, args):
            self.assertEqual((len(forget), len(retain), args.unlearn_epochs, args.unlearn_method), (200, 15_200, 15, "npo"))
            model.weight.data += 1
            return model

        class Background(list):
            def shuffle(self, seed):
                self.seed = seed
                return self

            def select(self, indices):
                self.assertion = len(indices)
                return self

        class Trainer:
            def add_callback(self, callback):
                self.callback = callback

        utils = SimpleNamespace(train_sft=sft, train_prefix=prefix, unlearn_model=npo, Trainer=Trainer,
                                load_data=lambda *a: (Background([0] * 15_000), None, None),
                                gaussian_kde=gaussian_kde, np=np, roc_curve=roc_curve, auc=auc,
                                accuracy_score=accuracy_score, MIAEvaluator=MIA)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory) / "experiment_2d" / "seed_42"
            ruli_root = Path(directory) / "Ruli"
            args = SimpleNamespace(seed=42, manifest=SCRIPTS / "results/intervention_manifest.json",
                ruli_root=ruli_root, shadow_path=None, target_data_path=None, output_root=root,
                experiment_output=root, device="cpu", validate_only=False, validate_manifest_only=False)
            patches = (
                (RUN, "parse_args", {"return_value": args}), (EVAL, "parse_args", {"return_value": args}),
                (RUN.PREVIOUS, "_source_metadata", {"return_value": sources}),
                (RUN.PREVIOUS, "_load_shadow", {"return_value": (shadow, {"synthetic": True})}),
                (RUN.COMMON, "_verify_file_artifact", {"return_value": {}}),
                (RUN.COMMON, "_verify_target_dataset_storage", {"return_value": {}}),
                (RUN.COMMON, "_load_ruli_utils", {"return_value": utils}),
                (RUN.COMMON, "_validate_background_dataset", {"return_value": background}),
                (RUN.COMMON, "_configure_training_arguments_seed", {"return_value": None}),
                (RUN.REFERENCE, "_validate_reference_loss_path", {"return_value": {"synthetic": True}}),
                (RUN.REFERENCE, "_validate_target_dataset", {"return_value": ([0] * 700, {"synthetic": True}, raw_tokens)}),
                (RUN.REFERENCE, "_run_reference_inference", {"side_effect": inference}),
                (RUN.REFERENCE, "_checkpoint_metadata", {"return_value": {"synthetic": True}}),
                (AutoModelForCausalLM, "from_pretrained", {"side_effect": Model}),
                (AutoTokenizer, "from_pretrained", {"return_value": Tokenizer()}),
            )
            for obj, field, kwargs in patches:
                stack.enter_context(patch.object(obj, field, **kwargs))
            RUN.main()
            self.assertEqual(initial_draws, [initial_draws[0]] * 3)
            payload = json.loads((root / "run_metadata.json").read_text())
            self.assertFalse((root / "trainer_work").exists())
            for c in RUN.CONDITIONS:
                self.assertEqual(payload["checkpoints"][c]["stage_order"], list(RUN.STAGES))
            EVAL.main()
            summary = json.loads((root / "evaluation/evaluation_summary.json").read_text())
            self.assertEqual(summary["sample_counts"]["per_sample_rows"], 200)
            self.assertEqual(summary["sample_counts"]["post_ft_ruli_rows"], 1200)
            self.assertEqual(summary["primary_descriptive_results"]["negative_controls"]["n"], 121)
            self.assertEqual(set(p.name for p in (root / "evaluation").iterdir()), set(EVAL.OUTPUT_FILENAMES))
            import csv
            with (root / "evaluation/primary_contrast.csv").open() as stream:
                self.assertEqual([int(r["sample_id"]) for r in csv.DictReader(stream)], SUPPORTED)
            self.assertEqual(len(evaluated_tokens), 18)  # nine synchronous + nine checkpoint passes
            self.assertTrue(all(t is evaluated_tokens[0] for t in evaluated_tokens[:9]))
            self.assertTrue(all(t is evaluated_tokens[9] for t in evaluated_tokens[9:]))
            args.validate_only = True
            RUN.main()
            EVAL.main()
            self.assertEqual(len(evaluated_tokens), 18)
            self.assertEqual(len(events), 6)
            args.validate_only = False
            with self.assertRaises(FileExistsError):
                EVAL.main()
            with self.assertRaises(FileExistsError):
                RUN.main()
            for mutation in ("seed", "epoch", "order", "capture", "tokens", "initial", "rng", "background", "ids", "checks"):
                changed = copy.deepcopy(payload)
                if mutation == "seed":
                    changed["seed"] = 43
                elif mutation == "epoch":
                    changed["checkpoints"]["LOW"]["final_retain_ft"]["trainer_states"][0]["epoch"] = 1
                elif mutation == "order":
                    changed["checkpoints"]["LOW"]["stage_order"].reverse()
                elif mutation == "capture":
                    changed["checkpoints"]["LOW"]["post_npo"]["completed_final_retain_epochs"] = 2
                elif mutation == "tokens":
                    changed["checkpoints"]["LOW"]["pre_npo"]["measurement"]["token_sequences_sha256"] = "b" * 64
                elif mutation == "initial":
                    changed["starting_parameter_identity"]["actual_training_branch_sha256"]["LOW"] = "b" * 64
                elif mutation == "rng":
                    changed["rng_policy"]["data_seed"] = 43
                elif mutation == "background":
                    changed["background_dataset"]["membership_sha256"] = "b" * 64
                elif mutation == "ids":
                    changed["ordered_target_dataset_ids"]["LOW"].reverse()
                else:
                    changed["validation_results"].pop("exactly_two_final_retain_epochs")
                (root / "run_metadata.json").write_text(json.dumps(changed))
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                    EVAL._validate_training_run_metadata(root, 42, MANIFEST)
            (root / "run_metadata.json").write_text(json.dumps(payload))
            (root / "measurements/LOW_post_npo.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "boundary file hash"):
                EVAL._load_boundary_measurements(root, payload, IDS, payload["token_sequences_sha256"])


if __name__ == "__main__":
    unittest.main()
