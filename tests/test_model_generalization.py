"""Identity, protocol, and contamination checks for the isolated model experiment."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments/model_generalization"
sys.path.insert(0, str(SCRIPTS))
import common as MG
import prepare_data as DATA
import train_shadows as SHADOW
import training as TRAIN
import freeze_matched_assignments as FREEZE


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, texts, **kwargs):
        ids = [self.mapping[text] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    def decode(self, ids, **kwargs):
        return "decoded prefix only"


def assignments():
    return MG.read_json(MG.ASSIGNMENT_PATH)


class RawIdentityTests(unittest.TestCase):
    def test_recovers_full_text_and_preserves_target_order(self):
        tokenizer = FakeTokenizer({"first full source": [1, 2], "second full source": [3, 4]})
        raw = [{"text": "first full source"}, {"text": "second full source"}]
        target = [{"input_ids": [3, 4]}, {"input_ids": [1, 2]}]
        result = DATA.recover_targets(target, raw, tokenizer)
        self.assertEqual([r["text"] for r in result], ["second full source", "first full source"])
        self.assertEqual([r["logical_id"] for r in result], [0, 1])
        self.assertNotEqual(result[0]["raw_text_sha256"], result[0]["gpt2_decoded_text_sha256"])

    def test_ambiguous_truncated_prefix_fails(self):
        tokenizer = FakeTokenizer({"shared prefix suffix A": [1, 2], "shared prefix suffix B": [1, 2]})
        with self.assertRaisesRegex(ValueError, "2 distinct"):
            DATA.recover_targets([{"input_ids": [1, 2]}],
                                 [{"text": text} for text in tokenizer.mapping], tokenizer)

    def test_missing_source_fails_instead_of_decoding(self):
        tokenizer = FakeTokenizer({"other text": [3, 4]})
        with self.assertRaisesRegex(ValueError, "0 distinct"):
            DATA.recover_targets([{"input_ids": [1, 2]}], [{"text": "other text"}], tokenizer)

    def test_identical_source_strings_are_not_false_ambiguity(self):
        tokenizer = FakeTokenizer({"same raw text": [1, 2]})
        result = DATA.recover_targets([{"input_ids": [1, 2]}], [{"text": "same raw text"}] * 2, tokenizer)
        self.assertEqual(result[0]["source_indices"], [0, 1])

    def test_target_too_short_is_rejected_without_changing_membership(self):
        tokenizer = FakeTokenizer({"a": list(range(7)), "b": list(range(8))})
        rows = [{"text": "a", "logical_id": 201}, {"text": "b", "logical_id": 202}]
        with self.assertRaisesRegex(ValueError, "201"):
            DATA.tokenize_rows(rows, tokenizer, "target")

    def test_frozen_reference_dataset_is_checked_before_download(self):
        from types import SimpleNamespace
        args = SimpleNamespace(manifest=MG.MANIFEST, target_data_path=Path("missing"))
        with patch.object(DATA.COMMON, "_verify_target_dataset_storage", side_effect=ValueError("frozen mismatch")), \
                patch("huggingface_hub.HfApi") as api:
            with self.assertRaisesRegex(ValueError, "frozen mismatch"):
                DATA.prepare(args)
            api.assert_not_called()


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.payload = assignments()
        self.manifest, _, _, _ = MG.manifest_at(MG.MANIFEST)

    def test_fresh_three_per_role_cover_all_nine_shadows(self):
        mapping = MG.validate_assignments(self.payload, self.manifest)
        for index in range(9):
            partition = SHADOW.shadow_partition(mapping, index)
            self.assertEqual(sorted(sum(partition.values(), [])), list(range(660)))

    def test_rejects_missing_ids_overlaps_and_wrong_reference(self):
        mutations = [lambda p: p["assignments"].pop("0"),
                     lambda p: p["assignments"]["0"].update(OUT=p["assignments"]["0"]["IN"]),
                     lambda p: p.update(historical_shadow_sha256="other"),
                     lambda p: p.update(original_assignment_reproduced=True)]
        for mutate in mutations:
            bad = copy.deepcopy(self.payload)
            mutate(bad)
            with self.assertRaises(ValueError):
                MG.validate_assignments(bad, self.manifest)

    def test_rehashed_but_different_valid_map_is_rejected_for_both_models(self):
        bad = copy.deepcopy(self.payload)
        bad['assignments']['0']['IN'], bad['assignments']['0']['OUT'] = (
            bad['assignments']['0']['OUT'], bad['assignments']['0']['IN'])
        bad['mapping_sha256'] = MG.digest(bad['assignments'])
        with self.assertRaisesRegex(ValueError, "exact frozen map"):
            MG.validate_assignments(bad, self.manifest)

    def test_map_generation_is_deterministic_and_does_not_advance_global_rng(self):
        state = random.getstate()
        first = FREEZE.build_assignments(list(range(660)))
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, FREEZE.build_assignments(list(range(660))))
        self.assertEqual(first, self.payload['assignments'])
        self.assertFalse(self.payload['original_assignment_reproduced'])
        self.assertEqual(self.payload['comparison_kind'], 'fresh_matched_comparison')

    def test_model_specific_paths_share_one_comparison_but_not_artifact_outputs(self):
        a, b = MG.model_paths('gpt2'), MG.model_paths(MG.DEFAULT_MODEL)
        for key in a:
            self.assertNotEqual(a[key], b[key])
            self.assertIn(MG.PROTOCOL_ID, a[key].parts)
            self.assertIn(MG.PROTOCOL_ID, b[key].parts)

    def test_missing_shadows_or_nonfinite_losses_fail(self):
        good = {field: {0: [1., 2., 3.]} for field in MG.SHADOW_FIELDS}
        MG.validate_shadow_losses(good, [0])
        for values in ([1.], [1., 2., float("nan")]):
            bad = copy.deepcopy(good)
            bad["out_original"][0] = values
            with self.assertRaises(ValueError):
                MG.validate_shadow_losses(bad, [0])

    def test_wrong_model_shadow_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shadow.pth"
            path.with_suffix(".metadata.json").write_text(json.dumps({
                "complete": True, "smoke_only": False, "data_identity": {"model": "gpt2"}}))
            meta = {"model": "pythia", "tokenizer": {}, "manifest_content_sha256": "x", "datasets": {}}
            with patch("torch.load") as load:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    MG.load_shadow(path, meta, self.manifest)
                load.assert_not_called()


class ProtocolTests(unittest.TestCase):
    def test_exact_25_exceptions_logged_and_primary_controls_protected(self):
        manifest, _, _, _ = MG.manifest_at(MG.MANIFEST)
        data = [{'input_ids': list(range(7 if i in MG.PYTHIA_EXCEPTIONS else 8)),
                 'attention_mask': [1] * (7 if i in MG.PYTHIA_EXCEPTIONS else 8)}
                for i in range(660)]
        rows, report = MG.prediction_policy(data, MG.DEFAULT_MODEL, manifest)
        self.assertEqual(report['exception_ids'], list(MG.PYTHIA_EXCEPTIONS))
        self.assertEqual(len(report['exceptions']), 25)
        self.assertTrue(all(r['prediction_token_count'] == 6 for r in report['exceptions']))
        self.assertEqual(sum(r['prediction_token_count'] == 6 for r in rows[200:600]), 16)
        for cohort in ('S_sample_ids', 'negative_control_sample_ids'):
            self.assertTrue(all(rows[i]['prediction_token_count'] == 7 for i in manifest['sets'][cohort]))
            mutated = copy.deepcopy(data)
            idx = manifest['sets'][cohort][0]
            mutated[idx] = {'input_ids': list(range(7)), 'attention_mask': [1] * 7}
            with self.assertRaisesRegex(ValueError, 'must still have seven'):
                MG.prediction_policy(mutated, MG.DEFAULT_MODEL, manifest)
        for idx, count in ((52, 6), (52, 8), (0, 7)):
            mutated = copy.deepcopy(data)
            mutated[idx] = {'input_ids': list(range(count)), 'attention_mask': [1] * count}
            with self.assertRaises(ValueError):
                MG.prediction_policy(mutated, MG.DEFAULT_MODEL, manifest)
        with self.assertRaises(ValueError):
            MG.prediction_policy(data, 'gpt2', manifest)

    def test_actual_target_and_shadow_losses_use_six_or_seven_predictions(self):
        import ast
        import torch
        from types import SimpleNamespace
        classes = []
        for filename, name in [('utils.py', 'MIAEvaluator'), ('train_text.py', 'LanguageMIA')]:
            tree = ast.parse((MG.RULI / 'text' / filename).read_text(encoding='utf-8'))
            classes.append(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name))
        namespace = {'torch': torch, 'F': torch.nn.functional, 'tqdm': lambda items, **kwargs: items}
        exec(compile(ast.Module(body=classes, type_ignores=[]), '<actual upstream losses>', 'exec'), namespace)
        class Model:
            def __call__(self, input_ids, attention_mask):
                n = input_ids.shape[1]
                logits = torch.arange(n * 16, dtype=torch.float32).reshape(1, n, 16) / (torch.arange(n).reshape(1, n, 1) + 1)
                return SimpleNamespace(logits=logits)
        model = Model()
        target = object.__new__(namespace['MIAEvaluator'])
        target.device = 'cpu'
        shadow = object.__new__(namespace['LanguageMIA'])
        for length in (7, 8, 12):
            ids = list(range(length))
            batch = torch.tensor([ids])
            logits = model(batch, torch.ones_like(batch)).logits
            k = min(7, length - 1)
            expected = torch.nn.functional.cross_entropy(logits[0, :-1][-k:], batch[0, 1:][-k:]).item()
            self.assertAlmostEqual(target._batch_inference(model, [ids])[0], expected)
            self.assertAlmostEqual(shadow.perform_lm_inference(model, [ids], None, 'cpu')[0], expected)

    def test_failed_data_audit_is_never_loadable_for_training(self):
        meta = {"model": MG.model_spec(MG.DEFAULT_MODEL), "manifest_content_sha256": "frozen",
                "validation": {"passed": False}}
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "metadata.json").write_text(json.dumps(meta))
            with patch.object(MG, "load_tokenizer") as tokenizer:
                with self.assertRaisesRegex(ValueError, "did not pass validation"):
                    MG.load_bundle(directory, {"manifest_hash": {"sha256": "frozen"}}, MG.DEFAULT_MODEL)
                tokenizer.assert_not_called()

    def test_architecture_smoke_alone_cannot_authorize_research_training(self):
        meta = {"model": MG.model_spec(MG.DEFAULT_MODEL), "tokenizer": {},
                "manifest_content_sha256": "frozen", "datasets": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            path.write_text(json.dumps({"passed": True, "smoke_only": True}))
            with self.assertRaisesRegex(ValueError, "validate-only"):
                MG.require_preflight(path, meta, MG.RULI)

    def test_frozen_conditions_remain_exactly_2b(self):
        manifest, _, conditions, _ = MG.manifest_at(MG.MANIFEST)
        self.assertEqual([len(conditions[c]) for c in MG.CONDITIONS], [200] * 3)
        self.assertEqual(set(conditions["LOW"]),
                         (set(range(200)) - set(manifest["sets"]["U_sample_ids"])) | set(manifest["sets"]["R_sample_ids"]))
        self.assertEqual(len(manifest["sets"]["negative_control_sample_ids"]), 121)

    def test_only_current_pinned_release_is_supported(self):
        self.assertEqual(MG.model_spec(MG.DEFAULT_MODEL)["model_type"], "gpt_neox")
        self.assertEqual(len(MG.model_spec(MG.DEFAULT_MODEL)["revision"]), 40)
        with self.assertRaises(ValueError):
            MG.model_spec("EleutherAI/pythia-160m-v0")

    def test_raw_hash_changes_on_reorder_or_membership_change(self):
        rows = [{"logical_id": i, "source_indices": [i], "raw_text_sha256": str(i)} for i in range(3)]
        self.assertNotEqual(MG.raw_hash(rows), MG.raw_hash(rows[::-1]))
        self.assertNotEqual(MG.raw_hash(rows), MG.raw_hash(rows[:-1]))

    def test_stage_argument_override_is_scoped_and_research_defaults_preserved(self):
        from types import SimpleNamespace
        from transformers import TrainingArguments
        utils = SimpleNamespace(TrainingArguments=TrainingArguments, Trainer=object, PrefixUnlearn=object)
        record = {}
        with TRAIN.stage_settings(utils, 44, "npo", False, "cpu", record):
            args = utils.TrainingArguments(output_dir="unused", num_train_epochs=15,
                                           per_device_train_batch_size=16,
                                           gradient_accumulation_steps=2, report_to="none")
            self.assertEqual((args.seed, args.data_seed, args.max_steps), (44, 44, -1))
            self.assertEqual((args.per_device_train_batch_size, args.gradient_accumulation_steps), (16, 2))
        self.assertIs(utils.TrainingArguments, TrainingArguments)
        self.assertIs(utils.Trainer, object)


if __name__ == "__main__":
    unittest.main()
