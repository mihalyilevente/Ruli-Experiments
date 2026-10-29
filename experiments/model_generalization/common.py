"""Model-neutral adapters; Experiment 2 and upstream RULI remain untouched."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RULI = ROOT.parent / "Ruli"
MANIFEST = HERE.parent / "experiment_2/results/intervention_manifest.json"
DEFAULT_MODEL = "EleutherAI/pythia-160m"
REVISIONS = json.loads((HERE / "model_revisions.json").read_text())
CONDITIONS = ("HIGH", "LOW", "PLACEBO")
SHADOW_FIELDS = tuple(f"{role}_{stage}" for stage in ("original", "unlearned")
                      for role in ("in", "out", "unlearn"))
PROTOCOL_ID = "matched_models_v1"
PROTOCOL_DIR = HERE / "protocol" / PROTOCOL_ID
ASSIGNMENT_PATH = PROTOCOL_DIR / "assignment_map.json"
COMPARISON_PATH = PROTOCOL_DIR / "comparison.json"
PYTHIA_EXCEPTIONS = (52, 59, 62, 74, 75, 117, 138, 175, 209, 239, 314, 315,
                     319, 350, 365, 368, 371, 402, 457, 488, 500, 518, 521, 525, 653)
LOSS_POLICY = "min(7, seq_len - 1)"
RAW_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def model_paths(name):
    model_spec(name)
    slug = name.split("/")[-1]
    return {"data": HERE / "artifacts" / PROTOCOL_ID / slug / "data",
            "shadows": HERE / "artifacts" / PROTOCOL_ID / slug / "shadows",
            "validation": HERE / "validation" / PROTOCOL_ID / f"{slug}_preflight.json",
            "results": HERE / "results" / PROTOCOL_ID / slug}


def comparison_config(manifest):
    config = read_json(COMPARISON_PATH)
    if (config.get("protocol_id") != PROTOCOL_ID
            or config.get("comparison_kind") != "fresh_matched_comparison"
            or config.get("original_assignment_reproduced") is not False
            or config.get("manifest_content_sha256") != manifest["manifest_hash"]["sha256"]
            or config.get("loss_policy") != LOSS_POLICY
            or config.get("pythia_exception_ids") != list(PYTHIA_EXCEPTIONS)
            or config.get("models") != {name: model_spec(name) for name in REVISIONS}):
        raise ValueError("Approved matched-comparison configuration mismatch.")
    return config


def prediction_policy(dataset, name, manifest):
    ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if ids != list(range(len(dataset))):
        raise ValueError("Prediction counts require the exact frozen target order.")
    primary = manifest["sets"]["S_sample_ids"]
    negative = manifest["sets"]["negative_control_sample_ids"]
    if len(primary) != 28 or len(negative) != 121:
        raise ValueError("Expected 28 primary and 121 negative-control samples.")
    records = []
    for idx, row in enumerate(dataset):
        length = len(row["input_ids"])
        if length < 2 or row["attention_mask"] != [1] * length:
            raise ValueError(f"Target {idx} contains padding or insufficient prediction tokens.")
        records.append({"sample_id": idx, "input_token_count": length,
                        "prediction_token_count": min(7, length - 1),
                        "short_sequence_exception": length < 8,
                        "loss_policy": LOSS_POLICY})
    for label, cohort in (("primary", primary), ("negative-control", negative)):
        if any(records[i]["prediction_token_count"] != 7 for i in cohort):
            raise ValueError(f"Every {label} sample must still have seven predictions.")
    exceptions = [r for r in records if r["short_sequence_exception"]]
    expected = list(PYTHIA_EXCEPTIONS) if name == DEFAULT_MODEL else []
    model_spec(name)
    if [r["sample_id"] for r in exceptions] != expected or any(r["input_token_count"] != 7 for r in exceptions):
        raise ValueError("Short-sequence exceptions differ from the approved exact 25-ID Pythia allowlist (none for GPT-2).")
    report = {"loss_policy": LOSS_POLICY, "per_sample_sha256": digest(records),
              "exception_ids": expected, "exceptions": exceptions,
              "primary_count": 28, "negative_control_count": 121,
              "primary_all_seven": True, "negative_controls_all_seven": True}
    return records, report


def module(filename):
    path = HERE.parent / "experiment_2" / filename
    spec = importlib.util.spec_from_file_location("_generalization_" + path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


BASE = module("run_experiment_2b.py")
COMMON = BASE.COMMON
REFERENCE = module("evaluate_experiment_2a.py")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    COMMON._atomic_json(value, path)


def model_spec(name):
    if name not in REVISIONS:
        raise ValueError(f"Supported models: {list(REVISIONS)}; got {name!r}")
    return {"model_name": name, **REVISIONS[name]}


def load_tokenizer(name):
    from transformers import AutoTokenizer
    spec = model_spec(name)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=spec["revision"])
    if tokenizer.eos_token_id is None:
        raise ValueError("The reference padding policy requires an EOS token.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenizer_metadata(tokenizer, name):
    from tokenizers import AddedToken

    def plain(value):
        if isinstance(value, AddedToken):
            return value.__getstate__()
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    # Cache filenames vary across machines; hash semantic configuration instead.
    config = {k: plain(v) for k, v in tokenizer.init_kwargs.items()
              if not k.endswith('_file') and k not in ('name_or_path', 'cache_dir')}
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    return {
        "name": name, "revision": model_spec(name)["revision"],
        "class": type(tokenizer).__name__, "vocab_size": tokenizer.vocab_size,
        "length_including_added_tokens": len(tokenizer),
        "backend_sha256": digest(backend),
        "config_sha256": digest({"init_kwargs": config,
                                  "special_tokens": tokenizer.special_tokens_map}),
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id, "padding_side": tokenizer.padding_side,
        "add_special_tokens": True, "truncation": True, "max_length": 128,
    }


def load_model(name, device="cpu", checkpoint=None):
    from transformers import AutoModelForCausalLM
    spec = model_spec(name)
    kwargs = {} if checkpoint is not None else {"revision": spec["revision"]}
    model = AutoModelForCausalLM.from_pretrained(checkpoint or name, **kwargs)
    if model.config.model_type != spec["model_type"]:
        raise ValueError("Checkpoint architecture does not match selected model.")
    return model.to(device)


def manifest_at(path):
    manifest, metadata = COMMON._load_and_validate_manifest(Path(path))
    conditions, checks = BASE._validate_experiment_2b_protocol(manifest)
    for key, expected in (("in_ids", range(200)), ("unlearn_ids", range(200, 400)),
                          ("out_ids", range(400, 600))):
        if manifest["evaluation_ids"][key] != list(expected):
            raise ValueError(f"Unexpected logical partition: {key}")
    return manifest, metadata, conditions, checks


def token_hash(dataset):
    return digest([{key: row[key] for key in ("input_ids", "attention_mask")}
                   for row in dataset])


def raw_hash(rows):
    return digest([{k: row[k] for k in ("logical_id", "source_indices", "raw_text_sha256")}
                   for row in rows])


def load_bundle(path, manifest, name):
    """Recompute hashes on every use; never infer model identity from a filename."""
    from datasets import load_from_disk
    path = Path(path)
    meta = read_json(path / "metadata.json")
    if meta["model"] != model_spec(name) or meta["manifest_content_sha256"] != manifest["manifest_hash"]["sha256"]:
        raise ValueError("Data bundle model/revision or manifest mismatch.")
    if meta["validation"]["passed"] is not True:
        raise ValueError(f"Data preparation did not pass validation: {meta['validation'].get('insufficient_tokens', {})}. See {path / 'metadata.json'}; research training is blocked.")
    if meta.get("comparison") != comparison_config(manifest):
        raise ValueError("Data bundle is not from the approved fresh matched comparison.")
    tokenizer = load_tokenizer(name)
    if tokenizer_metadata(tokenizer, name) != meta["tokenizer"]:
        raise ValueError("Tokenizer changed since data preparation.")
    datasets = {}
    for split in ("target", "background", "validation"):
        raw = read_json(path / f"{split}_raw.json")
        if any(text_hash(row["text"]) != row["raw_text_sha256"] for row in raw):
            raise ValueError(f"Changed raw text in {split}.")
        data = load_from_disk(str(path / split))
        record = meta["datasets"][split]
        if len(data) != record["count"] or raw_hash(raw) != record["raw_membership_sha256"] or token_hash(data) != record["tokenized_sha256"]:
            raise ValueError(f"Changed membership or tokenization in {split}.")
        if [row["logical_id"] for row in raw] != record["ordered_logical_ids"]:
            raise ValueError(f"Changed logical IDs in {split}.")
        encoded = tokenizer([row["text"] for row in raw], truncation=True, max_length=128)
        for idx, row in enumerate(data):
            if any(row[key] != encoded[key][idx] for key in ("input_ids", "attention_mask")):
                raise ValueError(f"Tokenization does not match raw text/model: {split}/{idx}.")
            if len(row["input_ids"]) < (2 if split == "target" else 1):
                raise ValueError(f"Insufficient prediction tokens: {split}/{idx}.")
        if split == "background":
            expected = manifest["shared_wikitext_background"]["rows"]
            if len(raw) != len(expected) or any(
                    row["logical_id"] != frozen["row_id"] or
                    row["gpt2_decoded_text_sha256"] != frozen["text_sha256"]
                    for row, frozen in zip(raw, expected, strict=True)):
                raise ValueError("Background membership differs from the frozen manifest.")
        if split == "target":
            records, policy = prediction_policy(data, name, manifest)
            if policy != meta.get("prediction_policy") or records != read_json(path / "target_prediction_counts.json"):
                raise ValueError("Prediction count log or loss policy changed.")
            for group in manifest["samples"].values():
                for sample in group:
                    if raw[sample["sample_id"]]["gpt2_decoded_text_sha256"] != sample["text_sha256"]:
                        raise ValueError("S/U/P/R text identity mismatch.")
        datasets[split] = data
        if record["raw_membership_sha256"] != meta["comparison"]["raw_membership_sha256"][split]:
            raise ValueError(f"Raw membership differs from the shared comparison: {split}.")
    if len(datasets["background"]) != 15000:
        raise ValueError("Background must have exactly 15000 examples.")
    if meta["datasets"]["target"]["ordered_logical_ids"] != manifest["evaluation_ids"]["ordered_shadow_target_ids"]:
        raise ValueError("Target IDs differ from frozen shadow target universe.")
    return datasets, tokenizer, meta


def source_metadata(ruli_root):
    return {name: text_hash((Path(ruli_root) / "text" / name).read_text(encoding="utf-8"))
            for name in ("utils.py", "unlearner.py", "train_text.py", "attack_main.py")}


def identity(meta):
    return {"model": meta["model"], "tokenizer": meta["tokenizer"],
            "manifest_content_sha256": meta["manifest_content_sha256"],
            "datasets": meta["datasets"], "comparison": meta.get("comparison"),
            "prediction_policy": meta.get("prediction_policy")}


def require_preflight(path, meta, ruli_root):
    report = read_json(path)
    if (report.get("passed") is not True or report.get("data_identity") != identity(meta)
            or report.get("source_hashes") != source_metadata(ruli_root)
            or report.get("architecture_smoke", {}).get("passed") is not True
            or report.get("architecture_smoke", {}).get("model") != meta["model"]
            or report.get("gpt2_regression_smoke", {}).get("passed") is not True
            or report.get("gpt2_regression_smoke", {}).get("model") != model_spec("gpt2")):
        raise ValueError("A successful --validate-only report for these exact inputs and sources is required before research training.")
    return {"path": str(Path(path).resolve()), "sha256": COMMON._sha256_file(Path(path))}


def validate_assignments(payload, manifest):
    """Both models must use the same frozen new map, never inferred old assignments."""
    config = comparison_config(manifest)
    if (payload.get("comparison_kind") != "fresh_matched_comparison"
            or payload.get("original_assignment_reproduced") is not False
            or payload.get("assignment_origin") != "new_deterministic_revision"
            or payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("manifest_content_sha256") != config["manifest_content_sha256"]
            or payload.get("assignment_seed") != 42
            or payload.get("historical_shadow_sha256") != manifest["input_artifacts"]["shadow_artifact"]["sha256"]):
        raise ValueError("Expected fresh matched-comparison assignment provenance, not reproduction of the old run.")
    mapping = payload.get("assignments", {})
    expected = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if sorted(map(int, mapping)) != expected:
        raise ValueError("Assignment IDs do not match all frozen target IDs.")
    for idx in expected:
        row = mapping[str(idx)]
        if set(row) != {"IN", "OUT", "UNLEARN"}:
            raise ValueError(f"Invalid assignment roles for {idx}.")
        flat = []
        for role in ("IN", "OUT", "UNLEARN"):
            ids = row[role]
            if len(ids) != 3 or any(type(i) is not int for i in ids):
                raise ValueError(f"Expected three integer shadows per role for {idx}.")
            flat.extend(ids)
        if sorted(flat) != list(range(9)):
            raise ValueError(f"Assignments overlap or omit shadows for {idx}.")
    if digest(mapping) != payload.get("mapping_sha256") or digest(payload) != config["assignment_payload_sha256"]:
        raise ValueError("Assignment map differs from the exact frozen map shared by GPT-2 and Pythia.")
    return mapping


def validate_shadow_losses(shadow, ids):
    if set(shadow) != set(SHADOW_FIELDS):
        raise ValueError("Expected all six original/unlearned shadow distributions.")
    for field, mapping in shadow.items():
        if sorted(mapping) != ids:
            raise ValueError(f"Shadow IDs differ in {field}.")
        for idx, values in mapping.items():
            if len(values) != 3 or not all(math.isfinite(v) for v in values):
                raise ValueError(f"Expected three finite observations: {field}/{idx}.")


def load_shadow(path, meta, manifest):
    import torch
    path = Path(path)
    info = read_json(path.with_suffix(".metadata.json"))
    if info.get("complete") is not True or info.get("smoke_only") is not False:
        raise ValueError("A complete nine-shadow research artifact is required.")
    if info["data_identity"] != identity(meta):
        raise ValueError("Shadow model/tokenizer/raw-data identity mismatch.")
    if info["sha256"] != COMMON._sha256_file(path):
        raise ValueError("Shadow file hash mismatch.")
    validate_assignments(info["assignment_provenance"], manifest)
    if info.get("comparison") != meta["comparison"] or info.get("prediction_policy") != meta["prediction_policy"]:
        raise ValueError("Shadow comparison label or prediction policy mismatch.")
    shadow = torch.load(path, map_location="cpu", weights_only=True)
    validate_shadow_losses(shadow, manifest["evaluation_ids"]["ordered_shadow_target_ids"])
    return shadow, info
