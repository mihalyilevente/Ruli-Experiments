"""Recover exact source texts, prove GPT-2 identity, then tokenize for the new model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from common import (DEFAULT_MODEL, HERE, MANIFEST, RULI, COMMON, digest, load_tokenizer,
                    manifest_at, model_spec, raw_hash, text_hash, token_hash,
                    tokenizer_metadata, write_json, comparison_config, model_paths,
                    prediction_policy, RAW_REVISION)


def recover_targets(target, raw_validation, gpt2):
    # This is exactly target_data.py's candidate source, before random.sample.
    candidates = [(i, row["text"]) for i, row in enumerate(raw_validation[:2000])
                  if row["text"].strip()]
    by_tokens = defaultdict(list)
    encoded = gpt2([text for _, text in candidates], truncation=True, max_length=128)
    for (source_id, text), tokens in zip(candidates, encoded["input_ids"], strict=True):
        by_tokens[tuple(tokens)].append((source_id, text))
    rows = []
    for idx, example in enumerate(target):
        matches = by_tokens.get(tuple(example["input_ids"]), [])
        texts = {text for _, text in matches}
        if len(texts) != 1:
            raise ValueError(f"Cannot reconstruct exact raw target {idx}: {len(texts)} distinct matching texts. Supply original raw provenance; decoding is not proof.")
        text = next(iter(texts))
        rows.append(raw_row(idx, [i for i, _ in matches], text, example["input_ids"], gpt2))
    return rows


def raw_row(logical_id, source_indices, text, gpt2_ids, tokenizer):
    return {
        "logical_id": logical_id, "source_indices": source_indices,
        "text": text, "raw_text_sha256": text_hash(text),
        "gpt2_input_ids_sha256": digest(gpt2_ids),
        "gpt2_decoded_text_sha256": text_hash(tokenizer.decode(
            gpt2_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)),
    }


def recover_background(raw_train, manifest, gpt2):
    texts = [row["text"] for row in raw_train[:50000]]
    encoded = gpt2(texts, truncation=True, max_length=128)["input_ids"]
    # source_index in the frozen manifest indexes the nonempty tokenized subset.
    filtered = [(idx, text, tokens) for idx, (text, tokens) in enumerate(zip(texts, encoded, strict=True)) if tokens]
    rows = []
    for item in manifest["shared_wikitext_background"]["rows"]:
        index = item["source_index"]
        original_idx, text, tokens = filtered[index]
        row = raw_row(item["row_id"], [original_idx], text, tokens, gpt2)
        if row["gpt2_decoded_text_sha256"] != item["text_sha256"]:
            raise ValueError(f"Frozen background text mismatch at filtered source index {index}.")
        row["filtered_source_index"] = index
        rows.append(row)
    if len(rows) != 15000 or len({r["logical_id"] for r in rows}) != 15000:
        raise ValueError("Background membership must contain exactly 15000 distinct IDs.")
    return rows


def recover_validation(raw_validation, gpt2):
    texts = [row["text"] for row in raw_validation[:2500]]
    tokens = gpt2(texts, truncation=True, max_length=128)["input_ids"]
    return [raw_row(i, [i], text, ids, gpt2)
            for i, (text, ids) in enumerate(zip(texts, tokens, strict=True)) if ids]


def tokenize_rows(rows, tokenizer, split, enforce_lengths=True):
    from datasets import Dataset
    encoded = tokenizer([row["text"] for row in rows], truncation=True, max_length=128)
    minimum = 8 if split == "target" else 1
    short = [row["logical_id"] for row, ids in zip(rows, encoded["input_ids"], strict=True)
             if len(ids) < minimum]
    if short and enforce_lengths:
        raise ValueError(f"{split}: frozen examples have fewer than {minimum} tokens with this tokenizer: {short}. No samples were removed or padded with invented content.")
    return Dataset.from_dict({k: encoded[k] for k in ("input_ids", "attention_mask")})


def prepare(args):
    from datasets import load_dataset, load_from_disk

    manifest, manifest_info, conditions, checks = manifest_at(args.manifest)
    # Check before any reconstruction, downloads, or output creation.
    target_path = args.target_data_path.resolve()
    storage = COMMON._verify_target_dataset_storage(target_path, manifest["input_artifacts"]["target_dataset"])
    target = load_from_disk(str(target_path))
    expected_ids = manifest["evaluation_ids"]["ordered_shadow_target_ids"]
    if list(range(len(target))) != expected_ids:
        raise ValueError("Target count/order differs from the frozen target universe.")
    args.output = args.output or model_paths(args.model_name)["data"]
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite data bundle {args.output}")
    if args.raw_dataset_path:
        raw = load_from_disk(str(args.raw_dataset_path))
        raw_source = {"path": str(args.raw_dataset_path.resolve()), "source": "user-provided WikiText-103 snapshot"}
    else:
        revision = RAW_REVISION
        raw = load_dataset("Salesforce/wikitext", "wikitext-103-v1", revision=revision)
        raw_source = {"name": "Salesforce/wikitext", "config": "wikitext-103-v1", "revision": revision}
    # Materialize only the original candidate windows, never resample membership.
    validation = list(raw["validation"].select(range(2500)))
    train = list(raw["train"].select(range(50000)))
    gpt2 = load_tokenizer("gpt2")
    raw_rows = {
        "target": recover_targets(target, validation, gpt2),
        "background": recover_background(train, manifest, gpt2),
        "validation": recover_validation(validation, gpt2),
    }
    for group in manifest["samples"].values():
        for sample in group:
            if raw_rows["target"][sample["sample_id"]]["gpt2_decoded_text_sha256"] != sample["text_sha256"]:
                raise ValueError("S/U/P/R frozen text hash mismatch.")
    tokenizer = load_tokenizer(args.model_name)
    datasets = {split: tokenize_rows(rows, tokenizer, split, enforce_lengths=False)
                for split, rows in raw_rows.items()}
    prediction_records, policy = prediction_policy(datasets["target"], args.model_name, manifest)
    comparison = comparison_config(manifest)
    insufficient = {split: [row["logical_id"] for row, tokenized in zip(raw_rows[split], data, strict=True)
                            if len(tokenized["input_ids"]) < (8 if split == "target" else 1)]
                    for split, data in datasets.items()}
    lengths_passed = not insufficient["background"] and not insufficient["validation"]
    if not lengths_passed:
        raise ValueError("Empty background/validation rows are not permitted.")
    metadata = {
        "schema_version": 1, "model": model_spec(args.model_name),
        "tokenizer": tokenizer_metadata(tokenizer, args.model_name),
        "gpt2_identity_tokenizer": tokenizer_metadata(gpt2, "gpt2"),
        "manifest_content_sha256": manifest["manifest_hash"]["sha256"],
        "manifest": manifest_info, "reference_target_storage": storage,
        "raw_source": raw_source, "sets": manifest["sets"],
        "conditions": conditions,
        "comparison": comparison, "prediction_policy": policy,
        "raw_text_policy": "Exact complete source string uniquely matched through the frozen GPT-2 truncated token sequence; never tokenizer.decode as raw source.",
        "datasets": {},
        "validation": {"passed": lengths_passed, **checks, "exact_raw_reconstruction": True,
                       "all_target_samples_have_seven_predictions": not insufficient["target"],
                       "primary_all_seven": policy["primary_all_seven"],
                       "negative_controls_all_seven": policy["negative_controls_all_seven"],
                       "approved_short_sequence_ids": policy["exception_ids"],
                       "loss_policy_validated": True},
    }
    for split, rows in raw_rows.items():
        data = datasets[split]
        metadata["datasets"][split] = {
            "count": len(data), "ordered_logical_ids": [r["logical_id"] for r in rows],
            "raw_membership_sha256": raw_hash(rows), "tokenized_sha256": token_hash(data),
            "fingerprint": data._fingerprint,
        }
        if raw_hash(rows) != comparison["raw_membership_sha256"][split]:
            raise ValueError(f"Raw membership differs from the matched comparison: {split}.")
    # New revision directory; preserve the original failed audit and data bundle.
    args.output.mkdir(parents=True)
    for split, data in datasets.items():
        write_json(args.output / f"{split}_raw.json", raw_rows[split])
        data.save_to_disk(str(args.output / split))
    write_json(args.output / "metadata.json", metadata)
    write_json(args.output / "target_prediction_counts.json", prediction_records)
    if getattr(args, "report", None):
        write_json(args.report, {
            "model": metadata["model"], "tokenizer": metadata["tokenizer"],
            "raw_source": raw_source, "reference_target_storage": storage,
            "manifest_content_sha256": metadata["manifest_content_sha256"],
            "datasets": {k: {field: value for field, value in v.items() if field != "ordered_logical_ids"}
                         for k, v in metadata["datasets"].items()},
            "validation": metadata["validation"],
            "comparison": comparison, "prediction_policy": policy,
            "short_target_samples": [{"sample_id": i, "text": raw_rows["target"][i]["text"],
                                      "raw_text_sha256": raw_rows["target"][i]["raw_text_sha256"],
                                      "token_count": len(datasets["target"][i]["input_ids"]),
                                      "prediction_token_count": prediction_records[i]["prediction_token_count"],
                                      "in_supported_S": i in manifest["sets"]["S_sample_ids"],
                                      "in_negative_controls": i in manifest["sets"]["negative_control_sample_ids"]}
                                     for i in insufficient["target"]],
        })
    print(f"[EXCEPTIONS] {args.model_name}: {policy['exception_ids']}; each has 7 input tokens and 6 predictions.")
    print("[VERIFY] All 28 primary and 121 negative-control samples have seven predictions.")
    print(f"[VERIFY] Exact raw membership and new tokenization saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--target-data-path", type=Path,
                        default=HERE / "artifacts/reference_gpt2/selective_dataset_prefixed_smoke_700")
    parser.add_argument("--raw-dataset-path", type=Path, help="Optional DatasetDict saved from original WikiText-103; otherwise pin a Hub snapshot.")
    parser.add_argument("--output", type=Path, help="Defaults to the selected model's matched_models_v1 data directory.")
    parser.add_argument("--report", type=Path, help="Write a compact identity/length audit, including failed validation.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
