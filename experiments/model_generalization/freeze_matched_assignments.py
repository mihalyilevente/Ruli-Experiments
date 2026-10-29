"""Freeze one new deterministic assignment map for both models; never train."""
import platform
import random

from common import (ASSIGNMENT_PATH, COMPARISON_PATH, DEFAULT_MODEL, HERE, LOSS_POLICY,
                    MANIFEST, PROTOCOL_ID, PYTHIA_EXCEPTIONS, REVISIONS, digest,
                    manifest_at, model_spec, read_json, validate_assignments, write_json)


def build_assignments(ids, seed=42):
    rng = random.Random(seed)
    result = {}
    for idx in ids:
        shadows = list(range(9))
        rng.shuffle(shadows)
        result[str(idx)] = {role: sorted(shadows[start:start + 3])
                            for role, start in (("IN", 0), ("OUT", 3), ("UNLEARN", 6))}
    return result


def main():
    manifest, _, _, _ = manifest_at(MANIFEST)
    if ASSIGNMENT_PATH.exists() or COMPARISON_PATH.exists():
        raise FileExistsError("Frozen assignment/configuration already exists; reuse it, do not regenerate it.")
    audit = read_json(HERE / "validation/data_audit.json")
    if audit["manifest_content_sha256"] != manifest["manifest_hash"]["sha256"]:
        raise ValueError("Original raw-identity audit does not match manifest.")
    mapping = build_assignments(manifest["evaluation_ids"]["ordered_shadow_target_ids"])
    payload = {
        "protocol_id": PROTOCOL_ID, "comparison_kind": "fresh_matched_comparison",
        "assignment_origin": "new_deterministic_revision", "original_assignment_reproduced": False,
        "assignment_seed": 42, "python_version": platform.python_version(),
        "algorithm": "Dedicated random.Random(42); ascending target IDs; shuffle 0..8; first three IN, next three OUT, last three UNLEARN; sorted role lists.",
        "manifest_content_sha256": manifest["manifest_hash"]["sha256"],
        "historical_shadow_sha256": manifest["input_artifacts"]["shadow_artifact"]["sha256"],
        "historical_shadow_usage": "Reference only; this new map does not describe the old loss arrays.",
        "assignments": mapping, "mapping_sha256": digest(mapping),
    }
    config = {
        "protocol_id": PROTOCOL_ID, "comparison_kind": "fresh_matched_comparison",
        "original_assignment_reproduced": False, "approved_date": "2026-10-29",
        "approval_scope": "Capped last-seven loss, logged 25 exceptions, protected primary/control cohorts, one deterministic map shared by GPT-2 and Pythia. No full training authorized.",
        "manifest_content_sha256": manifest["manifest_hash"]["sha256"],
        "loss_policy": LOSS_POLICY, "pythia_exception_ids": list(PYTHIA_EXCEPTIONS),
        "models": {name: model_spec(name) for name in REVISIONS},
        "assignment_payload_sha256": digest(payload),
        "raw_membership_sha256": {split: record["raw_membership_sha256"] for split, record in audit["datasets"].items()},
    }
    write_json(ASSIGNMENT_PATH, payload)
    write_json(COMPARISON_PATH, config)
    validate_assignments(payload, manifest)
    print(f"[FROZEN] Fresh matched comparison, not an exact reproduction: {ASSIGNMENT_PATH}")
    print(f"[VERIFY] Shared assignment SHA-256: {digest(payload)}")


if __name__ == "__main__":
    main()
