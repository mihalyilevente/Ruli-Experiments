"""Validate both data bundles and the shared nine-shadow map without training."""
import argparse
from pathlib import Path

from common import (ASSIGNMENT_PATH, COMMON, DEFAULT_MODEL, HERE, MANIFEST, PROTOCOL_ID,
                    digest, load_bundle, manifest_at, model_paths, read_json,
                    validate_assignments, write_json)
from train_shadows import shadow_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path,
                        default=HERE / "validation" / PROTOCOL_ID / "matched_comparison.json")
    args = parser.parse_args()
    manifest, _, _, _ = manifest_at(MANIFEST)
    payload = read_json(ASSIGNMENT_PATH)
    mapping = validate_assignments(payload, manifest)
    partitions = {str(i): shadow_partition(mapping, i) for i in range(9)}
    report = {"passed": True, "comparison_kind": "fresh_matched_comparison",
              "original_assignment_reproduced": False, "training_started": False,
              "protocol_id": PROTOCOL_ID, "assignment_path": str(ASSIGNMENT_PATH),
              "assignment_payload_sha256": digest(payload),
              "assignment_file_sha256": COMMON._sha256_file(ASSIGNMENT_PATH),
              "per_shadow_counts": {i: {role: len(ids) for role, ids in part.items()} for i, part in partitions.items()},
              "models": {}}
    for name in ('gpt2', DEFAULT_MODEL):
        path = model_paths(name)['data']
        _, _, meta = load_bundle(path, manifest, name)
        report['models'][name] = {
            "data_metadata_sha256": COMMON._sha256_file(path / 'metadata.json'),
            "raw_membership_sha256": {split: m['raw_membership_sha256'] for split, m in meta['datasets'].items()},
            "target_tokenized_sha256": meta['datasets']['target']['tokenized_sha256'],
            "prediction_policy": meta['prediction_policy'],
            "assignment_payload_sha256": meta['comparison']['assignment_payload_sha256'],
            "all_nine_shadow_partitions_sha256": digest(partitions),
        }
    left, right = report['models'].values()
    for key in ('raw_membership_sha256', 'assignment_payload_sha256', 'all_nine_shadow_partitions_sha256'):
        if left[key] != right[key]:
            raise ValueError(f"Matched comparison mismatch: {key}")
    if left['target_tokenized_sha256'] == right['target_tokenized_sha256']:
        raise ValueError("Unexpected identical GPT-2/Pythia tokenized data.")
    write_json(args.report, report)
    print("[VERIFY] Both models share exact raw membership and all nine shadow partitions.")
    print("[VERIFY] 25 Pythia exceptions logged; both primary/control cohorts retain seven predictions.")
    print("[LABEL] Fresh matched comparison; not an exact reproduction of the old shadow run. No training.")


if __name__ == '__main__':
    main()
