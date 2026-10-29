"""Recompute descriptive statistics from the verified compact shadow export."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    checks = json.loads((root / 'download_checksums.json').read_text())
    for name, expected in checks.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected['sha256'], name
    results = {'verified_downloads': len(checks), 'statistical_tests_performed': False,
               'delta_definition': 'unlearned_loss - original_loss', 'models': {}}
    assignments = None
    for model in ('gpt2', 'pythia-160m'):
        with (root / f'{model}_shadow_0_per_sample.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 660 and sorted(int(r['sample_id']) for r in rows) == list(range(660))
        original = json.loads((root / model / 'shadow_0/losses.json').read_text())
        roles = {int(r['sample_id']): r['role'] for r in rows}
        if assignments is not None:
            assert roles == assignments
        assignments = roles
        for row in rows:
            for field, stage in (('original_loss', 'original'), ('unlearned_loss', 'unlearned')):
                value = float(row[field])
                assert math.isfinite(value)
                assert value == original['losses'][f"{row['role'].lower()}_{stage}"][row['sample_id']]
        result = {}
        for role in ('IN', 'OUT', 'UNLEARN'):
            group = [r for r in rows if r['role'] == role]
            before = [float(r['original_loss']) for r in group]
            after = [float(r['unlearned_loss']) for r in group]
            delta = [b - a for a, b in zip(before, after)]
            seven = [float(r['unlearned_loss']) - float(r['original_loss']) for r in group
                     if int(r['prediction_token_count']) == 7]
            result[role] = {'n': len(group), 'original_mean': statistics.mean(before),
                            'unlearned_mean': statistics.mean(after), 'mean_delta': statistics.mean(delta),
                            'median_delta': statistics.median(delta), 'increased_count': sum(d > 0 for d in delta),
                            'increased_percent': 100 * sum(d > 0 for d in delta) / len(delta),
                            'seven_prediction_n': len(seven), 'seven_prediction_mean_delta': statistics.mean(seven)}
        results['models'][model] = result
    (root / 'descriptive_statistics.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
