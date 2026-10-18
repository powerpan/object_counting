"""Independently audit completed val outputs against annotation point counts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def aggregate(rows):
    if not rows:
        return {'n': 0}
    truth = [r['truth'] for r in rows]
    errors = [r['prediction'] - r['truth'] for r in rows]
    return {'n': len(rows), 'mae': statistics.mean(map(abs, errors)),
            'rmse': math.sqrt(statistics.mean(e * e for e in errors)),
            'truth_total': sum(truth), 'truth_min': min(truth),
            'truth_median': statistics.median(truth), 'truth_mean': statistics.mean(truth),
            'truth_max': max(truth), 'game1': statistics.mean(r['game1'] for r in rows),
            'game2': statistics.mean(r['game2'] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--previous-result', type=Path)
    args = p.parse_args()
    result = args.result
    protocol = json.loads((result / 'protocol.json').read_text())
    summary = json.loads((result / 'summary.json').read_text())
    rows = read_rows(result / 'predictions.jsonl')
    annotations_dir = args.root / 'data/fsc147/annotations'
    annotation_file = annotations_dir / 'annotation_FSC147_384.json'
    split_file = annotations_dir / 'Train_Test_Val_FSC_147.json'
    annotations = json.loads(annotation_file.read_text())
    splits = json.loads(split_file.read_text())
    assert summary['status'] == 'complete'
    assert protocol['split'] == 'val' and protocol['selection'] == 'all'
    assert [r['image'] for r in rows] == protocol['ids'] == splits['val']
    assert len(rows) == len(set(protocol['ids'])) == 1286
    assert not set(protocol['ids']) & (set(splits['train']) | set(splits['test']))
    assert protocol['annotation_sha256'] == sha(annotation_file)
    assert protocol['split_sha256'] == sha(split_file)
    assert protocol['runner_sha256'] == sha(args.root / 'scripts/evaluate_val.py')
    assert protocol['density_module_sha256'] == sha(args.root / 'planb/density.py')
    load = json.loads((result / 'checkpoint_load.json').read_text())
    assert not load['invalid_missing'] and not load['invalid_extra']
    for row in rows:
        assert row['truth'] == len(annotations[row['image']]['points'])
        assert row['prediction'] == len(row['boxes_xyxy'])
        assert all(math.isfinite(x) for box in row['boxes_xyxy'] for x in box)
        assert math.isclose(row['density_mass'], row['prediction'], abs_tol=1e-8, rel_tol=0)
        error = abs(row['prediction'] - row['truth'])
        assert row['game2'] + 1e-8 >= row['game1'] >= error - 1e-8
        assert math.isfinite(row['inference_seconds_per_image']) and row['inference_seconds_per_image'] > 0
    groups = {'<50': [r for r in rows if r['truth'] < 50],
              '50-299': [r for r in rows if 50 <= r['truth'] < 300],
              '>=300': [r for r in rows if r['truth'] >= 300]}
    for sample, recorded in [(rows, summary['overall'])] + [(sample, summary['groups'][key]) for key, sample in groups.items()]:
        for key, value in aggregate(sample).items():
            assert math.isclose(recorded[key], value, abs_tol=1e-10, rel_tol=1e-12), key
    previous = {'compared': False}
    if args.previous_result:
        old = read_rows(args.previous_result / 'predictions.jsonl')
        lookup = {r['image']: r for r in rows}
        changed_counts = [r['image'] for r in old if r['prediction'] != lookup[r['image']]['prediction']]
        changed_boxes = [r['image'] for r in old if r['boxes_xyxy'] != lookup[r['image']]['boxes_xyxy']]
        previous = {'compared': True, 'images': len(old), 'changed_count_images': changed_counts,
                    'changed_box_images': changed_boxes}
        assert not changed_counts and not changed_boxes, 'Unexpected model-output change after annotation-only fix'
    times = sorted(r['inference_seconds_per_image'] for r in rows)
    report = {'status': 'passed', 'scope': 'full_val', 'n': len(rows),
              'annotation_point_counts_match': True, 'independent_metrics_match': True,
              'density_mass_matches_counts': True, 'split_and_code_hashes_match': True,
              'previous_partial_run': previous,
              'game_boundary_adjusted_points': sum(r['game_boundary_adjusted_points'] for r in rows),
              'mean_signed_count_error': statistics.mean(r['prediction'] - r['truth'] for r in rows),
              'sum_absolute_errors_over_total_truth': sum(abs(r['prediction'] - r['truth']) for r in rows) / sum(r['truth'] for r in rows),
              'inference_seconds_mean': statistics.mean(times),
              'inference_seconds_median': statistics.median(times),
              'inference_seconds_p95': times[math.ceil(.95 * len(times)) - 1],
              'maximum_density_mass_error': max(abs(r['density_mass'] - r['prediction']) for r in rows),
              'files_sha256': {name: sha(result / name) for name in ('protocol.json', 'summary.json', 'predictions.jsonl', 'checkpoint_load.json')}}
    assert report['game_boundary_adjusted_points'] == summary['game_boundary_adjusted_points']
    (result / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
