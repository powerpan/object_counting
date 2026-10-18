"""Independent metrics audit and paired val bootstrap; no model inference."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

from audit_b0_results import aggregate, read_rows, sha


def paired_intervals(rows, reference, labels, replicates, unit):
    errors = np.array([r['prediction'] - r['truth'] for r in rows], dtype=float)
    base = np.array([reference[r['image']]['prediction'] - r['truth'] for r in rows], dtype=float)
    units = np.arange(len(rows)) if unit == 'image' else np.array([labels[r['image']] for r in rows])
    _, inverse = np.unique(units, return_inverse=True)
    counts = np.bincount(inverse)
    k = len(counts)
    delta_ae = np.bincount(inverse, weights=np.abs(errors) - np.abs(base))
    se = np.bincount(inverse, weights=errors**2)
    se0 = np.bincount(inverse, weights=base**2)
    rng = np.random.default_rng(42)
    weights = rng.multinomial(k, np.full(k, 1/k), size=replicates)
    denominators = weights @ counts
    mae = weights @ delta_ae / denominators
    rmse = np.sqrt(weights @ se / denominators) - np.sqrt(weights @ se0 / denominators)
    return {'unit': unit, 'units': k, 'replicates': replicates, 'seed': 42,
            'mae_delta': float(np.mean(np.abs(errors) - np.abs(base))),
            'mae_delta_95ci': np.quantile(mae, [.025, .975]).tolist(),
            'rmse_delta': float(np.sqrt(np.mean(errors**2)) - np.sqrt(np.mean(base**2))),
            'rmse_delta_95ci': np.quantile(rmse, [.025, .975]).tolist()}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    output = root / 'results/m2-m3-v1'
    bdir = root / 'results/b0-val-b1-boundary-r1'
    baseline_rows = read_rows(bdir / 'predictions.jsonl')
    baseline = {r['image']: r for r in baseline_rows}
    base_summary = json.loads((bdir / 'summary.json').read_text())
    config_file = root / 'configs/m2_m3_v1.json'
    cfg = json.loads(config_file.read_text())
    annotations = json.loads((root / 'data/fsc147/annotations/annotation_FSC147_384.json').read_text())
    class_file = root / 'data/fsc147/annotations/ImageClasses_FSC147.txt'
    labels = dict(line.strip().split(maxsplit=1) for line in class_file.read_text().splitlines() if line.strip())
    all_rows = {}
    for variant in cfg['variants']:
        d = output / ('val-' + variant)
        if not (d / 'summary.json').exists(): continue
        summary = json.loads((d / 'summary.json').read_text())
        if summary['status'] != 'complete': continue
        rows = read_rows(d / 'predictions.jsonl')
        protocol = json.loads((d / 'protocol.json').read_text())
        assert len(rows) == 1286 and protocol['phase'] == 'full'
        assert [r['image'] for r in rows] == protocol['ids'] == [r['image'] for r in baseline_rows]
        assert protocol['baseline_predictions_sha256'] == sha(bdir / 'predictions.jsonl')
        assert protocol['config_sha256'] == sha(config_file)
        for file, value in protocol['code_sha256'].items(): assert sha(root / file) == value, file
        for row in rows:
            assert row['truth'] == len(annotations[row['image']]['points']) == baseline[row['image']]['truth']
            assert row['detail']['base_count'] == baseline[row['image']]['prediction']
            assert row['prediction'] == len(row['boxes_xyxy'])
            assert all(math.isfinite(x) for box in row['boxes_xyxy'] for x in box)
            assert math.isclose(row['density_mass'], row['prediction'], abs_tol=1e-8, rel_tol=0)
            assert row['game2'] + 1e-8 >= row['game1'] >= abs(row['prediction'] - row['truth']) - 1e-8
            assert math.isfinite(row['inference_seconds_per_image']) and row['inference_seconds_per_image'] > 0
        groups = {'overall': rows, '<50': [r for r in rows if r['truth'] < 50],
                  '50-299': [r for r in rows if 50 <= r['truth'] < 300],
                  '>=300': [r for r in rows if r['truth'] >= 300]}
        intervals = {}
        for key, sample in groups.items():
            recorded = summary['overall'] if key == 'overall' else summary['groups'][key]
            for metric, value in aggregate(sample).items():
                assert math.isclose(recorded[metric], value, abs_tol=1e-10, rel_tol=1e-12), metric
            intervals[key] = {unit: paired_intervals(sample, baseline, labels, cfg['bootstrap_replicates'], unit)
                              for unit in ('image', 'class')}
        changes = []
        for r in rows:
            b = baseline[r['image']]
            changes.append({'image': r['image'], 'truth': r['truth'], 'baseline': b['prediction'],
                            'prediction': r['prediction'],
                            'absolute_error_delta': abs(r['prediction']-r['truth'])-abs(b['prediction']-b['truth']),
                            'class': labels[r['image']]})
        changes.sort(key=lambda r: (r['absolute_error_delta'], r['image']))
        pseudo = [p for r in rows for p in r['detail']['pseudo_exemplars']]
        untriggered = [r for r in rows if not r['detail']['m2_triggered'] and not r['detail']['m3_triggered']]
        assert all(r['boxes_xyxy'] == baseline[r['image']]['boxes_xyxy'] for r in untriggered)
        if variant == 'm3-repeat':
            assert all(r['boxes_xyxy'] == baseline[r['image']]['boxes_xyxy'] for r in rows)
        report = {'status': 'passed', 'variant': variant, 'n': len(rows),
                  'class_count': len({labels[r['image']] for r in rows}),
                  'classes_sha256': sha(class_file), 'analysis_code_sha256': sha(Path(__file__)),
                  'independent_metrics_match': True, 'annotation_counts_match': True,
                  'untriggered_images_identical_to_baseline': len(untriggered),
                  'paired_vs_b0': intervals,
                  'improved_images': sum(r['absolute_error_delta'] < 0 for r in changes),
                  'worsened_images': sum(r['absolute_error_delta'] > 0 for r in changes),
                  'unchanged_error_images': sum(r['absolute_error_delta'] == 0 for r in changes),
                  'largest_improvements': changes[:10], 'largest_regressions': changes[-10:][::-1],
                  'pseudo_candidates': len(pseudo),
                  'pseudo_candidates_with_annotation_inside': sum(p['annotated_points_inside'] > 0 for p in pseudo),
                  'pseudo_support_is_only_a_proxy': True,
                  'mean_backbone_passes': statistics.mean(r['detail']['backbone_passes'] for r in rows),
                  'mean_decoder_passes': statistics.mean(r['detail']['decoder_passes'] for r in rows),
                  'val_candidate_gate': (summary['overall']['mae'] <= base_summary['overall']['mae'] * 1.02
                    and summary['overall']['rmse'] < base_summary['overall']['rmse']
                    and summary['groups']['>=300']['mae'] < base_summary['groups']['>=300']['mae']),
                  'files_sha256': {file: sha(d / file) for file in ('protocol.json', 'summary.json', 'predictions.jsonl')}}
        (d / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
        all_rows[variant] = rows
        print(json.dumps({'variant': variant, 'audit': 'passed', 'overall': summary['overall'],
                          'dense_mae': summary['groups']['>=300']['mae'], 'candidate': report['val_candidate_gate']}), flush=True)
    contrasts = {}
    for treatment, control in [('m2', 'm2-repeat'), ('m3', 'm3-repeat'), ('m2-m3', 'm2'), ('m2-m3', 'm3')]:
        if treatment in all_rows and control in all_rows:
            ref = {r['image']: r for r in all_rows[control]}
            contrasts[treatment + '_vs_' + control] = {
                unit: paired_intervals(all_rows[treatment], ref, labels, cfg['bootstrap_replicates'], unit)
                for unit in ('image', 'class')}
    (output / 'contrasts.json').write_text(json.dumps({'completed_variants': list(all_rows), 'contrasts': contrasts}, indent=2) + '\n')


if __name__ == '__main__':
    main()
