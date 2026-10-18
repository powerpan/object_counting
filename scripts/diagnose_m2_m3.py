"""Exploratory category/trigger breakdown, after observing the M2 failure."""
import argparse
import json
from pathlib import Path
from audit_b0_results import read_rows, aggregate


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    class_file = root / 'data/fsc147/annotations/ImageClasses_FSC147.txt'
    labels = dict(line.strip().split(maxsplit=1) for line in class_file.read_text().splitlines() if line.strip())
    baseline = {r['image']: r for r in read_rows(root / 'results/b0-val-b1-boundary-r1/predictions.jsonl')}
    methods = {}
    for d in sorted((root / 'results/m2-m3-v1').glob('val-*')):
        if not (d / 'analysis.json').exists(): continue
        rows = read_rows(d / 'predictions.jsonl')
        if len(rows) != 1286: continue
        def comparison(sample):
            b = [baseline[r['image']] for r in sample]
            result, reference = aggregate(sample), aggregate(b)
            return {'n': len(sample), 'method': result, 'baseline': reference,
                    'absolute_error_sum_delta': sum(abs(r['prediction']-r['truth'])-abs(baseline[r['image']]['prediction']-r['truth']) for r in sample),
                    'mae_delta': result['mae']-reference['mae'] if sample else None}
        by_class = {label: comparison([r for r in rows if labels[r['image']] == label]) for label in sorted({labels[r['image']] for r in rows})}
        by_trigger = {
            'original_exemplar_median_le32': comparison([r for r in rows if r['detail']['median_exemplar_side_model_pixels'] <= 32]),
            'original_exemplar_median_gt32': comparison([r for r in rows if r['detail']['median_exemplar_side_model_pixels'] > 32]),
            'baseline_predicted_count_ge200': comparison([r for r in rows if r['detail']['base_count'] >= 200]),
            'baseline_predicted_count_lt200': comparison([r for r in rows if r['detail']['base_count'] < 200])}
        methods[d.name] = {'by_class': by_class, 'by_trigger': by_trigger}
    output = {'scope': 'exploratory_posthoc_after_M2_result; not used for routing or parameter selection',
              'trigger_groups_overlap': True, 'methods': methods}
    (root / 'results/m2-m3-v1/diagnostics.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
