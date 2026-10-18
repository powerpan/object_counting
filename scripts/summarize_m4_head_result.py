"""Read-only CPU summaries of audited val results using existing metric helpers.

This does not run inference, select a checkpoint, or change later experiments.
"""
import argparse
import json
import math
from pathlib import Path

from audit_b0_results import aggregate, read_rows, sha
from analyze_m2_m3 import paired_intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--classes', type=Path, required=True)
    parser.add_argument('--run', choices=['head-S-seed42', 'head-L-seed42'], required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    cfg = json.loads((root/'configs/round2_v1.json').read_text())
    splits = json.loads((root/'configs/round2_splits_v1.json').read_text())
    assert sha(args.classes) == splits['source_sha256']['ImageClasses_FSC147.txt']
    labels = dict(line.split(maxsplit=1) for line in args.classes.read_text().splitlines() if line.strip())
    folder = root/'results/round2-v1'/('val-'+args.run)
    target = folder/'comparison.json'
    if target.exists():
        raise FileExistsError('Do not overwrite an existing comparison')
    audit = json.loads((folder/'audit.json').read_text())
    summary = json.loads((folder/'summary.json').read_text())
    protocol = json.loads((folder/'protocol.json').read_text())
    assert audit['status'] == 'passed' and audit['n'] == 1286
    for name, digest in audit['files_sha256'].items():
        assert sha(folder/name) == digest
    base_file = root/cfg['baseline_result']/'predictions.jsonl'
    assert sha(base_file) == protocol['baseline_predictions_sha256']
    rows = read_rows(folder/'predictions.jsonl'); base_rows = read_rows(base_file)
    base = {row['image']:row for row in base_rows}
    assert [r['image'] for r in rows] == protocol['ids'] == [r['image'] for r in base_rows]
    assert len(rows) == 1286 and len(base) == 1286
    assert set(base) == set(splits['val_A']+splits['val_B'])
    assert all(r['truth'] == base[r['image']]['truth'] for r in rows)
    groups = {'overall':rows, '<50':[r for r in rows if r['truth'] < 50],
              '50-299':[r for r in rows if 50 <= r['truth'] < 300],
              '>=300':[r for r in rows if r['truth'] >= 300]}
    def compare(sample):
        candidate = aggregate(sample); reference = aggregate([base[r['image']] for r in sample])
        return {'candidate':candidate, 'B0':reference,
                'mae_delta':candidate['mae']-reference['mae'],
                'rmse_delta':candidate['rmse']-reference['rmse'],
                'candidate_mean_signed_error':math.fsum(r['prediction']-r['truth'] for r in sample)/len(sample),
                'candidate_under_count_images':sum(r['prediction'] < r['truth'] for r in sample)}
    grouped = {}
    for name, sample in groups.items():
        grouped[name] = compare(sample)
        recorded = summary['overall'] if name == 'overall' else summary['groups'][name]
        for key, value in grouped[name]['candidate'].items():
            assert math.isclose(value, recorded[key], rel_tol=1e-10, abs_tol=1e-8)
        grouped[name]['paired'] = {unit:paired_intervals(sample,base,labels,cfg['crossfit']['bootstrap_replicates'],unit)
                                  for unit in ('image','class')}
    classes = {name:compare([r for r in rows if labels[r['image']] == name])
               for name in sorted({labels[r['image']] for r in rows})}
    dense_classes = {name:compare([r for r in groups['>=300'] if labels[r['image']] == name])
                     for name in sorted({labels[r['image']] for r in groups['>=300']})}
    changes = [{'image':r['image'], 'class':labels[r['image']], 'truth':r['truth'],
                'B0_prediction':base[r['image']]['prediction'], 'prediction':r['prediction'],
                'absolute_error_delta':abs(r['prediction']-r['truth'])-abs(base[r['image']]['prediction']-r['truth'])}
               for r in rows]
    changes.sort(key=lambda r:(r['absolute_error_delta'],r['image']))
    thresholds = cfg['admission']; overall = grouped['overall']; dense = grouped['>=300']
    checks = {'mae_at_most_102_percent_B0':overall['candidate']['mae'] <= thresholds['mae_max_b0_ratio']*overall['B0']['mae'],
              'rmse_below_B0':overall['candidate']['rmse'] < overall['B0']['rmse'],
              'dense_mae_below_B0':dense['candidate']['mae'] < dense['B0']['mae'],
              'game2_below_M1':overall['candidate']['game2'] < overall['B0']['game2']}
    report = {'status':'passed','scope':'single fixed seed42 head on development val; not crossfit selection or final test',
              'run':args.run,'n':len(rows),'class_count':len(classes),'groups':grouped,
              'classes':classes,'dense_classes':dense_classes,'single_head_threshold_checks':checks,
              'improved_images':sum(r['absolute_error_delta'] < 0 for r in changes),
              'worsened_images':sum(r['absolute_error_delta'] > 0 for r in changes),
              'unchanged_images':sum(r['absolute_error_delta'] == 0 for r in changes),
              'largest_improvements':changes[:10],'largest_regressions':list(reversed(changes[-10:])),
              'source_sha256':{'classes':sha(args.classes),'audit':sha(folder/'audit.json'),
                               'summary_script':sha(Path(__file__)),
                               'existing_bootstrap_helper':sha(Path(__file__).with_name('analyze_m2_m3.py'))}}
    with target.open('x') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','n':len(rows),'classes':len(classes),'single_head_threshold_checks':checks}))


if __name__ == '__main__':
    main()
