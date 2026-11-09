"""One registered post-hoc candidate, replayed from audited maps on CPU only."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from fscd_lvis_common import checked_sources, read, sha, write_new
from audit_fscd_lvis_external import grid_error
from planb.density import game, points_for_game
from planb.half_blend import half_blend

METHODS = ('b0', 'original_final', 'half_blend')


def count_summary(rows, replicates=10000, seed=20261109):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate samples')
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError('Invalid bootstrap count')
    truth = np.asarray([r['truth'] for r in rows], dtype=np.float64)
    if not np.isfinite(truth).all() or (truth < 0).any() or (truth != np.floor(truth)).any():
        raise ValueError('Invalid truth counts')
    errors, metrics = {}, {}
    for method in METHODS:
        pred = np.asarray([r[method]['prediction'] for r in rows], dtype=np.float64)
        if not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError('Invalid predictions')
        errors[method] = pred - truth
        metrics[method] = {'mae': float(np.abs(errors[method]).mean()),
                           'rmse': float(np.sqrt(np.square(errors[method]).mean()))}
    units, inverse = np.unique([r['sampling_unit'] for r in rows], return_inverse=True)
    k = len(units)
    sizes = np.bincount(inverse)
    b, f = errors['b0'], errors['half_blend']
    ae = np.bincount(inverse, weights=np.abs(f)-np.abs(b))
    se_b, se_f = (np.bincount(inverse, weights=x*x) for x in (b, f))
    rng = np.random.default_rng(seed)
    dm, dr = [], []
    for start in range(0, replicates, 256):
        weights = rng.multinomial(k, np.full(k, 1/k), size=min(256, replicates-start))
        den = weights @ sizes
        dm.extend((weights @ ae / den).tolist())
        dr.extend((np.sqrt(weights @ se_f / den)-np.sqrt(weights @ se_b / den)).tolist())
    ci_mae, ci_rmse = (np.quantile(v, [.025, .975]).tolist() for v in (dm, dr))
    mae_delta = metrics['half_blend']['mae']-metrics['b0']['mae']
    rmse_delta = metrics['half_blend']['rmse']-metrics['b0']['rmse']
    return {'samples': len(rows), 'unique_source_images': k,
            'truth': {'total': int(truth.sum()), 'min': int(truth.min()), 'max': int(truth.max()),
                      'mean': float(truth.mean()), 'median': float(np.median(truth))},
            'methods': metrics,
            'paired_half_minus_b0': {'unit': 'source image SHA-256', 'units': k,
                'replicates': replicates, 'seed': seed, 'mae_delta': mae_delta,
                'rmse_delta': rmse_delta, 'mae_delta_95ci': ci_mae, 'rmse_delta_95ci': ci_rmse},
            'both_point_estimates_not_worse': mae_delta <= 0 and rmse_delta <= 0,
            'both_intervals_below_zero': ci_mae[1] < 0 and ci_rmse[1] < 0}


def independent_counts(rows, result):
    for row in rows:
        expected = math.fsum((row['b0']['prediction'], row['original_final']['prediction']))/2
        if row['half_blend']['prediction'] != expected:
            raise ValueError('Independent half-count disagreement')
    for method in METHODS:
        errors = [r[method]['prediction']-r['truth'] for r in rows]
        values = {'mae': math.fsum(map(abs, errors))/len(rows),
                  'rmse': math.sqrt(math.fsum(x*x for x in errors)/len(rows))}
        for key, value in values.items():
            if not math.isclose(value, result['methods'][method][key], rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError('Independent count metric disagreement')


def load_sources(root, cfg):
    receipts = {}
    for name in ('fscd_lvis_prepare_receipt_v1.json', 'fscd_lvis_eval_receipt_v1.json',
                 'fscd_lvis_freeze_receipt_v1.json', 'fscd_lvis_pipeline_receipt_v1.json',
                 'fscd_lvis_half_blend_receipt_v1.json'):
        receipts[name] = checked_sources(root, name)
    source = root/'results/fscd-lvis-external-v1/full'
    for name, expected in cfg['original_files_sha256'].items():
        if sha(source/name) != expected:
            raise ValueError('Original evidence changed')
    manifest_path = root/'data/fscd-lvis-v1/evaluation_manifest.json'
    if sha(manifest_path) != cfg['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    audit, original = read(source/'audit.json'), read(source/'summary.json')
    if audit['status'] != 'passed' or original['status'] != 'complete':
        raise ValueError('Original evaluation is incomplete')
    rows = [json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()]
    records = read(manifest_path)['splits']['full']
    if len(rows) != cfg['samples'] or len(records) != cfg['samples']:
        raise ValueError('Incomplete input')
    if [r['sample_id'] for r in rows] != [r['sample_id'] for r in records]:
        raise ValueError('Evaluation order changed')
    if len({r['sample_id'] for r in rows}) != len(rows) or sum(r['truth'] for r in rows) != cfg['truth_total']:
        raise ValueError('Duplicate input or wrong target total')
    return rows, records, original, receipts


def replay(root):
    if os.environ.get('SLURM_JOB_PARTITION') != 'amd' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Replay requires the registered CPU allocation with CUDA hidden')
    start = time.monotonic()
    cfg = read(root/'configs/fscd_lvis_half_blend_v1.json')
    if (cfg['weight_b0'], cfg['weight_original_final'], cfg['candidate_count']) != (.5, .5, 1):
        raise ValueError('Only the registered equal-weight candidate is implemented')
    out = root/'results/fscd-lvis-half-blend-v1'
    work = root/'work/fscd-lvis-half-blend-v1'
    with (root/'work/fscd-half-blend-v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        try:
            work.mkdir(exist_ok=False)
            source_rows, records, original, receipts = load_sources(root, cfg)
            receipt = receipts['fscd_lvis_half_blend_receipt_v1.json']
            write_new(out/'protocol.json', {**cfg, 'source_commit': receipt['code_commit'],
                'source_receipts_verified': 7, 'gpu_used': False,
                'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'old_final_unchanged': True, 'test_was_previously_observed': True})
            rows = []
            max_mass_error = max_grid_error = 0.
            row_path = work/'predictions.jsonl'
            with row_path.open('x') as stream:
                for index, (source_row, record) in enumerate(zip(source_rows, records), 1):
                    if time.monotonic()-start >= cfg['cpu_wall_seconds_limit']:
                        raise TimeoutError('CPU replay budget exhausted')
                    if (source_row['truth'] != len(record['points']) or source_row['truth'] != record['truth']
                            or source_row['sampling_unit'] != record['image_sha256']
                            or (source_row['height'], source_row['width']) != (record['height'], record['width'])):
                        raise ValueError('Prediction and manifest differ')
                    arrays = []
                    for method in ('b0', 'final'):
                        path = root/'work/fscd-lvis-external-v1/full'/(source_row['sample_id']+'-'+method+'.npz')
                        if sha(path) != source_row[method]['density_file_sha256']:
                            raise ValueError('Source density changed')
                        with np.load(path, allow_pickle=False) as saved:
                            arrays.append(saved['density'])
                    if any(a.shape != (record['height'], record['width']) for a in arrays):
                        raise ValueError('Source density size mismatch')
                    n, density = half_blend(source_row['b0']['prediction'], source_row['final']['prediction'], *arrays)
                    points = points_for_game(record['points'], record['height'], record['width'])
                    new = {'prediction': n}
                    for level in (1, 2):
                        value = game(density, points, level)
                        checked = grid_error(density, points, level)
                        delta = abs(value-checked)
                        max_grid_error = max(max_grid_error, delta)
                        if not math.isclose(value, checked, rel_tol=1e-10, abs_tol=1e-8):
                            raise ValueError('Independent spatial metric disagreement')
                        new['game'+str(level)] = value
                    mass_error = abs(float(density.sum())-n)
                    max_mass_error = max(max_mass_error, mass_error)
                    row = {k: source_row[k] for k in ('sample_id', 'sampling_unit', 'truth', 'route')}
                    for old, name in (('b0', 'b0'), ('final', 'original_final')):
                        row[name] = {k: source_row[old][k] for k in ('prediction', 'game1', 'game2')}
                    row['half_blend'] = new
                    row['density_array_sha256'] = hashlib.sha256(np.ascontiguousarray(density, dtype='<f8').tobytes()).hexdigest()
                    row['source_density_sha256'] = [source_row[m]['density_file_sha256'] for m in ('b0', 'final')]
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    rows.append(row)
                    if index % 100 == 0:
                        print(json.dumps({'processed': index, 'samples': len(source_rows),
                                          'elapsed_seconds': time.monotonic()-start}), flush=True)
            result = count_summary(rows, cfg['bootstrap_replicates'], cfg['bootstrap_seed'])
            for method in METHODS:
                for level in (1, 2):
                    key = 'game'+str(level)
                    result['methods'][method][key+'_mean'] = math.fsum(r[method][key] for r in rows)/len(rows)
            independent_counts(rows, result)
            for old, name in (('b0', 'b0'), ('final', 'original_final')):
                for metric, value in result['methods'][name].items():
                    if not math.isclose(value, original[old][metric], rel_tol=1e-12, abs_tol=1e-10):
                        raise ValueError('Original aggregate changed')
            if time.monotonic()-start >= cfg['cpu_wall_seconds_limit']:
                raise TimeoutError('CPU replay budget exhausted')
            result.update({'status': 'complete', 'source_commit': receipt['code_commit'],
                'evidence_class': cfg['evidence_class'], 'test_was_previously_observed': True,
                'routes': dict(Counter(r['route'] for r in rows)), 'weight_b0': .5,
                'weight_original_final': .5, 'candidate_count': 1, 'gpu_used': False,
                'gpu_hours': 0, 'elapsed_seconds': time.monotonic()-start,
                'timing_scope': 'CPU saved-map replay, not live model inference',
                'private_predictions_sha256': sha(row_path),
                'original_predictions_sha256': cfg['original_files_sha256']['predictions.jsonl'],
                'manifest_sha256': cfg['manifest_sha256']})
            write_new(out/'summary.json', result)
            write_new(out/'audit.json', {'status': 'passed', 'samples': len(rows),
                'source_commit': receipt['code_commit'], 'source_receipts_verified': 7,
                'source_density_files_verified': 2*len(rows),
                'original_evidence_sha256': cfg['original_files_sha256'],
                'manifest_sha256': cfg['manifest_sha256'],
                'summary_sha256': sha(out/'summary.json'), 'predictions_sha256': sha(row_path),
                'max_mass_error': max_mass_error, 'max_independent_game_difference': max_grid_error,
                'count_metrics_independently_recomputed': True,
                'spatial_metrics_independently_recomputed': True,
                'no_images_excluded': True, 'no_ground_truth_input_to_blender': True,
                'gpu_used': False, 'automatic_retry': False})
            print('HALF_BLEND_REPLAY_AUDIT_PASSED', len(rows), flush=True)
        except BaseException as exc:
            write_new(out/'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                'automatic_retry': False, 'elapsed_seconds': time.monotonic()-start})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    replay(parser.parse_args().root.resolve())
