"""CPU replay of metadata, saved maps, detections and paired metrics."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from fscd_lvis_common import checked_sources, read, sha, write_new, align_annotations, annotation_path
from planb.density import boxes_to_density, points_for_game
from planb.fscd_lvis_metrics import report


def rebuilt_density(row, method):
    h, w = row['height'], row['width']
    value = row[method]
    if method == 'b0' or row['route'] == 'M3-v2-g3':
        if value['prediction'] != len(value['boxes_xyxy']):
            raise ValueError('Detection count mismatch')
        return boxes_to_density(value['boxes_xyxy'], h, w)
    if row['route'] != 'T2':
        raise ValueError('Unknown final route')
    views = row['detail']['method_detail']['views']
    if len(views) != 2 or [x['flipped'] for x in views] != [False, True] or any(x['scale_factor'] != 1 for x in views):
        raise ValueError('T2 view definition changed')
    if views[0]['boxes_original_xyxy'] != row['b0']['boxes_xyxy']:
        raise ValueError('Original view does not reproduce B0')
    if any(v['count'] != len(v['boxes_original_xyxy']) for v in views):
        raise ValueError('TTA detection count mismatch')
    if value['prediction'] != sum(v['count'] for v in views)/2:
        raise ValueError('TTA mean count mismatch')
    return sum(boxes_to_density(v['boxes_original_xyxy'], h, w) for v in views)/2


def grid_error(density, points, level):
    h, w = density.shape
    side = 2**level
    ys, xs = np.arange(side+1)*h//side, np.arange(side+1)*w//side
    targets, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=(ys, xs))
    return math.fsum(abs(float(density[ys[y]:ys[y+1], xs[x]:xs[x+1]].sum())-targets[y,x])
                     for y in range(side) for x in range(side))


def audit(root, phase):
    if os.environ.get('SLURM_JOB_PARTITION') != 'amd' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Audit requires a CPU job with CUDA hidden')
    checked_sources(root, 'fscd_lvis_prepare_receipt_v1.json')
    receipt = checked_sources(root, 'fscd_lvis_eval_receipt_v1.json')
    checked_sources(root, 'fscd_lvis_freeze_receipt_v1.json')
    start = time.monotonic()
    out = root/'results/fscd-lvis-external-v1'/phase
    if (out/'audit.json').exists() or (out/'audit_failure.json').exists():
        raise FileExistsError('Audit evidence already exists')
    try:
        freeze = read(root/'configs/fscd_lvis_manifest_freeze_v1.json')
        manifest_path = root/'data/fscd-lvis-v1/evaluation_manifest.json'
        if sha(manifest_path) != freeze['manifest_sha256']:
            raise ValueError('Private manifest changed')
        manifest = read(manifest_path)
        cfg = read(root/'configs/fscd_lvis_external_v1.json')
        summary = read(out/'summary.json')
        protocol = read(out/'protocol.json')
        if summary['status'] != 'complete' or summary['source_commit'] != receipt['code_commit']:
            raise ValueError('Inference has not completed on this source')
        if (summary['manifest_sha256'] != freeze['manifest_sha256']
                or protocol['manifest_sha256'] != freeze['manifest_sha256']
                or protocol['source_commit'] != receipt['code_commit']
                or summary['samples'] != freeze['split_counts'][phase]
                or protocol['samples'] != freeze['split_counts'][phase]):
            raise ValueError('Inference provenance differs from frozen scope')
        if not summary['model_state_unchanged']:
            raise ValueError('Model changed during inference')
        inventory = read(root/'results/fscd-lvis-assets-v1/archive_inventory.json')
        split = 'train' if phase == 'smoke' else 'test'
        count_path = annotation_path(root, inventory, f'unseen_count_{split}.json')
        instance_path = annotation_path(root, inventory, f'unseen_instances_{split}.json')
        originals = align_annotations(read(count_path), read(instance_path), split)
        if phase == 'smoke': originals = originals[:20]
        records = manifest['splits'][phase]
        rows = [json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
        expected_ids = [r['sample_id'] for r in originals]
        if len(records) != freeze['split_counts'][phase] or expected_ids != [r['sample_id'] for r in rows] or expected_ids != [r['sample_id'] for r in records]:
            raise ValueError('Incomplete, duplicated or reordered evaluation')
        maximum_map_difference = 0.
        for original, record, row in zip(originals, records, rows):
            for field in ('width', 'height', 'exemplars_xyxy', 'points', 'truth'):
                if original[field] != record[field]:
                    raise ValueError('Manifest differs from annotation source')
            if sha(root/record['image_path']) != record['image_sha256'] or row['sampling_unit'] != record['image_sha256']:
                raise ValueError('Input identity mismatch')
            if row['truth'] != len(original['points']) or (row['width'], row['height']) != (original['width'], original['height']):
                raise ValueError('Wrong target count or dimensions')
            points = points_for_game(original['points'], row['height'], row['width'])
            expected_route = 'M3-v2-g3' if row['detail']['median_exemplar_side_model_pixels'] <= 32 or row['b0']['prediction'] >= 200 else 'T2'
            if row['route'] != expected_route or not row['detail']['original_view_matches_b0']:
                raise ValueError('Frozen routing or original-view check differs')
            if phase == 'smoke' and not row['detail']['native_adapter_gate_passed']:
                raise ValueError('Small-batch native adapter gate missing')
            for method in ('b0', 'final'):
                if not 0 <= row[method]['peak_allocated_mib'] <= cfg['max_allocated_mib']:
                    raise ValueError('Registered memory limit exceeded')
                path = root/'work/fscd-lvis-external-v1'/phase/(row['sample_id']+'-'+method+'.npz')
                if sha(path) != row[method]['density_file_sha256']:
                    raise ValueError('Density file changed')
                with np.load(path, allow_pickle=False) as data:
                    density = data['density']
                rebuilt = rebuilt_density(row, method)
                if density.shape != rebuilt.shape or not np.isfinite(density).all() or (density < 0).any():
                    raise ValueError('Invalid density array')
                delta = float(np.max(np.abs(density-rebuilt)))
                maximum_map_difference = max(maximum_map_difference, delta)
                if not np.allclose(density, rebuilt, rtol=1e-12, atol=1e-8):
                    raise ValueError('Density differs from detection reconstruction')
                if abs(float(density.sum())-row[method]['prediction']) > 1e-6:
                    raise ValueError('Density mass differs from count')
                for level in (1, 2):
                    if not np.isclose(grid_error(density, points, level), row[method]['game'+str(level)], rtol=1e-10, atol=1e-8):
                        raise ValueError('Spatial metric disagreement')
        computed = report(rows, cfg['bootstrap_replicates'], cfg['bootstrap_seed'])
        for method in ('b0', 'final'):
            errors = [row[method]['prediction']-row['truth'] for row in rows]
            independent = {'mae': math.fsum(abs(e) for e in errors)/len(errors),
                           'rmse': math.sqrt(math.fsum(e*e for e in errors)/len(errors))}
            for metric, value in independent.items():
                if not math.isclose(value, summary[method][metric], rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError('Independent count metric disagreement')
        if any(computed[key] != summary[key] for key in computed):
            raise ValueError('Summary or fixed bootstrap replay differs')
        write_new(out/'audit.json', {'status': 'passed', 'phase': phase,
                  'source_commit': receipt['code_commit'], 'samples': len(rows),
                  'manifest_sha256': freeze['manifest_sha256'],
                  'predictions_sha256': sha(out/'predictions.jsonl'), 'summary_sha256': sha(out/'summary.json'),
                  'density_max_abs_reconstruction_difference': maximum_map_difference,
                  'count_metrics_independently_recomputed': True,
                  'bootstrap_replayed_with_same_fixed_implementation': True,
                  'elapsed_seconds': time.monotonic()-start, 'gpu_used': False})
        print('FSCD_LVIS_AUDIT_PASSED', phase, len(rows), flush=True)
    except BaseException as exc:
        write_new(out/'audit_failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'message': str(exc), 'automatic_retry': False, 'elapsed_seconds': time.monotonic()-start})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'full'], required=True)
    a = p.parse_args()
    audit(a.root.resolve(), a.phase)
