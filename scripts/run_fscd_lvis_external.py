"""One frozen B0/final paired pass, with truth used only after inference."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from fscd_lvis_common import checked_sources, read, sha, write_new


def evaluate(root, phase):
    checked_sources(root, 'fscd_lvis_prepare_receipt_v1.json')
    receipt = checked_sources(root, 'fscd_lvis_eval_receipt_v1.json')
    checked_sources(root, 'fscd_lvis_freeze_receipt_v1.json')
    cfg = read(root/'configs/fscd_lvis_external_v1.json')
    freeze = read(root/'configs/fscd_lvis_manifest_freeze_v1.json')
    manifest_path = root/'data/fscd-lvis-v1/evaluation_manifest.json'
    if sha(manifest_path) != freeze['manifest_sha256']:
        raise ValueError('Manifest differs from public pre-inference freeze')
    manifest = read(manifest_path)
    records = manifest['splits'][phase]
    if len(records) != freeze['split_counts'][phase]:
        raise ValueError('Evaluation scope differs')
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'gpu-l20':
        raise RuntimeError('An allocated school GPU is required')
    if phase == 'full':
        smoke = read(root/'results/fscd-lvis-external-v1/smoke/summary.json')
        audit = read(root/'results/fscd-lvis-external-v1/smoke/audit.json')
        if (audit['status'] != 'passed' or smoke['status'] != 'complete'
                or smoke['source_commit'] != receipt['code_commit']
                or audit['source_commit'] != receipt['code_commit']
                or audit['summary_sha256'] != sha(root/'results/fscd-lvis-external-v1/smoke/summary.json')
                or audit['manifest_sha256'] != freeze['manifest_sha256']):
            raise ValueError('Small-batch correctness gate has not passed')
        estimate = 1.5 * smoke['elapsed_seconds'] / freeze['split_counts']['smoke'] * len(records) + 300
        if estimate > cfg['full_gpu_seconds']:
            raise RuntimeError('Full pass does not fit pre-registered time estimate')
    out = root/'results/fscd-lvis-external-v1'/phase
    out.mkdir(parents=True, exist_ok=False)
    maps = root/'work/fscd-lvis-external-v1'/phase
    maps.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    limit = cfg['smoke']['gpu_seconds'] if phase == 'smoke' else cfg['full_gpu_seconds']
    def stop(signum, frame):
        raise TimeoutError('Stage time limit or scheduler termination')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(limit-30)
    try:
        os.environ['TORCH_HOME'] = str(root/'cache/torch')
        sys.path[:0] = [str(root/'external/GECO2'), str(root), str(root/'scripts')]
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms as T
        from hpc4_port import asset_records
        from planb.round2.common import seed_everything, state_sha
        from run_m2_m3 import load_model
        from planb.final_inference import FinalEngine
        from planb.density import game, points_for_game
        from planb.fscd_lvis_metrics import report
        if not torch.cuda.is_available() or 'L20' not in torch.cuda.get_device_name(0):
            raise RuntimeError('Registered GPU is unavailable')
        for record in asset_records(root):
            if sha(root/record['path']) != record['sha256']:
                raise ValueError('Base asset changed')
        final_cfg = read(root/'configs/final_v1.json')
        if sha(root/final_cfg['method_config']) != final_cfg['method_config_sha256']:
            raise ValueError('Frozen method configuration changed')
        upstream = root/'external/GECO2'
        pin = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
        if pin != final_cfg['upstream_commit']:
            raise ValueError('Upstream revision changed')
        for diff in (['diff', '--quiet'], ['diff', '--cached', '--quiet']):
            subprocess.run(['git', '-C', str(upstream), *diff], check=True)
        seed_everything(42)
        torch.set_num_threads(4)
        model = load_model(root)
        model_before = state_sha(model)
        engine = FinalEngine(model, read(root/final_cfg['method_config']))
        write_new(out/'protocol.json', {'phase': phase, 'source_commit': receipt['code_commit'],
                  'manifest_sha256': freeze['manifest_sha256'], 'samples': len(records),
                  'model_state_sha256': model_before, 'gpu': torch.cuda.get_device_name(0),
                  'upstream_commit': pin,
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'precision': 'FP32', 'tf32': False, 'batch_size': 1,
                  'truth_supplied_to_model': False, 'time_limit_seconds': limit})
        rows = []
        with (out/'predictions.jsonl').open('x') as stream:
            for record in records:
                path = root/record['image_path']
                if sha(path) != record['image_sha256']:
                    raise ValueError('Input image changed')
                with Image.open(path) as im:
                    image = T.ToTensor()(im.convert('RGB'))
                prompts = torch.tensor(record['exemplars_xyxy'], dtype=torch.float32)
                b, f, bd, fd, detail = engine.run_pair(image, prompts, gate=phase=='smoke')
                # Targets are accessed after the two predictions are fixed.
                h, w = image.shape[-2:]
                points = points_for_game(record['points'], h, w)
                row = {'sample_id': record['sample_id'], 'sampling_unit': record['image_sha256'],
                       'width': w, 'height': h, 'truth': len(points),
                       'route': detail['route'], 'detail': detail}
                for label, result, density in [('b0', b, bd), ('final', f, fd)]:
                    if not np.isfinite(density).all() or (density < 0).any() or abs(float(density.sum())-result['prediction']) > 1e-6:
                        raise ValueError('Invalid density or count mass')
                    if result['peak_allocated_mib'] > cfg['max_allocated_mib']:
                        raise MemoryError('Registered memory limit exceeded')
                    map_path = maps/(record['sample_id']+'-'+label+'.npz')
                    np.savez_compressed(map_path, density=density)
                    result.update(density_mass=float(density.sum()), game1=game(density, points, 1),
                                  game2=game(density, points, 2), density_file_sha256=sha(map_path))
                    row[label] = result
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                progress = out/'progress.tmp'
                progress.write_text(json.dumps({'done': len(rows), 'total': len(records),
                                               'elapsed_seconds': time.monotonic()-start})+'\n')
                progress.replace(out/'progress.json')
                if len(rows) % 25 == 0 or len(rows) == len(records):
                    print('completed', len(rows), '/', len(records), flush=True)
        if state_sha(model) != model_before:
            raise ValueError('Frozen parameters changed')
        write_new(out/'summary.json', {'status': 'complete', 'phase': phase,
                  'source_commit': receipt['code_commit'], 'manifest_sha256': freeze['manifest_sha256'],
                  'elapsed_seconds': time.monotonic()-start, 'model_state_unchanged': True,
                  'deployment_gate_only': phase=='smoke',
                  **report(rows, cfg['bootstrap_replicates'], cfg['bootstrap_seed'])})
    except BaseException as exc:
        write_new(out/'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'elapsed_seconds': time.monotonic()-start, 'automatic_retry': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['smoke', 'full'], required=True)
    args = parser.parse_args()
    evaluate(args.root.resolve(), args.phase)
