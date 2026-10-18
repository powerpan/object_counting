"""Val-only, preregistered frozen-weight experiments with a baseline gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.density import boxes_to_density, points_for_game, game
from planb.inference_variants import ExperimentEngine
from planb.geco_adapter import postprocess
from evaluate_val import PIN, digest, write_json, metrics


def smoke_ids(names, baseline):
    selected = []
    for lo, hi in ((0, 50), (50, 300), (300, float('inf'))):
        selected.extend([name for name in names if lo <= baseline[name]['truth'] < hi][:8])
    selected.extend(sorted(names, key=lambda name: -baseline[name]['truth'])[:4])
    selected = list(dict.fromkeys(selected))
    for name in names:
        if len(selected) >= 32:
            break
        if name not in selected:
            selected.append(name)
    return selected


def load_model(root):
    from models.counter_infer import build_model
    from utils.arg_parser import get_argparser
    a = get_argparser().parse_args([])
    a.kernel_dim = 1; a.training = False
    model = build_model(a)
    state = torch.load(root / 'weights/GECO2FSCD.pth', map_location='cpu', weights_only=True)['model']
    mismatch = model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()}, strict=False)
    if any(not k.startswith('sam_mask.') for k in mismatch.missing_keys) or any(
            not k.startswith(('class_embed_aux.', 'bbox_embed_aux.')) for k in mismatch.unexpected_keys):
        raise RuntimeError('Unexpected weight mismatch')
    model.requires_grad_(False)
    return model.eval().cuda()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['gate', 'smoke', 'full'], required=True)
    p.add_argument('--variant', choices=['m2', 'm2-repeat', 'm3', 'm3-repeat', 'm2-m3'], default='m2')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'predictions.jsonl').exists() or (out / 'gate.json').exists():
        raise FileExistsError('Existing evidence must not be overwritten')
    upstream = root / 'external/GECO2'
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet'], check=True)
    os.environ['TORCH_HOME'] = str(root / 'cache/torch')
    sys.path.insert(0, str(upstream))
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg_path = root / 'configs/m2_m3_v1.json'
    config = json.loads(cfg_path.read_text())
    baseline_dir = root / 'results/b0-val-b1-boundary-r1'
    b_protocol = json.loads((baseline_dir / 'protocol.json').read_text())
    baseline = {r['image']: r for r in map(json.loads, (baseline_dir / 'predictions.jsonl').read_text().splitlines())}
    ann_dir = root / 'data/fsc147/annotations'
    ann_file = ann_dir / 'annotation_FSC147_384.json'
    split_file = ann_dir / 'Train_Test_Val_FSC_147.json'
    assert digest(ann_file) == b_protocol['annotation_sha256']
    assert digest(split_file) == b_protocol['split_sha256']
    assert digest(root / 'weights/GECO2FSCD.pth') == b_protocol['checkpoint_sha256']
    load = json.loads((baseline_dir / 'checkpoint_load.json').read_text())
    assert digest(root / 'cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt') == load['sam2_sha256']
    splits = json.loads(split_file.read_text())
    names = splits['val']; assert names == b_protocol['ids'] and len(names) == 1286
    assert not set(names) & (set(splits['test']) | set(splits['train']))
    if args.phase != 'full': names = smoke_ids(names, baseline)
    annotations = json.loads(ann_file.read_text())
    code_files = ['scripts/run_m2_m3.py', 'planb/geco_adapter.py', 'planb/inference_variants.py',
                  'planb/geometry.py', 'planb/density.py', 'scripts/evaluate_val.py']
    protocol = {'phase': args.phase, 'variant': args.variant, 'split': 'val', 'ids': names,
                'planned_n': len(names), 'config': config, 'config_sha256': digest(cfg_path),
                'code_sha256': {name: digest(root / name) for name in code_files},
                'baseline_predictions_sha256': digest(baseline_dir / 'predictions.jsonl'),
                'checkpoint_sha256': load['fsc_sha256'], 'sam2_sha256': load['sam2_sha256'],
                'annotation_sha256': digest(ann_file), 'split_sha256': digest(split_file),
                'upstream_commit': PIN, 'precision': 'FP32', 'tf32': False,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'ground_truth_used_by_policy': False, 'batch_size': 1}
    write_json(out / 'protocol.json', protocol)
    model = load_model(root); engine = ExperimentEngine(model, config)
    start = time.perf_counter(); rows = []; gate_rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), (out / 'predictions.jsonl').open('x') as stream:
        for name in names:
            anno = annotations[name]
            with Image.open(root / 'data/fsc147/images_384_VarV2' / name) as im:
                image = T.ToTensor()(im.convert('RGB'))
            h, w = image.shape[-2:]
            exemplars = torch.tensor(anno['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
            torch.cuda.synchronize(); tick = time.perf_counter()
            if args.phase == 'gate':
                x, prompts, scale = engine.prepare(image, exemplars)
                original = model(x, prompts)
                adapter, feats, app = engine.adapter.forward(x, prompts)
                for key in ('pred_boxes', 'box_v', 'scores'):
                    if not torch.equal(original[0][0][key], adapter[key]):
                        raise RuntimeError('Adapter is not bitwise equivalent: ' + key)
                boxes, _ = postprocess(adapter, int(w * scale), int(h * scale))
                boxes = boxes.cpu().numpy() * (1024 / scale)
                if not np.array_equal(boxes, np.asarray(baseline[name]['boxes_xyxy'], dtype=np.float32).reshape(-1, 4)):
                    raise RuntimeError('Adapter does not reproduce saved baseline boxes')
                gate_rows.append({'image': name, 'count': len(boxes), 'bitwise_equal': True})
                del original, adapter, feats, app, x, prompts
                engine.check_memory()
                print(json.dumps({'gate_done': len(gate_rows), 'total': len(names)}), flush=True)
                continue
            boxes, first_boxes, detail = engine.run(image, exemplars, args.variant)
            torch.cuda.synchronize(); seconds = time.perf_counter() - tick
            expected = np.asarray(baseline[name]['boxes_xyxy'], dtype=np.float32).reshape(-1, 4)
            if not np.array_equal(first_boxes, expected):
                raise RuntimeError('First pass changed relative to B0')
            # Ground truth enters only after the inference policy has finished.
            points = points_for_game(anno['points'], h, w)
            density = boxes_to_density(boxes, h, w)
            if not np.isclose(density.sum(), len(boxes), atol=1e-8, rtol=0):
                raise RuntimeError('Density mass is not preserved')
            for candidate in detail['pseudo_exemplars']:
                x0, y0, x1, y1 = candidate['box_xyxy']
                candidate['annotated_points_inside'] = int(((points[:, 0] >= x0) & (points[:, 0] <= x1)
                    & (points[:, 1] >= y0) & (points[:, 1] <= y1)).sum())
            row = {'image': name, 'truth': len(points), 'prediction': len(boxes), 'width': w, 'height': h,
                   'density_mass': float(density.sum()), 'game1': game(density, points, 1),
                   'game2': game(density, points, 2), 'inference_seconds_per_image': seconds,
                   'boxes_xyxy': boxes.tolist(), 'detail': detail}
            rows.append(row); stream.write(json.dumps(row) + '\n'); stream.flush()
            summary = {'status': 'complete' if len(rows) == len(names) else 'running',
                       'phase': args.phase, 'variant': args.variant, 'overall': metrics(rows),
                       'groups': {'<50': metrics([r for r in rows if r['truth'] < 50]),
                                  '50-299': metrics([r for r in rows if 50 <= r['truth'] < 300]),
                                  '>=300': metrics([r for r in rows if r['truth'] >= 300])},
                       'm2_triggered_images': sum(r['detail']['m2_triggered'] for r in rows),
                       'm3_triggered_images': sum(r['detail']['m3_triggered'] for r in rows),
                       'elapsed_seconds': time.perf_counter() - start,
                       'inference_seconds_mean': float(np.mean([r['inference_seconds_per_image'] for r in rows])),
                       'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                       'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
            write_json(out / 'summary.json', summary)
            if len(rows) % 25 == 0 or len(rows) == len(names):
                print(json.dumps({'done': len(rows), 'total': len(names), 'variant': args.variant,
                                  'mae': summary['overall']['mae'], 'peak_mib': summary['peak_allocated_mib']}), flush=True)
    if args.phase == 'gate':
        write_json(out / 'gate.json', {'status': 'passed', 'n': len(gate_rows), 'rows': gate_rows,
                                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20})


if __name__ == '__main__':
    main()
