"""Pinned GeCo2 counting on FSC-147 validation images, with M1 spatial checks.

Uses upstream preprocessing and the threshold/NMS policy in inference.py.
Ground truth is the exact number of annotation points, not a density-map sum.
No test-set execution or training is provided by this entry point.
"""
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
from torchvision import ops, transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.density import boxes_to_density, game, points_for_game

PIN = '67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def metrics(rows):
    if not rows:
        return {'n': 0}
    errors = np.array([r['prediction'] - r['truth'] for r in rows])
    gt = np.array([r['truth'] for r in rows])
    return {'n': len(rows), 'mae': float(np.abs(errors).mean()),
            'rmse': float(np.sqrt(np.square(errors).mean())),
            'truth_total': int(gt.sum()), 'truth_min': int(gt.min()),
            'truth_median': float(np.median(gt)), 'truth_mean': float(gt.mean()),
            'truth_max': int(gt.max()),
            'game1': float(np.mean([r['game1'] for r in rows])),
            'game2': float(np.mean([r['game2'] for r in rows]))}


def final_boxes(output, width, height):
    raw = output['pred_boxes']
    v = output['box_v']
    if raw.numel() == 0:
        return raw.reshape(0, 4)
    if not torch.isfinite(raw).all() or not torch.isfinite(v).all():
        raise ValueError('Non-finite prediction')
    mask = v > v.max() * .11
    boxes, scores = raw[mask], v[mask]
    keep = ops.nms(boxes, scores, .5)
    boxes = boxes[keep].clamp(0, 1)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = (centers[:, 0] * 1024 < width) & (centers[:, 1] * 1024 < height)
    return boxes[valid]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20, help='0 means all 1286 validation images')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--save-examples', type=int, default=3)
    args = p.parse_args()
    if args.limit < 0 or args.batch_size < 1:
        p.error('Invalid limit or batch size')
    root = args.root.resolve()
    upstream = root / 'external/GECO2'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise RuntimeError('Unexpected upstream revision')
    os.environ['TORCH_HOME'] = str(root / 'cache/torch')
    sys.path.insert(0, str(upstream))
    from models.counter_infer import build_model
    from utils.arg_parser import get_argparser
    from utils.data import resize_and_pad

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'predictions.jsonl').exists():
        raise FileExistsError('Choose a fresh output directory; existing evidence is preserved')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_args = get_argparser().parse_args([])
    model_args.kernel_dim = 1
    model_args.training = False
    model = build_model(model_args)
    weights = root / 'weights/GECO2FSCD.pth'
    state = torch.load(weights, map_location='cpu', weights_only=True)['model']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    mismatch = model.load_state_dict(state, strict=False)
    # Inference SAM refinement is initialized separately from official SAM2.
    # The FSC training checkpoint has auxiliary heads, unused at inference.
    invalid_missing = [k for k in mismatch.missing_keys if not k.startswith('sam_mask.')]
    invalid_extra = [k for k in mismatch.unexpected_keys
                     if not k.startswith(('class_embed_aux.', 'bbox_embed_aux.'))]
    load_report = {'missing_keys': mismatch.missing_keys, 'unexpected_keys': mismatch.unexpected_keys,
                   'invalid_missing': invalid_missing, 'invalid_extra': invalid_extra,
                   'fsc_sha256': digest(weights), 'sam2_sha256': digest(root / 'cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt'),
                   'matched_keys': len(model.state_dict()) - len(mismatch.missing_keys)}
    write_json(args.output / 'checkpoint_load.json', load_report)
    if invalid_missing or invalid_extra:
        raise RuntimeError('Unexplained checkpoint mismatch; see checkpoint_load.json')
    del state
    model = model.eval().cuda()
    annotations_dir = root / 'data/fsc147/annotations'
    splits = json.loads((annotations_dir / 'Train_Test_Val_FSC_147.json').read_text())
    assert len(splits['val']) == 1286 and len(set(splits['val'])) == 1286
    assert not set(splits['val']) & (set(splits['train']) | set(splits['test']))
    names = splits['val'][:args.limit] if args.limit else splits['val']
    annotations = json.loads((annotations_dir / 'annotation_FSC147_384.json').read_text())
    normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])
    provenance = {'split': 'val', 'selection': 'official_order_prefix' if args.limit else 'all',
                  'planned_n': len(names), 'batch_size': args.batch_size, 'precision': 'FP32',
                  'tf32': False, 'upstream_commit': PIN, 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                  'ground_truth': 'exact_annotation_point_count', 'relative_threshold': .11, 'nms_iou': .5,
                  'game_boundary_policy': 'clip_exact_edges_and_float64_roundoff_up_to_8_ulps;reject_larger_overflow',
                  'runner_sha256': digest(Path(__file__)),
                  'density_module_sha256': digest(Path(__file__).resolve().parents[1] / 'planb/density.py'),
                  'annotation_sha256': digest(annotations_dir / 'annotation_FSC147_384.json'),
                  'split_sha256': digest(annotations_dir / 'Train_Test_Val_FSC_147.json'),
                  'checkpoint_sha256': load_report['fsc_sha256'], 'ids': names}
    write_json(args.output / 'protocol.json', provenance)
    rows = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (args.output / 'predictions.jsonl').open('x') as stream, torch.inference_mode():
        for offset in range(0, len(names), args.batch_size):
            inputs, prompts, metas = [], [], []
            for name in names[offset:offset + args.batch_size]:
                anno = annotations[name]
                image_path = root / 'data/fsc147/images_384_VarV2' / name
                with Image.open(image_path) as image:
                    rgb = image.convert('RGB')
                    width, height = rgb.size
                    image_tensor = T.ToTensor()(rgb)
                boxes = torch.tensor(anno['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
                image_tensor, boxes, scale = resize_and_pad(image_tensor, boxes)
                scale = float(scale)
                inputs.append(normalize(image_tensor))
                prompts.append(boxes)
                metas.append((name, width, height, scale, anno))
            x = torch.stack(inputs).cuda()
            boxes = torch.stack(prompts).cuda()
            torch.cuda.synchronize()
            tick = time.perf_counter()
            outputs, refs, centerness, coords, masks = model(x, boxes)
            torch.cuda.synchronize()
            batch_seconds = time.perf_counter() - tick
            for idx, (name, w, h, scale, anno) in enumerate(metas):
                boxes_original = final_boxes(outputs[idx], int(w * scale), int(h * scale)).cpu().numpy() * (1024 / scale)
                density = boxes_to_density(boxes_original, h, w)
                original_points = np.asarray(anno['points'], dtype=np.float64).reshape(-1, 2)
                points = points_for_game(original_points, h, w)
                adjusted = int(np.any(points != original_points, axis=1).sum())
                mass = float(density.sum())
                if not np.isclose(mass, len(boxes_original), atol=1e-8, rtol=0):
                    raise ValueError('M1 mass is inconsistent with detection count')
                row = {'image': name, 'truth': len(points), 'prediction': len(boxes_original),
                       'game_boundary_adjusted_points': adjusted,
                       'density_mass': mass, 'game1': game(density, points, 1),
                       'game2': game(density, points, 2), 'inference_seconds_per_image': batch_seconds / len(metas),
                       'width': w, 'height': h, 'scale_factor': scale, 'boxes_xyxy': boxes_original.tolist()}
                rows.append(row)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if len(rows) <= args.save_examples:
                    np.savez_compressed(args.output / (Path(name).stem + '_m1.npz'), density=density.astype(np.float32))
            del outputs, refs, centerness, coords, masks, x, boxes
            summary = {'status': 'complete' if len(rows) == len(names) else 'running',
                       'scope': 'full_validation' if len(names) == 1286 else 'smoke_only',
                       'game_boundary_adjusted_points': sum(r['game_boundary_adjusted_points'] for r in rows),
                       'overall': metrics(rows), 'groups': {
                           '<50': metrics([r for r in rows if r['truth'] < 50]),
                           '50-299': metrics([r for r in rows if 50 <= r['truth'] < 300]),
                           '>=300': metrics([r for r in rows if r['truth'] >= 300])},
                       'elapsed_seconds': time.perf_counter() - start,
                       'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                       'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20}
            write_json(args.output / 'summary.json', summary)
            print(json.dumps({'done': len(rows), 'total': len(names), 'mae': summary['overall']['mae'],
                              'peak_mib': summary['peak_allocated_mib']}), flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
