"""Posthoc read-only hooks for two completed M3 cases; never changes policy."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.inference_variants import ExperimentEngine
from run_m2_m3 import load_model
from evaluate_val import PIN, digest


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root
    upstream = root / 'external/GECO2'
    assert subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    sys.path.insert(0, str(upstream)); os.environ['TORCH_HOME'] = str(root / 'cache/torch')
    torch.manual_seed(42); torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    d = root / 'results/m2-m3-v1/val-m3'
    analysis = json.loads((d / 'analysis.json').read_text())
    protocol = json.loads((d / 'protocol.json').read_text())
    assert analysis['status'] == 'passed'
    for file, value in protocol['code_sha256'].items(): assert digest(root / file) == value
    config = json.loads((root / 'configs/m2_m3_v1.json').read_text())
    assert digest(root / 'configs/m2_m3_v1.json') == protocol['config_sha256']
    rows = {r['image']: r for r in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines())}
    selected = list(dict.fromkeys([analysis['largest_regressions'][0]['image'], analysis['largest_improvements'][0]['image']]))
    annotations = json.loads((root / 'data/fsc147/annotations/annotation_FSC147_384.json').read_text())
    model = load_model(root); engine = ExperimentEngine(model, config)
    calls = []
    def capture(module, inputs):
        outputs = inputs[1]
        # Match the exact quantity used by the pinned SAMMaskProcessor branch.
        count = len(outputs[0]['pred_boxes'][0])
        assert count == outputs[0]['box_v'].numel()
        calls.append({'pre_refinement_candidates': count, 'upstream_skips_sam_mask_refinement': count > 800})
    handle = model.sam_mask.register_forward_pre_hook(capture)
    result = []
    with torch.inference_mode():
        for name in selected:
            calls.clear()
            with Image.open(root / 'data/fsc147/images_384_VarV2' / name) as im: image = T.ToTensor()(im.convert('RGB'))
            exemplars = torch.tensor(annotations[name]['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
            boxes, base, meta = engine.run(image, exemplars, 'm3')
            assert np.array_equal(boxes, np.asarray(rows[name]['boxes_xyxy'], dtype=np.float32).reshape(-1, 4))
            result.append({'image': name, 'truth': rows[name]['truth'], 'baseline_prediction': len(base),
                           'm3_prediction': len(boxes), 'saved_output_bitwise_equal': True,
                           'calls_in_order_global_then_tiles': list(calls), 'tile_details': meta.get('tiles', []),
                           'before_merge_count': meta.get('pre_merge_count'),
                           'removed_by_cross_tile_nms': meta.get('cross_tile_removed')})
    handle.remove()
    record = {'status': 'passed', 'scope': 'posthoc_two_cases; no parameter changes or causal proof',
              'script_sha256': digest(Path(__file__)), 'cases': result}
    (root / 'results/m2-m3-v1/trace_diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
