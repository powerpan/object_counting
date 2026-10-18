"""Gate and complete val inference for a completed, frozen M4 checkpoint."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup, execution_record, sha, state_sha, write_json, utc, check_memory, verify_inputs
from planb.round2.adapter import Round2Adapter
from planb.round2.data import Inputs
from planb.round2.head import DensityHead
from planb.round2.spatial import restore_density, cell_mass, point_cells, grid_features, localization, game_from_cells
from planb.density import boxes_to_density, points_for_game, game
from planb.geco_adapter import postprocess
from run_m2_m3 import load_model
from audit_b0_results import aggregate, read_rows


def analysis_record(root, stage):
    record = execution_record(root, stage)
    receipt = json.loads((root/'provenance/round2_analysis_receipt.json').read_text())
    if not receipt['remote_main_verified']:
        raise ValueError('Analysis code has not been verified on main')
    for name, value in receipt['files_sha256'].items():
        if sha(root/name) != value:
            raise ValueError('Analysis code differs from committed receipt')
    record['analysis_code'] = receipt
    return record


def summarize(rows):
    groups = {'<50': [r for r in rows if r['truth'] < 50],
              '50-299': [r for r in rows if 50 <= r['truth'] < 300],
              '>=300': [r for r in rows if r['truth'] >= 300]}
    return {'overall': aggregate(rows), 'groups': {k: aggregate(v) for k, v in groups.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', required=True, choices=[f'head-{w}-seed{s}' for w in ('S','L') for s in (42,43,44)])
    p.add_argument('--phase', choices=['gate','full'], required=True)
    p.add_argument('--time-limit-seconds', type=float, default=5400)
    args = p.parse_args(); root = args.root.resolve(); cfg, splits = setup(root)
    if not 0 < args.time_limit_seconds <= cfg['stop']['single_val_seconds']:
        p.error('Runtime limit may only tighten the registered cap')
    label = ('gate-density-' if args.phase == 'gate' else 'val-')+args.run
    out = root/'results/round2-v1'/label
    if out.exists():
        raise FileExistsError('Existing validation evidence must not be overwritten')
    out.mkdir(); started = time.monotonic()
    try:
        train = root/'results/round2-v1'/args.run
        status = json.loads((train/'status.json').read_text())
        manifest = json.loads((train/'checkpoint_manifest.json').read_text())
        checkpoint = root/'work/round2-v1'/args.run/'best.pth'
        assert status['status'] == 'complete' and manifest['complete_epochs'] == 30
        assert sha(checkpoint) == manifest['best_sha256']
        baseline_rows = read_rows(root/cfg['baseline_result']/'predictions.jsonl')
        baseline = {r['image']:r for r in baseline_rows}
        names = [r['image'] for r in baseline_rows]
        assert set(names) == set(splits['val_A']+splits['val_B']) and len(names) == 1286
        if args.phase == 'gate':
            names = [r['image'] for r in json.loads((root/'results/round2-v1/gate/gate.json').read_text())['rows']]
        else:
            prerequisite = root/'results/round2-v1'/('gate-density-'+args.run)
            assert json.loads((prerequisite/'summary.json').read_text())['status'] == 'complete'
            assert json.loads((prerequisite/'protocol.json').read_text())['checkpoint_sha256'] == sha(checkpoint)
        record = analysis_record(root, label)
        record.update(run=args.run, phase=args.phase, split='val', ids=names,
                      checkpoint_sha256=sha(checkpoint), training_manifest_sha256=sha(train/'checkpoint_manifest.json'),
                      baseline_predictions_sha256=sha(root/cfg['baseline_result']/'predictions.jsonl'),
                      time_limit_seconds=args.time_limit_seconds)
        write_json(out/'protocol.json', record)
        verify_inputs(root, cfg)
        inputs = Inputs(root, names)
        width = args.run.split('-')[1]
        model = load_model(root); adapter = Round2Adapter(model); frozen = state_sha(model)
        assert frozen == manifest['frozen_model_sha256']
        head = DensityHead(cfg['m4']['widths'][width]).cuda().eval()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        head.load_state_dict(saved['head']); del saved
        maps = root/'work/round2-v1'/label/'maps'; maps.mkdir(parents=True)
        rows = []; torch.cuda.reset_peak_memory_stats()
        with (out/'predictions.jsonl').open('x') as stream, torch.no_grad():
            for name in names:
                if time.monotonic()-started >= args.time_limit_seconds:
                    raise TimeoutError('Registered validation runtime limit reached')
                sample = inputs.prepare(name, with_target=False)
                begin = time.perf_counter()
                x, prompts = sample['image'].cuda(), sample['boxes'].cuda()
                output, features, appearance = adapter.forward(x, prompts)
                final, scores = postprocess(output, int(sample['width']*sample['scale']), int(sample['height']*sample['scale']))
                boxes = final.cpu().numpy()*(1024/sample['scale']); scores = scores.cpu().numpy()
                if not np.array_equal(boxes, np.asarray(baseline[name]['boxes_xyxy'], dtype=np.float32).reshape(-1,4)):
                    raise AssertionError('Frozen B0 detections changed')
                adapted, centerness, coords = adapter.dense(features, appearance, prompts[:,:,2:]-prompts[:,:,:2])
                q = head(torch.cat([adapted, centerness], dim=1), sample['mask'].unsqueeze(0).cuda())
                torch.cuda.synchronize(); seconds = time.perf_counter()-begin
                density = restore_density(q, sample['height'], sample['width'], sample['valid_height'], sample['valid_width'])
                check_memory()
                del x, prompts, output, features, appearance, final, adapted, centerness, coords, q
                # Ground truth enters only after inference is finished.
                annotation = inputs.annotations[name]
                points = points_for_game(annotation['points'], sample['height'], sample['width'])
                exemplars = np.asarray(annotation['box_examples_coordinates'], dtype=np.float64)[:3,[0,2],:].reshape(-1,4)
                assert len(points) == baseline[name]['truth']
                m1 = boxes_to_density(boxes, sample['height'], sample['width'])
                truth_cells = point_cells(points, sample['height'], sample['width'])[1]
                density_cells = cell_mass(density); m1_cells = cell_mass(m1)
                spatial = {f'game{level}': game(density, points, level) for level in (1,2)}
                for level in (1,2):
                    assert abs(spatial[f'game{level}']-game_from_cells(density_cells, truth_cells, level)) < 1e-6
                map_file = maps/(Path(name).stem+'.npz')
                np.savez_compressed(map_file, density=density)
                row = {'image':name, 'truth':len(points), 'prediction':float(density.sum()), **spatial,
                       'width':sample['width'], 'height':sample['height'],
                       'detector_prediction':len(boxes), 'retained_raw_objectness':scores.tolist(),
                       'density_mass':float(density.sum()), 'density_map_sha256':sha(map_file),
                       'density_cell_mass':density_cells.tolist(), 'm1_cell_mass':m1_cells.tolist(),
                       'truth_cell_count':truth_cells.tolist(),
                       'features':grid_features(boxes, scores, exemplars, density).tolist(),
                       'detector_localization':localization(boxes, points, exemplars),
                       'inference_seconds_per_image':seconds}
                rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
                if len(rows)%50 == 0:
                    progress = {'done':len(rows), 'total':len(names), 'elapsed_seconds':time.monotonic()-started}
                    write_json(out/'progress.json', progress); print(json.dumps(progress), flush=True)
        assert state_sha(model) == frozen
        if time.monotonic()-started >= args.time_limit_seconds:
            raise TimeoutError('Registered validation runtime limit reached')
        write_json(out/'summary.json', {'status':'complete', 'run':args.run, 'phase':args.phase,
                   **summarize(rows), 'frozen_b0_boxes_bitwise_equal':len(rows),
                   'elapsed_seconds':time.monotonic()-started, 'peak_allocated_mib':check_memory(),
                   'finished_utc':utc(), 'maps_published':False})
    except BaseException as exc:
        write_json(out/'error.json', {'status':'failed', 'error_type':type(exc).__name__,
                   'elapsed_seconds':time.monotonic()-started, 'automatic_retry':False, 'updated_utc':utc()})
        raise


if __name__ == '__main__':
    main()
