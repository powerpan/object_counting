"""Gate, smoke and val-only TTA. Truth is read for scoring after inference."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from round3_tta_common import sources, assets, runtime, read, write_json, sha, utc
from run_m2_m3 import load_model, smoke_ids
from planb.round3_tta import TTAEngine, VIEWS
from planb.round3_m3 import require_equal
from planb.round2.common import state_sha
from planb.density import boxes_to_density, points_for_game, game
from planb.round2.spatial import cell_mass, point_cells, grid_features, localization


def main(root, phase, variant):
    cfg, splits, receipt = sources(root)
    if variant not in VIEWS or (phase == 'gate' and variant != 'tta-repeat'):
        raise ValueError('Unregistered TTA phase/variant')
    label = 'gate' if phase == 'gate' else phase+'-'+variant
    out = root/'results/round3-tta-v1'/label
    if out.exists():
        raise FileExistsError('TTA evidence already exists')
    out.mkdir(parents=True); started = time.monotonic()
    try:
        runtime(root); names, baseline, verified = assets(root, cfg, splits)
        if phase != 'full':
            names = smoke_ids(names, baseline)
        write_json(out/'protocol.json', {**receipt, **verified, 'phase':phase, 'variant':variant,
                   'ids':names, 'split':'val', 'started_utc':utc(), 'precision':'FP32', 'tf32':False,
                   'config_sha256':sha(root/'configs/round3_v1.json'), 'ground_truth_used_by_policy':False,
                   'density_projection':'inverse_box_geometry_then_original_grid_M1',
                   'localization_and_box_features':'original_view_only'})
        model = load_model(root); frozen = state_sha(model); engine = TTAEngine(model, cfg)
        annotations = read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
        maps = root/'work/round3-tta-v1'/label
        maps.mkdir(parents=True, exist_ok=False)
        torch.cuda.reset_peak_memory_stats(); n = 0
        with torch.inference_mode(), (out/'predictions.jsonl').open('x') as stream:
            for name in names:
                with Image.open(root/'data/fsc147/images_384_VarV2'/name) as im:
                    image = T.ToTensor()(im.convert('RGB'))
                h, w = image.shape[-2:]; anno = annotations[name]
                prompts = torch.tensor(anno['box_examples_coordinates'], dtype=torch.float32)[:3,[0,2],:].reshape(-1,4)
                if prompts.shape != (3,4) or not ((prompts[:,2:]-prompts[:,:2])>0).all():
                    raise ValueError('Invalid exemplar boxes')
                if phase == 'gate':
                    x, p, scale = engine.prepare(image, prompts)
                    native = model(x, p)
                    adapted, f, app = engine.adapter.forward(x, p)
                    require_equal(native[0][0], adapted)
                    del native, adapted, f, app, x, p
                torch.cuda.synchronize(); tick = time.monotonic()
                density, prediction, boxes, scores, detail = engine.run(image, prompts, variant)
                torch.cuda.synchronize(); seconds = time.monotonic()-tick
                expected = np.asarray(baseline[name]['boxes_xyxy'], dtype=np.float32).reshape(-1,4)
                if not np.array_equal(boxes, expected):
                    raise ValueError('Original TTA view differs from saved B0')
                if variant == 'tta-repeat' and (prediction != baseline[name]['prediction'] or
                        not np.array_equal(density, boxes_to_density(expected, h, w))):
                    raise ValueError('Repeated count or map differs from B0')
                points = points_for_game(anno['points'], h, w)
                mass = cell_mass(density); _, truth8 = point_cells(points, h, w)
                path = maps/(Path(name).stem+'.npz'); np.savez_compressed(path, density=density)
                row = {'image':name, 'width':w, 'height':h, 'truth':len(points), 'prediction':prediction,
                       'density_mass':float(density.sum()), 'game1':game(density,points,1), 'game2':game(density,points,2),
                       'boxes_xyxy':boxes.tolist(), 'objectness':scores.tolist(), 'exemplars_xyxy':prompts.tolist(),
                       'mass8':mass.tolist(), 'truth8':truth8.tolist(),
                       'grid_features':grid_features(boxes,scores,prompts.numpy(),density).tolist(),
                       'localization':localization(boxes,points,prompts.numpy()),
                       'inference_seconds_per_image':seconds, 'density_array_sha256':hashlib.sha256(density.tobytes()).hexdigest(),
                       'density_file_sha256':sha(path), 'detail':detail}
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();n+=1
                write_json(out/'progress.json', {'done':n,'total':len(names),'elapsed_seconds':time.monotonic()-started,
                           'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'updated_utc':utc()})
                if n%25 == 0 or n == len(names):
                    print(json.dumps({'done':n,'total':len(names),'variant':variant}),flush=True)
        if state_sha(model) != frozen:
            raise ValueError('Frozen model parameters changed')
        write_json(out/'summary.json', {'status':'complete','n':n,'phase':phase,'variant':variant,
                   'frozen_sha256':frozen,'finished_utc':utc(),'elapsed_seconds':time.monotonic()-started,
                   'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                   'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                   'native_raw_gate_passed':phase=='gate'})
    except Exception as exc:
        write_json(out/'error.json', {'status':'failed','error_type':type(exc).__name__,
                   'elapsed_seconds':time.monotonic()-started,'automatic_retry':False,'updated_utc':utc()})
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['gate','smoke','full'],required=True)
    p.add_argument('--variant',choices=list(VIEWS),required=True);a=p.parse_args()
    main(a.root.resolve(),a.phase,a.variant)
