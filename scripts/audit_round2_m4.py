"""Recompute M4 val metrics from saved maps, detections, and official annotations."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup, sha, write_json
from planb.round2.spatial import cell_mass, point_cells, grid_features, localization
from planb.density import boxes_to_density, points_for_game, game
from audit_b0_results import read_rows


def metrics(rows):
    n = len(rows)
    errors = [r['prediction']-r['truth'] for r in rows]
    return {'n':n, 'mae':math.fsum(map(abs,errors))/n,
            'rmse':math.sqrt(math.fsum(e*e for e in errors)/n),
            'game1':math.fsum(r['game1'] for r in rows)/n,
            'game2':math.fsum(r['game2'] for r in rows)/n}


def audit(root, label):
    cfg, splits = setup(root)
    out = root/'results/round2-v1'/label
    protocol = json.loads((out/'protocol.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    rows = read_rows(out/'predictions.jsonl')
    assert summary['status'] == 'complete'
    assert [r['image'] for r in rows] == protocol['ids']
    assert len(rows) == len(set(protocol['ids']))
    assert set(protocol['ids']) <= set(splits['val_A']+splits['val_B'])
    if protocol['phase'] == 'full':
        assert len(rows) == 1286
    assert protocol['config_sha256'] == sha(root/'configs/round2_v1.json')
    for record in (protocol, protocol['analysis_code']):
        for file, value in record['files_sha256'].items():
            assert sha(root/file) == value
    baseline_file = root/cfg['baseline_result']/'predictions.jsonl'
    assert sha(baseline_file) == protocol['baseline_predictions_sha256']
    baseline = {r['image']:r for r in read_rows(baseline_file)}
    annotations = json.loads((root/'data/fsc147/annotations/annotation_FSC147_384.json').read_text())
    maximum_mass_difference = 0.
    for row in rows:
        name = row['image']; b = baseline[name]; annotation = annotations[name]
        assert row['truth'] == b['truth'] == len(annotation['points'])
        assert row['width'] == b['width'] and row['height'] == b['height']
        assert row['detector_prediction'] == len(b['boxes_xyxy'])
        path = root/'work/round2-v1'/label/'maps'/(Path(name).stem+'.npz')
        assert sha(path) == row['density_map_sha256']
        with np.load(path) as arrays:
            density = arrays['density']
        assert density.shape == (b['height'],b['width'])
        assert np.isfinite(density).all() and (density>=0).all()
        mass_delta = abs(float(density.sum())-row['prediction'])
        assert mass_delta < 1e-6 and row['prediction'] == row['density_mass']
        maximum_mass_difference = max(maximum_mass_difference,mass_delta)
        points = points_for_game(annotation['points'],b['height'],b['width'])
        exemplars = np.asarray(annotation['box_examples_coordinates'],dtype=np.float64)[:3,[0,2],:].reshape(-1,4)
        m1 = boxes_to_density(b['boxes_xyxy'],b['height'],b['width'])
        for recorded, expected in [(row['density_cell_mass'],cell_mass(density)),
                                   (row['m1_cell_mass'],cell_mass(m1)),
                                   (row['truth_cell_count'],point_cells(points,b['height'],b['width'])[1]),
                                   (row['features'],grid_features(b['boxes_xyxy'],row['retained_raw_objectness'],exemplars,density))]:
            np.testing.assert_allclose(recorded,expected,atol=1e-9,rtol=1e-12)
        for level in (1,2):
            assert abs(row[f'game{level}']-game(density,points,level)) < 1e-6
        assert row['detector_localization'] == localization(b['boxes_xyxy'],points,exemplars)
        assert row['game2']+1e-6 >= row['game1'] >= abs(row['prediction']-row['truth'])-1e-6
    groups = {'overall':rows,'<50':[r for r in rows if r['truth']<50],
              '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    for name, sample in groups.items():
        if not sample:continue
        recorded = summary['overall'] if name=='overall' else summary['groups'][name]
        for key,value in metrics(sample).items():
            assert math.isclose(value,recorded[key],rel_tol=1e-10,abs_tol=1e-8)
    report = {'status':'passed','n':len(rows),'phase':protocol['phase'],
              'saved_maps_and_hashes_checked':True,'mass_and_spatial_metrics_recomputed':True,
              'maximum_mass_difference':maximum_mass_difference,'official_point_counts_checked':True,
              'frozen_detector_localization_recomputed':True,'config_and_code_hashes_checked':True,
              'summary':metrics(rows),'audit_code_sha256':sha(Path(__file__)),
              'files_sha256':{file:sha(out/file) for file in ('protocol.json','summary.json','predictions.jsonl')}}
    if (out/'audit.json').exists():
        assert json.loads((out/'audit.json').read_text())==report
    else:
        write_json(out/'audit.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--label',required=True);a=p.parse_args()
    if '/' in a.label or '..' in a.label: p.error('Expected result-directory name')
    print(json.dumps(audit(a.root.resolve(),a.label)),flush=True)


if __name__=='__main__':main()
