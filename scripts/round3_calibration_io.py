"""Local, CPU-only source/data boundaries for R3-3; no remote or model imports."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.density import boxes_to_density, points_for_game, game
from planb.round3_calibration import features, masses, point_grid, admitted
from audit_b0_results import read_rows, aggregate
from analyze_m2_m3 import paired_intervals

REGISTRATION='ab4f53a840a6a8a6d28992c9368b82b98e8c1102'
NEW_FILES=('planb/round3_calibration.py','scripts/round3_calibration_io.py',
           'scripts/run_round3_calibration.py','scripts/audit_round3_calibration.py',
           'tests/test_round3_calibration.py','docs/ROUND3_CALIBRATION_EXECUTION.md')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def rows_write(p,rows):
    with Path(p).open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')


def load(root, asset_root, receipt_path):
    receipt=read(receipt_path)
    if receipt['registration_commit']!=REGISTRATION or not receipt['remote_main_verified'] or len(receipt['code_commit'])!=40:
        raise ValueError('Calibration has not been committed and pushed')
    if any(n not in receipt['files_sha256'] for n in NEW_FILES):
        raise ValueError('Incomplete calibration source receipt')
    for n,digest in receipt['files_sha256'].items():
        if sha(root/n)!=digest:raise ValueError('Calibration source changed: '+n)
    cfg=read(root/'configs/round3_v1.json');split=read(root/cfg['split_file'])
    if cfg['final_test']['enabled'] or sha(root/cfg['split_file'])!=cfg['split_sha256']:
        raise ValueError('Unexpected scope or partition')
    ta=root/'results/round3-tta-v1';manifest=read(ta/'complete_files_v1.json')
    if manifest['status']!='passed' or manifest['scope']!='TTA complete val and crossfit; no calibration or test':
        raise ValueError('TTA completion evidence missing')
    for n,digest in manifest['files_sha256'].items():
        if sha(ta/n)!=digest:raise ValueError('TTA completed evidence changed')
    if read(ta/'workflow_v1.json')['state']!='awaiting_publication' or read(ta/'budget_ledger_v1.json')['active'] is not None:
        raise ValueError('TTA workflow is incomplete')
    if read(ta/'crossfit/audit.json')['status']!='passed':raise ValueError('TTA independent audit missing')
    prior=read(ta/'gate/protocol.json')['files_sha256']
    if len(prior)!=93 or any(receipt['files_sha256'].get(n)!=h for n,h in prior.items()):
        raise ValueError('Calibration does not preserve prior 93 sources')
    for n,h in split['source_sha256'].items():
        if sha(asset_root/'annotations'/n)!=h:raise ValueError('Annotation asset changed')
    official=read(asset_root/'annotations/Train_Test_Val_FSC_147.json')
    annotations=read(asset_root/'annotations/annotation_FSC147_384.json')
    labels=dict(x.strip().split(maxsplit=1) for x in (asset_root/'annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    bdir=root/cfg['baseline_result'];audit=read(bdir/'audit.json')
    if audit['status']!='passed':raise ValueError('Baseline audit missing')
    for n,h in audit['files_sha256'].items():
        if sha(bdir/n)!=h:raise ValueError('Baseline evidence changed')
    baseline=read_rows(bdir/'predictions.jsonl');names=[r['image'] for r in baseline]
    if names!=official['val'] or len(names)!=1286 or len(set(names))!=1286:
        raise ValueError('Unexpected validation set')
    if set(names)&set(official['train']+official['test']):raise ValueError('Data split overlap')
    if set(split['val_A'])&set(split['val_B']) or set(split['val_A']+split['val_B'])!=set(names):
        raise ValueError('Outer partition mismatch')
    for f in ('A','B'):
        if {labels[n] for n in split['val_'+f]}!=set(split['val_'+f+'_classes']):raise ValueError('Class partition mismatch')
    return cfg,split,receipt,baseline,annotations,labels,{
        'tta_completion_sha256':sha(ta/'complete_files_v1.json'),
        'baseline_audit_sha256':sha(bdir/'audit.json'),
        'baseline_predictions_sha256':sha(bdir/'predictions.jsonl'),
        'source_receipt_sha256':sha(receipt_path),'annotation_sha256':dict(split['source_sha256'])}


def prepare(baseline, annotations):
    result={}
    for row in baseline:
        n=row['image'];h,w=row['height'],row['width'];a=annotations[n]
        points=points_for_game(a['points'],h,w)
        if row['truth']!=len(points) or row['prediction']!=len(row['boxes_xyxy']):raise ValueError('Baseline counts differ')
        exemplar=np.asarray(a['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4)
        density=boxes_to_density(row['boxes_xyxy'],h,w);mass=masses(density)
        _,truth=point_grid(points,h,w)
        for level in (1,2):
            if not np.isclose(game(density,points,level),row['game'+str(level)],rtol=1e-12,atol=1e-8):
                raise ValueError('Baseline spatial metric differs')
        if abs(float(mass.sum())-row['prediction'])>1e-6:raise ValueError('Baseline mass differs')
        x=features(mass,row['boxes_xyxy'],exemplar,h,w,row['scale_factor'])
        result[n]={'image':n,'truth':row['truth'],'prediction':row['prediction'],
                   'width':w,'height':h,'base_scale':row['scale_factor'],
                   'mass8':mass.tolist(),'truth8':truth.tolist(),'features':x.tolist(),
                   'game1':row['game1'],'game2':row['game2']}
    return result


def report(rows,baseline,labels,reps):
    groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
            '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    return {'metrics':{k:aggregate(v) for k,v in groups.items()},
            'per_class':{k:aggregate([r for r in rows if labels[r['image']]==k]) for k in sorted({labels[r['image']] for r in rows})},
            'candidate_gate':admitted(rows,baseline),
            'paired_vs_b0':{k:{unit:paired_intervals(v,baseline,labels,reps,unit) for unit in ('image','class')}
                            for k,v in groups.items() if v}}
