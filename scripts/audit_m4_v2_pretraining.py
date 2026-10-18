"""Separate arithmetic and artifact audit for M4-v2 prerequisites/training."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import sha,write_json
from planb.round2.fusion import fit_gate
from planb.density import boxes_to_density,points_for_game,game
from m4_v2_common import settings,receipt,read,deadline
from audit_b0_results import read_rows
from audit_round2_crossfit import equal,check_scaler
from analyze_m2_m3 import paired_intervals


def errors(rows,key='prediction'):
    e=[r[key]-r['truth'] for r in rows]
    return {'n':len(e),'mae':math.fsum(abs(x) for x in e)/len(e),
            'rmse':math.sqrt(math.fsum(x*x for x in e)/len(e))}


def audit(root,stage,limit):
    start=time.monotonic();base,old,c,splits=settings(root);receipt(root,'audit-'+stage)
    out=root/'results/m4-v2-v1'/stage;protocol=read(out/'protocol.json');result=read(out/'summary.json')
    for n,h in protocol['files_sha256'].items():assert sha(root/n)==h
    assert protocol['config_sha256']==sha(root/'configs/m4_v2.json')
    annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    evidence={'status':'passed','stage':stage,'source_and_config_hashes_verified':True}
    tracked=['protocol.json','summary.json']
    if stage=='initial-gate':
        rows=result['rows'];names=[r['image'] for r in read(root/'results/round2-v1/gate/gate.json')['rows']]
        baseline={r['image']:r for r in read_rows(root/base['baseline_result']/'predictions.jsonl')}
        assert result['status']=='passed' and [r['image'] for r in rows]==names and len(rows)==32
        for r in rows:
            assert r['B0_count']==baseline[r['image']]['prediction']
            assert r['bitwise_equal'] is True
            assert abs(r['prediction']-r['B0_count'])<=1e-4
        evidence.update(n=32,reported_gate_counts_recomputed=True)
    elif stage=='s-dense-check':
        rows=read_rows(out/'predictions.jsonl');tracked.append('predictions.jsonl')
        assert [r['image'] for r in rows]==splits['s_dense_check'] and len(rows)==43
        assert protocol['checkpoint_sha256']==read(root/'results/round2-v1/head-S-seed42/checkpoint_manifest.json')['best_sha256']
        for r in rows:
            deadline(start,limit);assert r['truth']==len(annotations[r['image']]['points'])>=300
            assert r['detector_prediction']==len(r['boxes_xyxy'])
            assert r['original_partition']==('fit' if r['image'] in old['train_fit'] else 'holdout')
            path=root/'work/m4-v2-v1/s-dense-check/maps'/(Path(r['image']).stem+'.npz');assert sha(path)==r['density_map_sha256']
            with np.load(path) as z:d=z['density']
            assert d.shape==(r['height'],r['width']) and np.isfinite(d).all() and (d>=0).all()
            assert abs(float(d.sum())-r['prediction'])<1e-6
        chosen=[r['prediction']/r['truth'] for r in rows if r['image'] in old['train_fit'] and r['truth']>=500]
        assert len(chosen)==15;ratio=float(np.median(chosen));equal(result['decision_median_ratio'],ratio)
        label='fit_problem_signal' if ratio<.7 else ('generalization_problem_signal' if ratio>=.9 else 'mixed')
        assert label==result['descriptive_label'] and result['v2_settings_may_change'] is False
        for part in ('fit','holdout'):
            for lo,hi in [(300,500),(500,1000),(1000,1000000)]:
                z=[r for r in rows if r['original_partition']==part and lo<=r['truth']<hi];record=result['groups'][part][str(lo)+'-'+str(hi)]
                assert record['n']==len(z)
                if z:
                    for k,v in errors(z).items():equal(record[k],v)
                    equal(record['median_S_over_truth'],float(np.median([r['prediction']/r['truth'] for r in z])))
                    equal(record['median_B0_over_truth'],float(np.median([r['detector_prediction']/r['truth'] for r in z])))
        evidence.update(n=43,maps_and_fixed_decision_recomputed=True)
    elif stage=='baseline-holdout':
        rows=read_rows(out/'predictions.jsonl');tracked.append('predictions.jsonl')
        assert [r['image'] for r in rows]==splits['train_holdout'] and len(rows)==396
        for r in rows:
            assert r['truth']==len(annotations[r['image']]['points'])
            assert r['prediction']==len(r['boxes_xyxy']) and np.isfinite(r['boxes_xyxy']).all()
        for k,v in errors(rows).items():equal(result[k],v)
        assert result['source_predictions_sha256']==sha(out/'predictions.jsonl')
        evidence.update(n=396,official_counts_and_reference_metrics_recomputed=True)
    elif stage=='s-fusion-record':
        source=root/'results/round2-v1/val-head-S-seed42'
        for n,h in protocol['S_evidence_sha256'].items():assert sha(source/n)==h
        original={r['image']:r for r in read_rows(source/'predictions.jsonl')}
        base_rows=read_rows(root/base['baseline_result']/'predictions.jsonl');baseline={r['image']:r for r in base_rows}
        labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
        folds=read(out/'folds.json');rows=read_rows(out/'predictions.jsonl');tracked+=['folds.json','predictions.jsonl']
        assert [r['image'] for r in rows]==list(baseline) and len(rows)==1286
        for f,t in [('A','B'),('B','A')]:
            record=folds[f];fit=splits['val_'+f];assert record['fit_images']==fit and record['assess_images']==splits['val_'+t]
            assert set(labels[n] for n in fit).isdisjoint(labels[n] for n in splits['val_'+t])
            r=[original[n] for n in fit];check_scaler([x['features'] for x in r],record['gate'])
            refit=fit_gate([x['features'] for x in r],[x['m1_cell_mass'] for x in r],
                [x['density_cell_mass'] for x in r],[x['truth_cell_count'] for x in r],base['m4']['fusion'])
            equal(record['gate'],refit)
        for r in rows:
            deadline(start,limit);n=r['image'];b=baseline[n];s=original[n];fit='B' if n in splits['val_A'] else 'A'
            assert r['fit_fold']==fit and n not in folds[fit]['fit_images']
            g=folds[fit]['gate'];beta=np.asarray(g['coefficients']);z=(np.asarray(s['features'])-g['mean'])/g['std']
            weights=np.exp(-np.logaddexp(0.,-(beta[0]+z@beta[1:])));np.testing.assert_allclose(weights,r['weights'],atol=1e-10)
            path=root/'work/round2-v1/val-head-S-seed42/maps'/(Path(n).stem+'.npz');assert sha(path)==s['density_map_sha256']
            with np.load(path) as z:d=z['density']
            m1=boxes_to_density(b['boxes_xyxy'],b['height'],b['width']);combined=np.empty_like(m1)
            ys=np.linspace(0,b['height'],9,dtype=int);xs=np.linspace(0,b['width'],9,dtype=int);cells=np.empty((8,8))
            for j in range(8):
                for i in range(8):
                    sl=np.s_[ys[j]:ys[j+1],xs[i]:xs[i+1]];combined[sl]=weights[j,i]*m1[sl]+(1-weights[j,i])*d[sl];cells[j,i]=combined[sl].sum()
            assert r['truth']==len(annotations[n]['points'])==b['truth'];equal(float(combined.sum()),r['prediction'])
            np.testing.assert_allclose(cells,r['cell_mass'],atol=1e-9,rtol=1e-12)
            points=points_for_game(annotations[n]['points'],b['height'],b['width'])
            for l in (1,2):equal(game(combined,points,l),r['game'+str(l)])
        groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],'50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
        for group,z in groups.items():
            record=result['overall'] if group=='overall' else result['groups'][group]
            for k,v in errors(z).items():equal(record[k],v)
            for l in (1,2):equal(record['game'+str(l)],math.fsum(r['game'+str(l)] for r in z)/len(z))
        for unit in ('image','class'):equal(result['paired_vs_b0'][unit],paired_intervals(rows,baseline,labels,2000,unit))
        assert result['candidate_eligible'] is False and result['width_selection_performed'] is False
        evidence.update(n=1286,folds_maps_metrics_verified=True,coefficients_reproduced_with_registered_optimizer=True,candidate_eligible=False)
    elif stage.startswith('train-seed'):
        status=read(out/'status.json');history=read(out/'history.json');tracked+=['status.json','history.json']
        epochs=30 if result['status']=='complete' else 10
        assert result['status'] in ('complete','negative_safety_stop') and status['epoch']==epochs
        assert [r['epoch'] for r in history]==list(range(1,epochs+1))
        reference=read(root/'results/m4-v2-v1/baseline-holdout/summary.json')
        frozen=reference['frozen_state_sha256'];best=min(history,key=lambda r:(r['holdout_rmse'],r['epoch']))
        for r in history:
            deadline(start,limit);f=out/('holdout-epoch%02d.jsonl'%r['epoch']);assert sha(f)==r['predictions_sha256']
            z=read_rows(f);assert [x['image'] for x in z]==splits['train_holdout']
            assert all(x['truth']==len(annotations[x['image']]['points']) for x in z)
            for x in z:equal(x['raw_prediction']+x['negative_mass'],x['prediction'])
            m=errors(z);equal(m['mae'],r['holdout_mae']);equal(m['rmse'],r['holdout_rmse'])
            assert r['train_n']==3359 and r['holdout_n']==396 and r['frozen_state_sha256']==frozen
        equal(status['best_holdout_rmse'],best['holdout_rmse']);assert status['best_epoch']==best['epoch']
        tenth=history[9];trigger=tenth['holdout_rmse']>reference['rmse']
        assert trigger==(result['status']=='negative_safety_stop')
        if result['status']=='complete':
            manifest=read(out/'checkpoint_manifest.json');tracked.append('checkpoint_manifest.json')
            assert manifest['complete_epochs']==30 and manifest['best_epoch']==best['epoch']
        else:manifest=result
        for kind in ('best','last'):
            p=root/'work/m4-v2-v1'/stage/(kind+'.pth');assert sha(p)==manifest[kind+'_sha256']
            saved=torch.load(p,map_location='cpu',weights_only=True)
            assert saved['frozen_state_sha256']==frozen and saved['epoch']==(best['epoch'] if kind=='best' else epochs)
            del saved
        evidence.update(completed_epochs=epochs,holdout_metrics_selection_and_safety_recomputed=True,
                        negative_safety_stop=trigger,checkpoints_cpu_load_verified=True)
    else:raise ValueError('Unknown audit stage')
    deadline(start,limit);evidence.update(files_sha256={n:sha(out/n) for n in tracked},audit_code_sha256=sha(Path(__file__)),elapsed_seconds=time.monotonic()-start)
    if (out/'audit.json').exists():raise FileExistsError('Existing audit must not be overwritten')
    write_json(out/'audit.json',evidence);return evidence


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--time-limit-seconds',type=float,required=True)
    a=p.parse_args();assert 0<a.time_limit_seconds<=5400
    print(json.dumps(audit(a.root.resolve(),a.stage,a.time_limit_seconds)),flush=True)


if __name__=='__main__':main()
