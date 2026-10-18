"""Audit fold ownership, fitting scope, saved maps and reported M4 crossfit metrics.

Selection and map/metric reconstruction are independent of the crossfit writer.
Coefficient reproducibility reuses the registered optimizer on each declared
fitting fold; this is recorded separately from the independent arithmetic audit.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup,sha,write_json
from planb.round2.fusion import fit_gate
from planb.density import boxes_to_density,points_for_game,game
from audit_b0_results import read_rows
from audit_round2_m4 import metrics
from analyze_m2_m3 import paired_intervals


def equal(actual,expected):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:equal(actual[key],expected[key])
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):equal(a,b)
    elif isinstance(expected,float):
        assert math.isfinite(actual) and math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-8)
    else:assert actual==expected


def check_partitions(folds,splits,labels):
    assert set(folds)=={'A','B'}
    for fit,assess in [('A','B'),('B','A')]:
        f=folds[fit]
        assert f['fit_images']==splits['val_'+fit]
        assert f['assess_images']==splits['val_'+assess]
        assert not set(f['fit_images'])&set(f['assess_images'])
        assert not {labels[n] for n in f['fit_images']}&{labels[n] for n in f['assess_images']}


def check_scaler(features,model):
    x=np.asarray(features,dtype=float).reshape(-1,6)
    expected_std=x.std(0)
    for column in range(6):
        if len(set(x[:,column].tolist()))==1 or expected_std[column]==0:
            expected_std[column]=1.
    np.testing.assert_allclose(model['mean'],x.mean(0),atol=1e-10,rtol=1e-12)
    np.testing.assert_allclose(model['std'],expected_std,atol=1e-10,rtol=1e-12)
    assert model['fitting_cells']==len(x) and model['fitting_images']==len(x)//64


def choose_width(names,source,baseline):
    candidates=[]
    for rank,width in enumerate(('S','L')):
        errors=[source[width][n]['prediction']-baseline[n]['truth'] for n in names]
        candidates.append((math.fsum(map(abs,errors))/len(names),
                           math.sqrt(math.fsum(e*e for e in errors)/len(names)),rank,width))
    base=math.sqrt(math.fsum((baseline[n]['prediction']-baseline[n]['truth'])**2 for n in names)/len(names))
    eligible=[r for r in candidates if r[1]<=base]
    return min(eligible)[3] if eligible else min(candidates,key=lambda r:(r[1],r[0],r[2]))[3]


def audit(root):
    cfg,splits=setup(root);out=root/'results/round2-v1/m4-crossfit-seed42'
    protocol=json.loads((out/'protocol.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert summary['status']=='complete' and summary['n']==1286 and summary['class_count']==29
    assert summary['development_evidence_only'] is True
    for receipt in (protocol,protocol['analysis_code']):
        for name,value in receipt['files_sha256'].items():assert sha(root/name)==value
    assert protocol['config_sha256']==sha(root/'configs/round2_v1.json')
    baseline_path=root/cfg['baseline_result']/'predictions.jsonl'
    assert sha(baseline_path)==protocol['baseline_predictions_sha256']
    base_rows=read_rows(baseline_path);baseline={r['image']:r for r in base_rows};names=list(baseline)
    class_file=root/'data/fsc147/annotations/ImageClasses_FSC147.txt'
    assert sha(class_file)==splits['source_sha256']['ImageClasses_FSC147.txt']
    labels=dict(line.strip().split(maxsplit=1) for line in class_file.read_text().splitlines() if line.strip())
    annotations=json.loads((root/'data/fsc147/annotations/annotation_FSC147_384.json').read_text())
    source={}
    for width in ('S','L'):
        d=root/'results/round2-v1'/('val-head-'+width+'-seed42')
        for file,want in protocol['inputs'][width].items():assert sha(d/file)==want
        evidence=json.loads((d/'audit.json').read_text());assert evidence['status']=='passed' and evidence['n']==1286
        source[width]={r['image']:r for r in read_rows(d/'predictions.jsonl')}
        assert list(source[width])==names
    folds=json.loads((out/'folds.json').read_text());check_partitions(folds,splits,labels)
    final=json.loads((out/'full-val-selection.json').read_text());assert final['fit_images']==names
    all_fits={**folds,'full-val':final}
    for name,record in all_fits.items():
        fit=record['fit_images'];width=choose_width(fit,source,baseline)
        assert width==record['selection']['selected']
        base_rmse=math.sqrt(math.fsum((baseline[n]['prediction']-baseline[n]['truth'])**2 for n in fit)/len(fit))
        equal(record['selection']['baseline_rmse'],base_rmse)
        eligible=[]
        for candidate in ('S','L'):
            errors=[source[candidate][n]['prediction']-baseline[n]['truth'] for n in fit]
            candidate_metrics={'mae':math.fsum(map(abs,errors))/len(fit),
                               'rmse':math.sqrt(math.fsum(e*e for e in errors)/len(fit))}
            equal(record['selection']['metrics'][candidate],candidate_metrics)
            if candidate_metrics['rmse']<=base_rmse:eligible.append(candidate)
        assert record['selection']['rmse_eligible']==eligible
        if name=='full-val':assert width==record['selected_width']==summary['full_val_selected_width']
        rows=[source[width][n] for n in fit]
        check_scaler([r['features'] for r in rows],record['gate'])
        refit=fit_gate([r['features'] for r in rows],[r['m1_cell_mass'] for r in rows],
                       [r['density_cell_mass'] for r in rows],[r['truth_cell_count'] for r in rows],cfg['m4']['fusion'])
        equal(record['gate'],refit)
    predictions={kind:read_rows(out/(kind+'-predictions.jsonl')) for kind in ('density','fusion')}
    for rows in predictions.values():assert [r['image'] for r in rows]==names and len(rows)==1286
    maximum_delta=0.
    for density_row,fusion_row in zip(predictions['density'],predictions['fusion']):
        n=density_row['image'];b=baseline[n];fit='B' if n in splits['val_A'] else 'A'
        assessed='A' if fit=='B' else 'B';record=folds[fit];width=record['selection']['selected'];r=source[width][n]
        for row in (density_row,fusion_row):
            assert row['truth']==len(annotations[n]['points'])==b['truth']
            assert row['fit_fold']==fit and row['assess_fold']==assessed and row['selected_width']==width
            assert n not in record['fit_images']
        for key in ('prediction','game1','game2'):equal(density_row[key],r[key])
        model=record['gate'];beta=np.asarray(model['coefficients'])
        z=(np.asarray(r['features'])-np.asarray(model['mean']))/np.asarray(model['std'])
        logits=beta[0]+z@beta[1:]
        weights=np.exp(-np.logaddexp(0.,-logits))
        np.testing.assert_allclose(weights,fusion_row['weights'],atol=1e-10,rtol=1e-12)
        source_map=root/'work/round2-v1'/('val-head-'+width+'-seed42')/'maps'/(Path(n).stem+'.npz')
        assert sha(source_map)==r['density_map_sha256']
        with np.load(source_map) as arrays:m4=arrays['density']
        m1=boxes_to_density(b['boxes_xyxy'],b['height'],b['width'])
        expected=np.empty_like(m1);cells=np.empty((8,8));ys=np.linspace(0,b['height'],9,dtype=int);xs=np.linspace(0,b['width'],9,dtype=int)
        for j in range(8):
            for i in range(8):
                sl=np.s_[ys[j]:ys[j+1],xs[i]:xs[i+1]]
                expected[sl]=weights[j,i]*m1[sl]+(1-weights[j,i])*m4[sl]
                cells[j,i]=expected[sl].sum()
        path=root/'work/round2-v1/m4-crossfit-seed42/maps'/(Path(n).stem+'.npz')
        assert sha(path)==fusion_row['density_map_sha256']
        with np.load(path) as arrays:saved=arrays['density']
        np.testing.assert_allclose(saved,expected,atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(fusion_row['cell_mass'],cells,atol=1e-9,rtol=1e-12)
        delta=abs(float(saved.sum())-fusion_row['prediction']);assert delta<1e-6;maximum_delta=max(maximum_delta,delta)
        points=points_for_game(annotations[n]['points'],b['height'],b['width'])
        for level in (1,2):assert abs(game(saved,points,level)-fusion_row[f'game{level}'])<1e-6
    base_metrics=metrics(base_rows);base_dense=metrics([r for r in base_rows if r['truth']>=300])
    for kind,rows in predictions.items():
        report=summary['reports'][kind]
        groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
                '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
        for group,subset in groups.items():
            stored=report['overall'] if group=='overall' else report['groups'][group]
            for key,value in metrics(subset).items():equal(stored[key],value)
            for unit in ('image','class'):
                equal(report['paired_vs_b0'][group][unit],paired_intervals(subset,baseline,labels,2000,unit))
        for category in set(labels[n] for n in names):
            for key,value in metrics([r for r in rows if labels[r['image']]==category]).items():
                equal(report['per_class'][category][key],value)
        m=metrics(rows);dense=metrics(groups['>=300'])
        admitted=m['mae']<=base_metrics['mae']*1.02 and m['rmse']<base_metrics['rmse'] and dense['mae']<base_dense['mae']
        assert report['candidate_gate']==admitted and report['game2_below_m1']==(m['game2']<base_metrics['game2'])
        equal(report['mean_signed_count_error'],math.fsum(r['prediction']-r['truth'] for r in rows)/len(rows))
    loc=[source['S'][n]['detector_localization'] for n in names]
    assert all(source['L'][n]['detector_localization']==source['S'][n]['detector_localization'] for n in names)
    tp,fp,fn=[sum(r[k] for r in loc) for k in ('tp','fp','fn')]
    for key,value in {'tp':tp,'fp':fp,'fn':fn,'micro_f1':2*tp/(2*tp+fp+fn),'macro_f1':math.fsum(r['f1'] for r in loc)/len(loc)}.items():
        equal(summary['detector_localization'][key],value)
    result={'status':'passed','n':1286,'class_count':29,'fold_image_and_class_separation':True,
            'width_selection_independently_recomputed':True,'scalers_fit_fold_only':True,
            'coefficients_reproduced_with_registered_optimizer':True,
            'full_val_selection_not_used_for_oof':True,'saved_maps_and_metrics_recomputed':True,
            'paired_intervals_recomputed_with_fixed_routine':True,'maximum_density_mass_delta':maximum_delta,
            'audit_code_sha256':sha(Path(__file__)),
            'files_sha256':{file:sha(out/file) for file in ['protocol.json','folds.json','summary.json','full-val-selection.json','density-predictions.jsonl','fusion-predictions.jsonl']}}
    if (out/'audit.json').exists():equal(json.loads((out/'audit.json').read_text()),result)
    else:write_json(out/'audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:print(json.dumps(audit(args.root.resolve())),flush=True)
    except BaseException as exc:
        out=args.root/'results/round2-v1/m4-crossfit-seed42'
        if out.exists():write_json(out/'audit_error.json',{'status':'failed','error_type':type(exc).__name__})
        raise


if __name__=='__main__':main()
