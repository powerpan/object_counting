"""Preregistered seed-42, class-disjoint width choice and regional fusion."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup, sha, write_json, utc
from planb.round2.fusion import select_candidate, fit_gate, gate_weights
from planb.round2.spatial import fused_map, cell_mass, game_from_cells
from planb.density import boxes_to_density
from audit_b0_results import read_rows, aggregate
from analyze_m2_m3 import paired_intervals
from round2_m4_validate import analysis_record, summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();cfg,splits=setup(root)
    out=root/'results/round2-v1/m4-crossfit-seed42'
    if out.exists():raise FileExistsError('Existing model-selection evidence must not be overwritten')
    results={};inputs={}
    for width in cfg['m4']['candidate_order']:
        d=root/'results/round2-v1'/('val-head-'+width+'-seed42')
        assert json.loads((d/'audit.json').read_text())['status']=='passed'
        assert json.loads((d/'summary.json').read_text())['status']=='complete'
        assert json.loads((d/'protocol.json').read_text())['phase']=='full'
        results[width]={r['image']:r for r in read_rows(d/'predictions.jsonl')}
        inputs[width]={f:sha(d/f) for f in ('predictions.jsonl','audit.json','protocol.json','summary.json')}
    baseline_rows=read_rows(root/cfg['baseline_result']/'predictions.jsonl')
    baseline={r['image']:r for r in baseline_rows};names=[r['image'] for r in baseline_rows]
    class_file=root/'data/fsc147/annotations/ImageClasses_FSC147.txt'
    assert sha(class_file)==splits['source_sha256']['ImageClasses_FSC147.txt']
    labels=dict(line.strip().split(maxsplit=1) for line in class_file.read_text().splitlines() if line.strip())
    assert all(set(rows)==set(names) for rows in results.values())
    out.mkdir();maps=root/'work/round2-v1/m4-crossfit-seed42/maps';maps.mkdir(parents=True)
    record=analysis_record(root,'R2-2-seed42')
    record.update(inputs=inputs,baseline_predictions_sha256=sha(root/cfg['baseline_result']/'predictions.jsonl'))
    write_json(out/'protocol.json',record)
    def choose(fit_names):
        chosen,selection=select_candidate({w:[results[w][n]['prediction'] for n in fit_names] for w in results},
                                         [baseline[n]['truth'] for n in fit_names],
                                         [baseline[n]['prediction'] for n in fit_names],cfg['m4']['candidate_order'])
        rows=[results[chosen][n] for n in fit_names]
        model=fit_gate([r['features'] for r in rows],[r['m1_cell_mass'] for r in rows],
                       [r['density_cell_mass'] for r in rows],[r['truth_cell_count'] for r in rows],cfg['m4']['fusion'])
        return chosen,selection,model
    folds={};density_rows={};fusion_rows={}
    for fit_name,assess_name in [('A','B'),('B','A')]:
        fit_names=splits['val_'+fit_name];assess=splits['val_'+assess_name]
        assert not set(fit_names)&set(assess)
        assert not {labels[n] for n in fit_names}&{labels[n] for n in assess}
        width,selection,model=choose(fit_names)
        folds[fit_name]={'fit_images':fit_names,'assess_images':assess,'selection':selection,'gate':model}
        for name in assess:
            r=results[width][name];b=baseline[name];w=gate_weights(r['features'],model)
            assert r['truth']==b['truth']
            density_rows[name]={key:r[key] for key in ('image','truth','prediction','game1','game2')}
            density_rows[name].update(selected_width=width,fit_fold=fit_name,assess_fold=assess_name)
            source_map=root/'work/round2-v1'/('val-head-'+width+'-seed42')/'maps'/(Path(name).stem+'.npz')
            assert sha(source_map)==r['density_map_sha256']
            with np.load(source_map) as arrays:m4=arrays['density']
            m1=boxes_to_density(b['boxes_xyxy'],b['height'],b['width'])
            combined=fused_map(m1,m4,w)
            expected=w*np.asarray(r['m1_cell_mass'])+(1-w)*np.asarray(r['density_cell_mass'])
            np.testing.assert_allclose(cell_mass(combined),expected,atol=1e-9,rtol=1e-12)
            save=maps/(Path(name).stem+'.npz');np.savez_compressed(save,density=combined)
            fusion_rows[name]={'image':name,'truth':b['truth'],'prediction':float(combined.sum()),
                              'game1':game_from_cells(expected,r['truth_cell_count'],1),
                              'game2':game_from_cells(expected,r['truth_cell_count'],2),
                              'selected_width':width,'fit_fold':fit_name,'assess_fold':assess_name,
                              'weights':w.tolist(),'cell_mass':expected.tolist(),'density_map_sha256':sha(save)}
    write_json(out/'folds.json',folds)
    groups={'overall':names,'<50':[n for n in names if baseline[n]['truth']<50],
            '50-299':[n for n in names if 50<=baseline[n]['truth']<300],
            '>=300':[n for n in names if baseline[n]['truth']>=300]}
    reports={}
    for title,lookup in [('density',density_rows),('fusion',fusion_rows)]:
        ordered=[lookup[n] for n in names]
        with (out/(title+'-predictions.jsonl')).open('x') as f:
            for row in ordered:f.write(json.dumps(row)+'\n')
        summary=summarize(ordered);overall=summary['overall'];dense=summary['groups']['>=300']
        base=aggregate(baseline_rows);base_dense=aggregate([baseline[n] for n in groups['>=300']])
        intervals={key:{unit:paired_intervals([lookup[n] for n in ids],baseline,labels,2000,unit)
                        for unit in ('image','class')} for key,ids in groups.items()}
        per_class={label:aggregate([lookup[n] for n in names if labels[n]==label]) for label in sorted(set(labels[n] for n in names))}
        reports[title]={**summary,'paired_vs_b0':intervals,'per_class':per_class,
                        'mean_signed_count_error':float(np.mean([r['prediction']-r['truth'] for r in ordered])),
                        'candidate_gate':overall['mae']<=1.02*base['mae'] and overall['rmse']<base['rmse'] and dense['mae']<base_dense['mae'],
                        'game2_below_m1':overall['game2']<base['game2']}
    width,selection,model=choose(names)
    write_json(out/'full-val-selection.json',{'purpose':'configuration for later seed sensitivity and final freeze; not independent performance',
               'selection':selection,'gate':model,'fit_images':names,'selected_width':width})
    localizations=[results['S'][n]['detector_localization'] for n in names]
    assert all(results['L'][n]['detector_localization']==results['S'][n]['detector_localization'] for n in names)
    tp,fp,fn=[sum(r[k] for r in localizations) for k in ('tp','fp','fn')]
    write_json(out/'summary.json',{'status':'complete','n':len(names),'class_count':29,'reports':reports,
               'full_val_selected_width':width,'detector_localization':{'tp':tp,'fp':fp,'fn':fn,
                   'micro_f1':2*tp/max(2*tp+fp+fn,1),'macro_f1':float(np.mean([r['f1'] for r in localizations])),
                   'scope':'unchanged B0 centers; not density or fusion localization'},
               'development_evidence_only':True,'independent_crossfit_audit':'pending','finished_utc':utc()})
    print(json.dumps({'status':'complete','selected_width_for_seed_repeats':width,
                      'density':reports['density']['overall'],'fusion':reports['fusion']['overall']}),flush=True)


if __name__=='__main__':main()
