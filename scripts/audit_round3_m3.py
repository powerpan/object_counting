"""CPU reconstruction of every output and category-fold analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_common import sources,read,sha,write_json
from audit_b0_results import aggregate,read_rows
from analyze_m2_m3 import paired_intervals
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import cell_mass,point_cells,grid_features,localization
from planb.round3_selection import crossfit_choices,candidate_gate
from planb.round2.fusion import select_candidate


def close(a,b):
    if not np.allclose(a,b,rtol=1e-12,atol=1e-8):raise ValueError('Independent reconstruction mismatch')


def report(rows,base,labels,reps):
    groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
            '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    metrics={k:aggregate(v) for k,v in groups.items()}
    return {'metrics':metrics,'per_class':{k:aggregate([r for r in rows if labels[r['image']]==k])
                                          for k in sorted({labels[r['image']] for r in rows})},
            'candidate_gate':candidate_gate(rows,base),
            'paired_vs_b0':{k:{u:paired_intervals(v,base,labels,reps,u) for u in ('image','class')}
                            for k,v in groups.items() if v},
            'localization':{'micro_f1':2*sum(r['localization']['tp'] for r in rows)/max(1,sum(
                2*r['localization']['tp']+r['localization']['fp']+r['localization']['fn'] for r in rows)),
                'image_macro_f1':statistics.mean(r['localization']['f1'] for r in rows)},
            'inference_seconds_mean':statistics.mean(r['inference_seconds_per_image'] for r in rows)}


def audit(root,label):
    cfg,splits,receipt=sources(root);out=root/'results/round3-v1'/label
    if (out/'audit.json').exists():raise FileExistsError('Existing audit')
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');rows=read_rows(out/'predictions.jsonl')
    if summary['status']!='complete' or summary['n']!=len(rows):raise ValueError('Incomplete run')
    if protocol['files_sha256']!=receipt['files_sha256'] or protocol['config_sha256']!=sha(root/'configs/round3_v1.json'):
        raise ValueError('Recorded source differs')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in baseline}
    if protocol['baseline_predictions_sha256']!=sha(root/cfg['baseline_result']/'predictions.jsonl'):
        raise ValueError('Baseline changed')
    from run_m2_m3 import smoke_ids
    names=[r['image'] for r in baseline]
    expected=names if protocol['phase']=='full' else smoke_ids(names,base)
    if [r['image'] for r in rows]!=expected or expected!=protocol['ids']:raise ValueError('Output order differs')
    annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    for r in rows:
        name=r['image'];h,w=r['height'],r['width'];a=annotations[name]
        boxes=np.asarray(r['boxes_xyxy'],dtype=np.float32).reshape(-1,4)
        prompts=np.asarray(a['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4)
        close(prompts,r['exemplars_xyxy']);points=points_for_game(a['points'],h,w)
        if r['prediction']!=len(boxes) or r['truth']!=len(points) or r['truth']!=base[name]['truth']:
            raise ValueError('Count or annotation mismatch')
        if (w,h)!=(base[name]['width'],base[name]['height']):raise ValueError('Original dimensions changed')
        density=boxes_to_density(boxes,h,w);mass=cell_mass(density);_,truth8=point_cells(points,h,w)
        close(mass,r['mass8']);close(truth8,r['truth8']);close(density.sum(),r['density_mass'])
        close(density.sum(),r['prediction']);close(game(density,points,1),r['game1']);close(game(density,points,2),r['game2'])
        close(grid_features(boxes,r['objectness'],prompts,density),r['grid_features'])
        if localization(boxes,points,prompts)!=r['localization']:raise ValueError('Localization mismatch')
        path=root/'work/round3-v1'/label/(Path(name).stem+'.npz')
        if sha(path)!=r['density_file_sha256'] or hashlib.sha256(density.tobytes()).hexdigest()!=r['density_array_sha256']:
            raise ValueError('Density artifact differs')
        with np.load(path) as saved:
            if not np.array_equal(saved['density'],density):raise ValueError('Saved density differs')
        if not math.isfinite(r['inference_seconds_per_image']) or r['inference_seconds_per_image']<=0:
            raise ValueError('Invalid timing')
        d=r['detail'];v=protocol['variant']
        should_trigger=d['median_exemplar_side_model_pixels']<=32 or d['base_count']>=200
        if d['triggered']!=should_trigger or d['base_count']!=base[name]['prediction']:raise ValueError('Trigger mismatch')
        if v.startswith('m3-') and (not d['triggered'] or '-repeat-' in v):
            if not np.array_equal(boxes,np.asarray(base[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)):
                raise ValueError('Control/untriggered image differs')
        if '-repeat-' in v and d['triggered']:
            if not d['repeat_bitwise_equal'] or d['backbone_passes']!=1+int(v[-1])**2:raise ValueError('Repeat audit failed')
        if d['tiling_applied']:
            if len(d['tiles'])!=d['grid']**2:raise ValueError('Missing tile')
            for t in d['tiles']:
                if t['refinement_executed']!=d['base_refinement']['refinement_executed']:
                    raise ValueError('Inconsistent SAM branch')
                if v!='m3-v2-branch-only' and t['threshold_maximum']!=d['global_maximum_objectness']:
                    raise ValueError('Inconsistent threshold')
            if sum(t['owned_count'] for t in d['tiles'])-d['cross_tile_removed']!=len(boxes):
                raise ValueError('Merge count mismatch')
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    if sha(root/'data/fsc147/annotations/ImageClasses_FSC147.txt')!=splits['source_sha256']['ImageClasses_FSC147.txt']:
        raise ValueError('Class labels changed')
    result={'status':'passed','n':len(rows),'variant':protocol['variant'],
            'annotation_and_spatial_reconstruction':'passed',
            'files_sha256':{n:sha(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}}
    result.update(report(rows,base,labels,cfg['bootstrap_replicates']) if protocol['phase']=='full'
                  else {'smoke_not_used_for_selection':True})
    result['diagnostics']={'triggered_images':sum(r['detail']['triggered'] for r in rows),
                           'tiled_images':sum(r['detail']['tiling_applied'] for r in rows),
                           'cross_tile_removed':sum(r['detail']['cross_tile_removed'] for r in rows),
                           'cases':[{k:r[k] for k in ('image','truth','prediction','detail')}
                                    for r in rows if r['image'] in ('865.jpg','3665.jpg')]}
    write_json(out/'audit.json',result)


def analyze(root):
    cfg,splits,_=sources(root);out=root/'results/round3-v1/m3-crossfit'
    if out.exists():raise FileExistsError('Existing crossfit evidence')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');names=[r['image'] for r in baseline]
    base={r['image']:r for r in baseline};variants={};evidence={}
    for v in cfg['m3']['runs']:
        d=root/'results/round3-v1'/('full-'+v);a=read(d/'audit.json')
        if a['status']!='passed' or a['n']!=1286:raise ValueError('Missing full audit')
        if sha(d/'predictions.jsonl')!=a['files_sha256']['predictions.jsonl']:raise ValueError('Post-audit mutation')
        variants[v]={r['image']:r for r in read_rows(d/'predictions.jsonl')};evidence[v]=sha(d/'audit.json')
    folds={'A':splits['val_A'],'B':splits['val_B']}
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    for f in ('A','B'):
        if {labels[n] for n in folds[f]}!=set(splits['val_'+f+'_classes']):raise ValueError('Class fold membership differs')
    groups={};choices={};final={}
    for role,order in [('m3',cfg['m3']['candidate_order']),('mask',cfg['m3']['mask_candidate_order'])]:
        chosen,records=crossfit_choices(names,folds,variants,base,order)
        groups[role]=[variants[chosen[n]][n] for n in names];choices[role]=records
        final[role]=select_candidate({k:[variants[k][n]['prediction'] for n in names] for k in order},
                                    [base[n]['truth'] for n in names],[base[n]['prediction'] for n in names],order)[1]
        if role=='m3':groups['repeat']=[variants['m3-v2-repeat-g'+chosen[n][-1]][n] for n in names]
    groups['branch-only']=[variants['m3-v2-branch-only'][n] for n in names]
    result={'status':'complete','scope':'development_only_no_test','selection':choices,'all_val_final_selection':final,
            'prerequisite_audits_sha256':evidence,'groups':{k:report(v,base,labels,cfg['bootstrap_replicates']) for k,v in groups.items()},
            'contrasts':{k:{u:paired_intervals(groups['m3'],{r['image']:r for r in groups[k]},labels,cfg['bootstrap_replicates'],u)
                            for u in ('image','class')} for k in ('repeat','branch-only','mask')},
            'combination_admission_requires_fit_fold_only':True,'independent_crossfit_audit':'pending'}
    out.mkdir()
    for role,rows in groups.items():
        with (out/(role+'.jsonl')).open('x') as f:
            for r in rows:f.write(json.dumps({k:r[k] for k in ('image','truth','prediction','game1','game2','localization','inference_seconds_per_image')})+'\n')
    write_json(out/'summary.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--label');p.add_argument('--crossfit',action='store_true');a=p.parse_args()
    if a.crossfit:analyze(a.root)
    elif a.label:audit(a.root,a.label)
    else:p.error('Choose a label or crossfit')
