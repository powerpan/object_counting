"""Independent reconstruction and category-fold analysis of saved TTA outputs."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_tta_common import sources,read,sha,write_json
from audit_b0_results import read_rows
from audit_round3_m3 import report
from analyze_m2_m3 import paired_intervals
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import cell_mass,point_cells,grid_features,localization
from planb.round3_selection import crossfit_choices
from planb.round2.fusion import select_candidate


def close(a,b):
    if not np.allclose(a,b,rtol=1e-12,atol=1e-8):
        raise ValueError('TTA reconstruction differs')


def rebuild_view(view,height,width):
    b=np.asarray(view['boxes_normalized_xyxy'],dtype=np.float32).reshape(-1,4)*(1024/view['effective_scale'])
    if view['flipped']:
        b=b.astype(np.float64);left=width-b[:,2].copy();right=width-b[:,0].copy();b[:,0]=left;b[:,2]=right
        centers=(b[:,:2]+b[:,2:])/2;bounds=np.array([width,height],dtype=np.float64)
        if ((centers < -8*np.spacing(bounds)) | (centers > bounds+8*np.spacing(bounds))).any():
            raise ValueError('Inverse geometry outside image')
        shift=np.clip(centers,0.,np.nextafter(bounds,0.))-centers;b[:,:2]+=shift;b[:,2:]+=shift
    close(b,np.asarray(view['boxes_original_xyxy'],dtype=np.float64).reshape(-1,4))
    if len(b)!=view['count'] or len(b)!=len(view['objectness']) or not np.isfinite(view['objectness']).all():
        raise ValueError('Invalid view detection record')
    return boxes_to_density(b,height,width),b


def correlation(rows):
    x=np.array([r['detail']['view_count_std'] for r in rows],dtype=np.float64)
    y=np.array([abs(r['prediction']-r['truth']) for r in rows],dtype=np.float64)
    value=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
    return {'definition':'Pearson population view-count standard deviation versus averaged-count absolute error',
            'pearson_r':value,'n':len(x),'view_count_std_mean':float(x.mean()),
            'view_count_std_median':float(np.median(x)),'view_count_std_max':float(x.max())}


def audit(root,label):
    cfg,splits,receipt=sources(root);out=root/'results/round3-tta-v1'/label
    if (out/'audit.json').exists():raise FileExistsError('Existing TTA audit')
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');rows=read_rows(out/'predictions.jsonl')
    if summary['status']!='complete' or summary['n']!=len(rows):raise ValueError('Incomplete TTA output')
    if protocol['files_sha256']!=receipt['files_sha256'] or protocol['config_sha256']!=sha(root/'configs/round3_v1.json'):
        raise ValueError('Source record differs')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in baseline}
    if protocol['baseline_predictions_sha256']!=sha(root/cfg['baseline_result']/'predictions.jsonl'):
        raise ValueError('Baseline changed')
    names=list(base)
    if protocol['phase']!='full':
        from run_m2_m3 import smoke_ids
        names=smoke_ids(names,base)
    if names!=protocol['ids'] or [r['image'] for r in rows]!=names:raise ValueError('Wrong TTA image order')
    annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    expected_views={'T2':[(1.,False),(1.,True)],'T4':[(1.,False),(1.,True),(.8,False),(.8,True)],'tta-repeat':[(1.,False)]*4}
    v=protocol['variant']
    for row in rows:
        n=row['image'];h,w=row['height'],row['width'];anno=annotations[n]
        points=points_for_game(anno['points'],h,w)
        if row['truth']!=len(points) or len(points)!=base[n]['truth']:raise ValueError('Truth differs')
        if (w,h)!=(base[n]['width'],base[n]['height']):raise ValueError('Dimensions differ')
        prompts=np.asarray(anno['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4)
        close(prompts,row['exemplars_xyxy']);detail=row['detail'];views=detail['views']
        if [(x['scale_factor'],x['flipped']) for x in views]!=expected_views[v]:raise ValueError('Unexpected views')
        maps=[];counts=[]
        for view in views:
            scale=detail['base_scale']*view['scale_factor']
            if scale!=view['effective_scale'] or (int(h*scale),int(w*scale))!=(view['valid_height'],view['valid_width']):
                raise ValueError('TTA scaling record differs')
            den,boxes=rebuild_view(view,h,w);maps.append(den);counts.append(len(boxes))
        first=np.asarray(views[0]['boxes_original_xyxy'],dtype=np.float32).reshape(-1,4)
        if not np.array_equal(first,np.asarray(base[n]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)):
            raise ValueError('Original pass differs from B0')
        close(first,np.asarray(row['boxes_xyxy'],dtype=np.float64).reshape(-1,4));close(views[0]['objectness'],row['objectness'])
        if v=='tta-repeat':
            if not detail['repeat_bitwise_equal'] or any(not np.array_equal(maps[0],d) for d in maps[1:]):
                raise ValueError('Repeated maps differ')
            density=maps[0]
        else:
            density=np.stack(maps).mean(axis=0,dtype=np.float64)
        close(np.mean(counts),row['prediction']);close(np.std(counts,ddof=0),detail['view_count_std'])
        close(density.sum(),row['prediction']);close(density.sum(),row['density_mass'])
        mass=cell_mass(density);_,truth8=point_cells(points,h,w);close(mass,row['mass8']);close(truth8,row['truth8'])
        for level in (1,2):close(game(density,points,level),row['game'+str(level)])
        close(grid_features(first,row['objectness'],prompts,density),row['grid_features'])
        if localization(first,points,prompts)!=row['localization']:raise ValueError('Localization differs')
        path=root/'work/round3-tta-v1'/label/(Path(n).stem+'.npz')
        if sha(path)!=row['density_file_sha256']:raise ValueError('Density file changed')
        with np.load(path) as saved:
            actual=saved['density'];close(actual,density)
            if hashlib.sha256(actual.tobytes()).hexdigest()!=row['density_array_sha256']:
                raise ValueError('Density array changed')
        if not math.isfinite(row['inference_seconds_per_image']) or row['inference_seconds_per_image']<=0:
            raise ValueError('Invalid timing')
    if protocol['phase']=='gate' and not summary['native_raw_gate_passed']:raise ValueError('Raw model gate missing')
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    if sha(root/'data/fsc147/annotations/ImageClasses_FSC147.txt')!=splits['source_sha256']['ImageClasses_FSC147.txt']:
        raise ValueError('Class labels changed')
    result={'status':'passed','n':len(rows),'variant':v,'phase':protocol['phase'],
            'view_geometry_mass_game_localization_reconstructed':True,
            'files_sha256':{n:sha(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}}
    if protocol['phase']=='full':
        result.update(report(rows,base,labels,cfg['bootstrap_replicates']));result['view_disagreement']=correlation(rows)
    else:result['not_used_for_selection']=True
    write_json(out/'audit.json',result)


def analyze(root):
    cfg,splits,receipt=sources(root);out=root/'results/round3-tta-v1/crossfit'
    if out.exists():raise FileExistsError('Existing TTA crossfit')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in baseline};names=list(base)
    variants={};evidence={}
    for v in ('T2','T4','tta-repeat'):
        d=root/'results/round3-tta-v1'/('full-'+v);a=read(d/'audit.json')
        if a['status']!='passed' or a['n']!=1286 or sha(d/'predictions.jsonl')!=a['files_sha256']['predictions.jsonl']:
            raise ValueError('Incomplete or changed TTA evidence')
        variants[v]={r['image']:r for r in read_rows(d/'predictions.jsonl')};evidence[v]=sha(d/'audit.json')
    folds={'A':splits['val_A'],'B':splits['val_B']};order=cfg['tta']['candidate_order']
    selected,choices=crossfit_choices(names,folds,variants,base,order)
    rows=[variants[selected[n]][n] for n in names]
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    for f in ('A','B'):
        if {labels[n] for n in folds[f]}!=set(splits['val_'+f+'_classes']):raise ValueError('Fold category mismatch')
    _,final=select_candidate({v:[variants[v][n]['prediction'] for n in names] for v in order},
                            [base[n]['truth'] for n in names],[base[n]['prediction'] for n in names],order)
    result={'status':'complete','scope':'development_only_no_test','selection':choices,'all_val_final_selection':final,
            'prerequisite_audits_sha256':evidence,'result':report(rows,base,labels,cfg['bootstrap_replicates']),
            'view_disagreement':correlation(rows),'independent_crossfit_audit':'pending',
            'combination_admission_requires_fit_fold_only':True}
    out.mkdir()
    with (out/'predictions.jsonl').open('x') as f:
        for r in rows:
            keep={k:r[k] for k in ('image','truth','prediction','game1','game2','localization','inference_seconds_per_image')}
            keep['selected_variant']=selected[r['image']];f.write(json.dumps(keep)+'\n')
    write_json(out/'summary.json',result)


def audit_crossfit(root):
    from audit_round3_crossfit import choose,metrics
    cfg,splits,_=sources(root);out=root/'results/round3-tta-v1/crossfit';s=read(out/'summary.json')
    if (out/'audit.json').exists():raise FileExistsError('Existing independent TTA audit')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in baseline};names=list(base)
    variants={}
    for v in ('T2','T4','tta-repeat'):
        d=root/'results/round3-tta-v1'/('full-'+v);a=read(d/'audit.json')
        if sha(d/'audit.json')!=s['prerequisite_audits_sha256'][v] or sha(d/'predictions.jsonl')!=a['files_sha256']['predictions.jsonl']:
            raise ValueError('Post-audit mutation')
        variants[v]={r['image']:r for r in read_rows(d/'predictions.jsonl')}
    wanted={};order=cfg['tta']['candidate_order']
    for fit,assess in [('A','B'),('B','A')]:
        chosen=choose(splits['val_'+fit],variants,base,order)
        if chosen!=s['selection'][assess]['selected'] or s['selection'][assess]['fit_fold']!=fit:
            raise ValueError('TTA fold choice differs')
        for n in splits['val_'+assess]:wanted[n]=(chosen,variants[chosen][n])
    if choose(names,variants,base,order)!=s['all_val_final_selection']['selected']:raise ValueError('Final choice differs')
    rows=read_rows(out/'predictions.jsonl')
    if [r['image'] for r in rows]!=names:raise ValueError('Crossfit IDs differ')
    for r in rows:
        v,w=wanted[r['image']]
        if r['selected_variant']!=v or any(r[k]!=w[k] for k in r if k!='selected_variant'):
            raise ValueError('Crossfit prediction differs')
    groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
            '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    for k,rs in groups.items():
        for field,value in metrics(rs).items():close(value,s['result']['metrics'][k][field])
    m=metrics(rows);b=metrics(baseline)
    passed=m['mae']<=1.02*b['mae'] and m['rmse']<b['rmse'] and metrics(groups['>=300'])['mae']<metrics([r for r in baseline if r['truth']>=300])['mae']
    if passed!=s['result']['candidate_gate']:raise ValueError('Candidate gate differs')
    write_json(out/'audit.json',{'status':'passed','n':1286,'fold_local_choices_reproduced':True,
               'selected_rows_and_metrics_reproduced':True,'summary_sha256':sha(out/'summary.json'),
               'scope':'development_only_no_test'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--label');g.add_argument('--crossfit',action='store_true');g.add_argument('--audit-crossfit',action='store_true');a=p.parse_args()
    if a.label:audit(a.root,a.label)
    elif a.crossfit:analyze(a.root)
    else:audit_crossfit(a.root)
