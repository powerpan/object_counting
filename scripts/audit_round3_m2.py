"""Independent M2-v2 policy/density reconstruction and fixed category crossfit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_m2_common import sources,read,sha,write_json
from audit_b0_results import read_rows
from audit_round3_m3 import report
from audit_round3_crossfit import choose,metrics
from audit_round3_calibration import check_tree
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import cell_mass,point_cells,localization
from planb.round3_selection import crossfit_choices,candidate_gate
from planb.round2.fusion import select_candidate


def close(a,b):
    if not np.allclose(a,b,rtol=1e-12,atol=1e-8):raise ValueError('M2 independent reconstruction differs')


def independent_selection(boxes,exemplars,relative,cosine,cfg):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4);e=np.asarray(exemplars,dtype=np.float64).reshape(3,4)
    r=np.asarray(relative);c=np.asarray(cosine)
    if r.shape!=(len(b),) or c.shape!=(len(b),) or not all(np.isfinite(x).all() for x in (b,e,r,c)):
        raise ValueError('Invalid M2 policy input')
    if (e[:,2:]<=e[:,:2]).any() or (b[:,2:]<b[:,:2]).any():raise ValueError('Invalid M2 geometry')
    centers=(b[:,:2]+b[:,2:])/2;orig=[math.sqrt((q[2]-q[0])*(q[3]-q[1])) for q in e]
    eligible=[];rejected=[];other=[];sizes=[]
    for i,q in enumerate(b):
        count=int(((centers[:,0]>=q[0])&(centers[:,0]<=q[2])&(centers[:,1]>=q[1])&(centers[:,1]<=q[3])).sum())-1
        other.append(count);dx,dy=q[2]-q[0],q[3]-q[1];size=math.sqrt(dx*dy);sizes.append(size)
        ratio=min(max(size/x,x/max(size,1e-12)) for x in orig)
        if dx>=cfg['min_side_in_model_pixels'] and dy>=cfg['min_side_in_model_pixels'] and ratio>=cfg['min_scale_ratio'] and r[i]>=cfg['min_relative_objectness'] and c[i]>=cfg['min_mean_cosine']:
            eligible.append(i)
            if count>=cfg['other_centers_rejection_minimum']:rejected.append(i)
    chosen=[];bins=set()
    for i in sorted(eligible,key=lambda i:(-float(r[i]*c[i]),i)):
        if i in rejected:continue
        bucket=math.floor(math.log(sizes[i]/float(np.median(orig)),cfg['min_scale_ratio']))
        if bucket in bins:continue
        bins.add(bucket);chosen.append(i)
        if len(chosen)==cfg['max_extra_exemplars']:break
    return {'selected_indices':chosen,'other_centers':other,'eligible_before_single_object_filter':eligible,
            'rejected_by_single_object_filter':rejected}


def verify_inputs(root,cfg,splits):
    for name,digest in splits['source_sha256'].items():
        if sha(root/'data/fsc147/annotations'/name)!=digest:raise ValueError('M2 annotations or split changed')
    path=root/cfg['baseline_result'];a=read(path/'audit.json')
    if a['status']!='passed' or a['n']!=1286:raise ValueError('B0 audit missing')
    for name,digest in a['files_sha256'].items():
        if sha(path/name)!=digest:raise ValueError('B0 evidence changed')


def diagnostics(rows):
    counts=[c['annotated_points_inside'] for r in rows for c in r['detail']['pseudo_exemplars']]
    return {'triggered_images':sum(r['detail']['triggered'] for r in rows),
            'first_detection_candidates':sum(r['detail']['base_count'] for r in rows),
            'eligible_before_single_object_filter':sum(len(r['detail']['selection']['eligible_before_single_object_filter']) for r in rows),
            'rejected_by_single_object_filter':sum(len(r['detail']['selection']['rejected_by_single_object_filter']) for r in rows),
            'selected_pseudo_exemplars':len(counts),
            'fallback_images':sum(r['detail']['fallback'] and not r.get('diagnostic_only',False) for r in rows),
            'fallback_rule_triggered_images':sum(r['detail']['fallback'] for r in rows),
            'candidate_point_histogram':{str(k):counts.count(k) for k in sorted(set(counts))},
            'candidate_contains_2plus_points':sum(x>=2 for x in counts),
            'candidate_point_diagnosis_only_after_inference':True,
            'case_3481':next(({'truth':r['truth'],'prediction':r['prediction'],'base_count':r['detail']['base_count'],
                             'second_count':r['detail']['second_count'],
                             'fallback':r['detail']['fallback'] and not r.get('diagnostic_only',False),
                             'fallback_rule_triggered':r['detail']['fallback'],
                             'pseudo_exemplars':r['detail']['pseudo_exemplars']} for r in rows if r['image']=='3481.jpg'),None)}


def audit(root,label):
    cfg,splits,m2,receipt=sources(root);parent=root/'results/round3-m2-v1';out=parent/label
    verify_inputs(root,cfg,splits)
    if (out/'audit.json').exists():raise FileExistsError('Existing M2 audit')
    p=read(out/'protocol.json');s=read(out/'summary.json');rows=read_rows(out/'predictions.jsonl')
    if p['files_sha256']!=receipt['files_sha256'] or p['m2_config_sha256']!=sha(root/'configs/round3_m2_v1.json'):
        raise ValueError('M2 source or config changed')
    if s['status']!='complete' or s['n']!=len(rows):raise ValueError('Incomplete M2 output')
    base_rows=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in base_rows}
    names=list(base)
    if p['phase']!='full':
        from run_m2_m3 import smoke_ids
        names=smoke_ids(names,base)
    if p['ids']!=names or [r['image'] for r in rows]!=names:raise ValueError('Wrong image order')
    policy={**m2['policy'],**m2['candidates'][p['candidate']]};annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    raw_rows=[];main_predictions_sha=sha(out/'predictions.jsonl')
    for r in rows:
        n=r['image'];h,w=r['height'],r['width'];d=r['detail'];anno=annotations[n]
        pts=points_for_game(anno['points'],h,w);ex=np.asarray(anno['box_examples_coordinates'],np.float32)[:3,[0,2],:].reshape(3,4)
        if len(pts)!=r['truth'] or r['truth']!=base[n]['truth'] or (h,w)!=(base[n]['height'],base[n]['width']):raise ValueError('Wrong truth or dimensions')
        close(ex,r['exemplars_xyxy'])
        norm=np.asarray(d['first_detection_boxes_normalized_xyxy'],np.float32).reshape(-1,4)
        first=norm*(1024/d['base_scale'])
        if not np.array_equal(first,np.asarray(base[n]['boxes_xyxy'],np.float32).reshape(-1,4)):raise ValueError('First pass changed')
        if d['base_scale']!=base[n]['scale_factor'] or len(first)!=d['base_count']:raise ValueError('Base scale/count changed')
        close(norm*1024,np.asarray(d['first_detection_boxes_model_pixels']).reshape(-1,4))
        close(ex*d['base_scale'],d['original_exemplars_model_pixels'])
        selection=independent_selection(d['first_detection_boxes_model_pixels'],d['original_exemplars_model_pixels'],
                                        d['first_relative_objectness'],d['candidate_cosine'],policy)
        if selection!=d['selection']:raise ValueError('Single-object selection differs')
        ids=selection['selected_indices']
        if bool(ids)!=d['triggered'] or len(ids)!=len(d['pseudo_exemplars']):raise ValueError('Trigger differs')
        for i,c in zip(ids,d['pseudo_exemplars']):
            if c['first_detection_index']!=i or c['other_detection_centers']!=selection['other_centers'][i]:raise ValueError('Candidate identity differs')
            close(first[i],c['box_xyxy']);close(c['cosine'],d['candidate_cosine'][i]);close(c['relative_objectness'],d['first_relative_objectness'][i])
            x0,y0,x1,y1=c['box_xyxy'];truth_inside=int(((pts[:,0]>=x0)&(pts[:,0]<=x1)&(pts[:,1]>=y0)&(pts[:,1]<=y1)).sum())
            if c['annotated_points_inside']!=truth_inside:raise ValueError('Post-inference candidate diagnosis differs')
        if ids:
            original=np.asarray(d['original_exemplars_model_pixels'])
            wanted=original[np.arange(len(ids))%3] if p['repeat'] else np.asarray(d['first_detection_boxes_model_pixels'])[ids]
            close(wanted,d['appended_exemplars_model_pixels'])
        second=np.asarray(d['second_boxes_xyxy'],np.float32).reshape(-1,4);scores=np.asarray(d['second_objectness'])
        if len(second)!=d['second_count'] or len(scores)!=len(second) or not np.isfinite(second).all() or not np.isfinite(scores).all():raise ValueError('Invalid second pass')
        fallback=bool(ids) and abs(len(second)-len(first))/max(len(first),1)>policy['fallback_relative_count_difference']
        if d['fallback']!=fallback:raise ValueError('Fallback differs')
        if not ids and not np.array_equal(first,second):raise ValueError('Untriggered image changed')
        expected=first if fallback else second
        if not np.array_equal(np.asarray(r['boxes_xyxy'],np.float32).reshape(-1,4),expected) or r['prediction']!=len(expected):raise ValueError('Chosen prediction differs')
        close(r['objectness'],d['first_detection_objectness'] if fallback else scores)
        for diagnostic in ([False] if p['repeat'] else [False,True]):
            b=second if diagnostic else expected;den=boxes_to_density(b,h,w)
            path=root/'work/round3-m2-v1'/label/(Path(n).stem+('-no-fallback' if diagnostic and fallback else '')+'.npz')
            file_hash=d['no_fallback_density_file_sha256'] if diagnostic else r['density_file_sha256']
            array_hash=d['no_fallback_density_array_sha256'] if diagnostic else r['density_array_sha256']
            if sha(path)!=file_hash or hashlib.sha256(den.tobytes()).hexdigest()!=array_hash:raise ValueError('Density evidence changed')
            with np.load(path) as stored:close(den,stored['density'])
            close(den.sum(),len(b));_,truth8=point_cells(pts,h,w)
            games={f'game{lev}':game(den,pts,lev) for lev in (1,2)}
            loc=localization(b,pts,ex)
            if diagnostic:
                raw_rows.append({**r,'prediction':len(b),'density_mass':float(den.sum()),**games,
                    'boxes_xyxy':b.tolist(),'objectness':scores.tolist(),'mass8':cell_mass(den).tolist(),
                    'localization':loc,'density_file_sha256':file_hash,'density_array_sha256':array_hash,
                    'diagnostic_only':True,'source_main_predictions_sha256':main_predictions_sha})
            else:
                close(cell_mass(den),r['mass8']);close(truth8,r['truth8']);close(den.sum(),r['density_mass'])
                for k,v in games.items():close(v,r[k])
                if loc!=r['localization']:raise ValueError('Localization differs')
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    result={'status':'passed','n':len(rows),'variant':p['variant'],'phase':p['phase'],
            'policy_geometry_density_truth_reconstructed':True,'diagnostics':diagnostics(rows),
            'files_sha256':{n:sha(out/n) for n in ('protocol.json','summary.json','predictions.jsonl')}}
    if p['phase']=='full':result.update(report(rows,base,labels,cfg['bootstrap_replicates']))
    if not p['repeat']:
        raw_out=parent/(label+'-no-fallback')
        if raw_out.exists():raise FileExistsError('No-fallback diagnostic already exists')
        raw_out.mkdir()
        with (raw_out/'predictions.jsonl').open('x') as f:
            for r in raw_rows:f.write(json.dumps(r,allow_nan=False)+'\n')
        write_json(raw_out/'summary.json',{'status':'complete','n':len(raw_rows),'diagnostic_only':True,
            'source_main_predictions_sha256':sha(out/'predictions.jsonl'),
            'source':'same saved second pass; no additional inference','independently_reconstructed':True})
        raw_audit={'status':'passed','n':len(raw_rows),'diagnostic_only':True,
                   'files_sha256':{n:sha(raw_out/n) for n in ('predictions.jsonl','summary.json')}}
        if p['phase']=='full':raw_audit.update(report(raw_rows,base,labels,cfg['bootstrap_replicates']))
        write_json(raw_out/'audit.json',raw_audit)
        result['no_fallback_audit_sha256']=sha(raw_out/'audit.json')
    write_json(out/'audit.json',result)


def crossfit(root,verify=False):
    cfg,splits,m2,receipt=sources(root);parent=root/'results/round3-m2-v1';out=parent/'crossfit'
    verify_inputs(root,cfg,splits)
    if not verify and out.exists():raise FileExistsError('Existing M2 crossfit')
    baseline=read_rows(root/cfg['baseline_result']/'predictions.jsonl');base={r['image']:r for r in baseline};names=list(base)
    variants={};evidence={};order=m2['candidate_order']
    for v in order:
        for suffix in ('','-repeat','-no-fallback'):
            key=v+suffix;d=parent/('full-'+key);a=read(d/'audit.json')
            if a['status']!='passed' or a['n']!=1286 or sha(d/'predictions.jsonl')!=a['files_sha256']['predictions.jsonl']:raise ValueError('Incomplete M2 evidence')
            variants[key]={r['image']:r for r in read_rows(d/'predictions.jsonl')};evidence[key]=sha(d/'audit.json')
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    if verify:
        if (out/'audit.json').exists():raise FileExistsError('Existing M2 crossfit audit')
        saved=read(out/'summary.json');chosen={}
        for fit,assess in (('A','B'),('B','A')):
            v=choose(splits['val_'+fit],variants,base,order)
            decision=saved['selection'][assess]
            if decision['selected']!=v or decision['fit_fold']!=fit or decision['assessment_fold']!=assess or decision['fit_n']!=len(splits['val_'+fit]):
                raise ValueError('M2 fold choice differs')
            for n in splits['val_'+assess]:chosen[n]=v
        if choose(names,variants,base,order)!=saved['all_val_final_selection']['selected']:raise ValueError('M2 final choice differs')
        for role,suffix in (('m2',''),('repeat','-repeat'),('no-fallback','-no-fallback')):
            rows=read_rows(out/(role+'.jsonl'))
            if [r['image'] for r in rows]!=names:raise ValueError('M2 crossfit order differs')
            for r in rows:
                source=variants[chosen[r['image']]+suffix][r['image']]
                if r['selected_candidate']!=chosen[r['image']] or any(r[k]!=source[k] for k in r if k!='selected_candidate'):
                    raise ValueError('M2 selected rows differ')
            groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
                    '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
            for g,rr in groups.items():
                for k,v in metrics(rr).items():close(v,saved['groups'][role]['metrics'][g][k])
            if candidate_gate(rows,base)!=saved['groups'][role]['candidate_gate']:raise ValueError('M2 candidate gate differs')
            check_tree(saved['groups'][role],report(rows,base,labels,cfg['bootstrap_replicates']))
            expected_rows=[variants[chosen[n]+suffix][n] for n in names]
            if diagnostics(expected_rows)!=saved['diagnostics'][role]:raise ValueError('M2 diagnostics differ')
        discarded=saved['groups']['m2']['paired_vs_b0']['overall']['image']['mae_delta_95ci'][0]>0
        if discarded!=saved['negative_result_exclusion']:raise ValueError('M2 exclusion rule differs')
        if evidence!=saved['prerequisite_audits_sha256']:raise ValueError('M2 prerequisites changed')
        if not saved['no_fallback_is_diagnostic_only'] or saved['scope']!='development_only_no_test':raise ValueError('M2 scope changed')
        write_json(out/'audit.json',{'status':'passed','n':1286,'fold_choices_controls_and_metrics_reproduced':True,
                   'summary_sha256':sha(out/'summary.json'),'scope':'development_only_no_test'})
        return
    selected,decisions=crossfit_choices(names,{'A':splits['val_A'],'B':splits['val_B']},variants,base,order)
    _,final=select_candidate({v:[variants[v][n]['prediction'] for n in names] for v in order},
                            [base[n]['truth'] for n in names],[base[n]['prediction'] for n in names],order)
    labels=dict(x.strip().split(maxsplit=1) for x in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if x.strip())
    out.mkdir();groups={};diag={}
    for role,suffix in (('m2',''),('repeat','-repeat'),('no-fallback','-no-fallback')):
        rows=[variants[selected[n]+suffix][n] for n in names];groups[role]=report(rows,base,labels,cfg['bootstrap_replicates']);diag[role]=diagnostics(rows)
        with (out/(role+'.jsonl')).open('x') as f:
            for r in rows:
                keep={k:r[k] for k in ('image','truth','prediction','game1','game2','localization','inference_seconds_per_image')}
                keep['selected_candidate']=selected[r['image']];f.write(json.dumps(keep)+'\n')
    write_json(out/'summary.json',{'status':'complete','scope':'development_only_no_test','selection':decisions,
        'all_val_final_selection':final,'prerequisite_audits_sha256':evidence,'groups':groups,'diagnostics':diag,
        'negative_result_exclusion':groups['m2']['paired_vs_b0']['overall']['image']['mae_delta_95ci'][0]>0,
        'no_fallback_is_diagnostic_only':True,'independent_audit':'pending'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--label');g.add_argument('--crossfit',action='store_true');g.add_argument('--audit-crossfit',action='store_true')
    a=p.parse_args()
    if a.label:audit(a.root.resolve(),a.label)
    else:crossfit(a.root.resolve(),a.audit_crossfit)
