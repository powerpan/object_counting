"""Reconstruct calibration inputs, nested decisions, maps and metrics on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_calibration_io import load,read,sha,write,read_rows,utc,report
from planb.density import boxes_to_density,points_for_game,game
from planb.round3_calibration import fit


def close(a,b):
    if not np.allclose(a,b,rtol=1e-12,atol=1e-8):raise ValueError('Independent calibration reconstruction differs')


def check_tree(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('Report fields differ')
        for key in expected:check_tree(actual[key],expected[key])
    elif isinstance(expected,(str,bool)) or expected is None:
        if actual!=expected:raise ValueError('Report value differs')
    else:close(actual,expected)


def metric(rows):
    errors=[r['prediction']-r['truth'] for r in rows];n=len(errors)
    return {'n':n,'mae':sum(abs(x) for x in errors)/n,'rmse':math.sqrt(sum(x*x for x in errors)/n),
            'game1':sum(r['game1'] for r in rows)/n,'game2':sum(r['game2'] for r in rows)/n}


def passes(rows,base):
    dense=[r for r in rows if r['truth']>=300]
    if not dense:return False
    m=metric(rows);b=metric([base[r['image']] for r in rows])
    return m['mae']<=1.02*b['mae'] and m['rmse']<b['rmse'] and metric(dense)['mae']<metric([base[r['image']] for r in dense])['mae']


def reconstruct(baseline,annotations):
    records={}
    for row in baseline:
        n=row['image'];h,w=row['height'],row['width'];a=annotations[n]
        boxes=np.asarray(row['boxes_xyxy'],dtype=np.float64).reshape(-1,4)
        e=np.asarray(a['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(-1,4).astype(np.float64)
        pts=points_for_game(a['points'],h,w);den=boxes_to_density(boxes,h,w)
        ex=float(np.median(np.sum(e[:,2:]-e[:,:2],axis=1)/2))
        cy=(boxes[:,1]+boxes[:,3])/2;cx=(boxes[:,0]+boxes[:,2])/2
        lengths=np.sum(boxes[:,2:]-boxes[:,:2],axis=1)/2
        ys=np.array([int(j*h/8) for j in range(9)]);xs=np.array([int(i*w/8) for i in range(9)])
        mass=np.zeros((8,8));truth=np.zeros((8,8));x=np.zeros((8,8,4))
        for j in range(8):
            for i in range(8):
                mass[j,i]=den[ys[j]:ys[j+1],xs[i]:xs[i+1]].sum()
                truth[j,i]=((pts[:,0]>=xs[i])&(pts[:,0]<xs[i+1])&(pts[:,1]>=ys[j])&(pts[:,1]<ys[j+1])).sum()
                inside=(cx>=xs[i])&(cx<xs[i+1])&(cy>=ys[j])&(cy<ys[j+1])
                side=float(lengths[inside].mean()) if inside.any() else ex
                area=(ys[j+1]-ys[j])*(xs[i+1]-xs[i])*row['scale_factor']**2
                x[j,i]=[math.log1p(mass[j,i]),math.log(ex),math.log(side),mass[j,i]/area]
        close(mass.sum(),row['prediction']);close(truth.sum(),row['truth'])
        records[n]={'image':n,'truth':row['truth'],'prediction':row['prediction'],
                    'features':x.tolist(),'mass8':mass.tolist(),'truth8':truth.tolist(),
                    'game1':row['game1'],'game2':row['game2']}
    return records


def audit_model(model,names,records,config):
    if model['fit_images']!=names or not model['converged']:raise ValueError('Wrong fitting images or convergence')
    d=np.concatenate([np.asarray(records[n]['mass8']).ravel() for n in names]);valid=d>0
    y=np.concatenate([np.asarray(records[n]['truth8']).ravel() for n in names])[valid];d=d[valid]
    x=np.concatenate([np.asarray(records[n]['features']).reshape(64,4) for n in names])[valid]
    if model['positive_mass_cells']!=len(d) or model['zero_mass_cells_excluded']!=len(names)*64-len(d):
        raise ValueError('Wrong regression support')
    if model['kind']=='global':
        close(model['mass_sum'],d.sum());close(model['target_sum'],y.sum())
        if y.sum()==0:
            if model['intercept'] is not None or not model['zero_target_boundary']:raise ValueError('Wrong global boundary')
        else:close(model['intercept'],math.log(float(y.sum()/d.sum())))
    else:
        mean=x.mean(0);std=x.std(0);constant=np.all(x==x[0],axis=0);std[constant|(std==0)]=1
        close(mean,model['mean']);close(std,model['std'])
        if constant.tolist()!=model['constant_features']:raise ValueError('Constant feature handling differs')
        beta=np.asarray(model['coefficients']);z=np.column_stack([np.ones(len(d)),(x-mean)/std])
        eta=np.log(d)+z@beta;mu=np.exp(eta)
        obj=float(np.mean(mu-y*eta)+config['l2']*np.dot(beta[1:],beta[1:]))
        grad=z.T@(mu-y)/len(d);grad[1:]+=2*config['l2']*beta[1:]
        close(obj,model['objective']);close(np.abs(grad).max(),model['gradient_max_abs'])
        # Same fixed optimizer, independently reconstructed inputs. This is repeatability,
        # in addition to the separate objective/prediction reconstruction above.
        refit=fit([records[n] for n in names],'regional',config)
        if not np.allclose(refit['coefficients'],beta,rtol=1e-7,atol=1e-7):
            raise ValueError('Fixed optimizer refit differs')


def reconstruct_prediction(record,model):
    if model['kind']=='global':
        value=.5 if model['zero_target_boundary'] else math.exp(max(math.log(.5),min(math.log(2),model['intercept'])))
        gain=np.full((8,8),value)
    else:
        x=(np.asarray(record['features'])-model['mean'])/model['std'];beta=np.asarray(model['coefficients'])
        eta=beta[0]+np.sum(x*beta[1:],axis=-1)
        gain=np.exp(np.minimum(np.maximum(eta,math.log(.5)),math.log(2)))
    mass=np.asarray(record['mass8'])*gain;truth=np.asarray(record['truth8']);games={}
    for level in (1,2):
        size=2**level;step=8//size;value=0.
        for j in range(size):
            for i in range(size):value+=abs(float(mass[j*step:(j+1)*step,i*step:(i+1)*step].sum()-truth[j*step:(j+1)*step,i*step:(i+1)*step].sum()))
        games['game'+str(level)]=value
    return {'image':record['image'],'truth':record['truth'],'prediction':float(mass.sum()),
            'mass8':mass.tolist(),'gains8':gain.tolist(),**games}


def check_row(actual,expected):
    if actual['image']!=expected['image'] or actual['truth']!=expected['truth']:raise ValueError('Wrong assessment identity')
    for key in ('prediction','mass8','gains8','game1','game2'):close(actual[key],expected[key])


def audit_decision(decision,names,records,labels,base,config):
    if decision['fit_images']!=names:raise ValueError('Outer labels entered selection')
    classes=sorted({labels[n] for n in names});groups={'A':set(classes[::2]),'B':set(classes[1::2])}
    folds={f:[n for n in names if labels[n] in groups[f]] for f in ('A','B')}
    if decision['inner_folds']!=folds:raise ValueError('Wrong inner category partition')
    predictions={k:{} for k in ('regional','global')}
    for fit_fold,assess_fold in (('A','B'),('B','A')):
        for kind in predictions:
            model=decision['inner_models'][assess_fold][kind];audit_model(model,folds[fit_fold],records,config)
            for n in folds[assess_fold]:predictions[kind][n]=reconstruct_prediction(records[n],model)
    metrics={}
    for kind in predictions:
        rows=[predictions[kind][n] for n in names];saved=decision['inner_predictions'][kind]
        if len(saved)!=len(rows):raise ValueError('Incomplete inner predictions')
        for x,y in zip(saved,rows):check_row(x,y)
        metrics[kind]=metric(rows)
        for k,v in decision['metrics'][kind].items():close(v,metrics[kind][k])
    selected='regional' if metrics['regional']['rmse']<metrics['global']['rmse'] else 'global'
    if selected!=decision['selected'] or bool(decision['admitted'])!=bool(passes([predictions[selected][n] for n in names],base)):
        raise ValueError('Inner-only version or admission decision differs')


def main(root,asset_root,receipt_path):
    cfg,split,receipt,baseline,annotations,labels,evidence=load(root,asset_root,receipt_path)
    out=root/'results/round3-calibration-v1'
    if (out/'audit.json').exists():raise FileExistsError('Existing calibration audit')
    protocol=read(out/'protocol.json');summary=read(out/'summary.json');models=read(out/'models.json')
    if protocol['files_sha256']!=receipt['files_sha256'] or any(protocol[k]!=v for k,v in evidence.items()):
        raise ValueError('Recorded inputs or sources differ')
    if summary['status']!='complete' or summary['n']!=1286:raise ValueError('Incomplete calibration')
    names=[r['image'] for r in baseline];base={r['image']:r for r in baseline};records=reconstruct(baseline,annotations)
    inputs=read_rows(out/'input_cells.jsonl')
    if [r['image'] for r in inputs]!=names:raise ValueError('Input order differs')
    for row in inputs:
        expected=records[row['image']]
        for key in ('truth','mass8','truth8','features'):close(row[key],expected[key])
    for fit_fold,assess_fold in (('A','B'),('B','A')):
        fit_names=split['val_'+fit_fold]
        audit_decision(models['outer_decisions'][assess_fold],fit_names,records,labels,base,cfg['m4_v3'])
        for kind in ('regional','global'):audit_model(models['outer_models'][assess_fold][kind],fit_names,records,cfg['m4_v3'])
    audit_decision(models['final_decision'],names,records,labels,base,cfg['m4_v3'])
    for kind in ('regional','global'):audit_model(models['final_models'][kind],names,records,cfg['m4_v3'])
    check_tree(summary['outer_choices'],{k:{z:v[z] for z in ('selected','admitted','metrics')}
                                        for k,v in models['outer_decisions'].items()})
    check_tree(summary['all_val_final_choice'],{z:models['final_decision'][z] for z in ('selected','admitted','metrics')})
    signs=(np.sign(models['outer_models']['A']['regional']['coefficients'][1:])==
           np.sign(models['outer_models']['B']['regional']['coefficients'][1:])).tolist()
    if signs!=summary['regional_slope_signs_agree']:raise ValueError('Coefficient direction report differs')
    rows={k:read_rows(out/(k+'.jsonl')) for k in ('regional','global','selected','admitted')}
    for group,data in rows.items():
        if [r['image'] for r in data]!=names:raise ValueError('Output order differs')
        for row in data:
            n=row['image'];fold='A' if n in set(split['val_A']) else 'B';decision=models['outer_decisions'][fold]
            kind=group if group in ('regional','global') else decision['selected']
            expected=reconstruct_prediction(records[n],models['outer_models'][fold][kind])
            if group=='admitted' and not decision['admitted']:
                expected={**base[n],'mass8':records[n]['mass8'],'gains8':np.ones((8,8)).tolist()}
            check_row(row,expected)
            if row['assessment_fold']!=fold:raise ValueError('Wrong outer fold')
        groups={'overall':data,'<50':[r for r in data if r['truth']<50],
                '50-299':[r for r in data if 50<=r['truth']<300],'>=300':[r for r in data if r['truth']>=300]}
        for key,group_rows in groups.items():
            for key2,value in metric(group_rows).items():close(value,summary['roles'][group]['metrics'][key][key2])
        if bool(passes(data,base))!=summary['roles'][group]['candidate_gate']:raise ValueError('Reported development gate differs')
        # Fixed statistics are reproduced separately; no resampling rule is changed.
        check_tree(summary['roles'][group],report(data,base,labels,cfg['bootstrap_replicates']))
    map_max_error=0.
    for i,n in enumerate(names):
        b=base[n];h,w=b['height'],b['width'];den=boxes_to_density(b['boxes_xyxy'],h,w)
        pts=points_for_game(annotations[n]['points'],h,w);ys=[int(j*h/8) for j in range(9)];xs=[int(j*w/8) for j in range(9)]
        for role,data in rows.items():
            row=data[i];corrected=np.empty_like(den)
            for j in range(8):
                for k in range(8):corrected[ys[j]:ys[j+1],xs[k]:xs[k+1]]=den[ys[j]:ys[j+1],xs[k]:xs[k+1]]*row['gains8'][j][k]
            map_max_error=max(map_max_error,abs(corrected.sum()-row['prediction']))
            if map_max_error>1e-6:raise ValueError('Corrected map mass differs')
            for level in (1,2):close(game(corrected,pts,level),row['game'+str(level)])
    files=['protocol.json','input_cells.jsonl','models.json','summary.json']+[k+'.jsonl' for k in rows]
    write(out/'audit.json',{'status':'passed','n':1286,'verified_utc':utc(),
          'inputs_and_features_independently_reconstructed':True,'fold_local_selection_and_admission_reproduced':True,
          'coefficients_refit_with_same_fixed_optimizer':True,'objective_prediction_maps_metrics_independently_recomputed':True,
          'paired_intervals_recomputed_with_registered_statistics':True,
          'maximum_map_mass_error':float(map_max_error),'gpu_seconds':0,
          'files_sha256':{n:sha(out/n) for n in files},'scope':'development_only_no_test'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.asset_root.resolve(),a.receipt.resolve())
