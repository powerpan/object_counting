"""Frozen descriptive summaries and paired intervals. No method selection."""
import numpy as np


def size_group(exemplars,width,height,cutoffs):
    e=np.asarray(exemplars,dtype=float).reshape(3,4)
    area=np.prod(e[:,2:]-e[:,:2],axis=1)
    if min(width,height)<=0 or (area<=0).any(): raise ValueError('Invalid size inputs')
    r=float(np.sqrt(area.mean()/(width*height)))
    return r, 'small' if r<=cutoffs[0] else ('medium' if r<=cutoffs[1] else 'large')


def aggregate(rows,method):
    if not rows:return {'n':0,'mae':None,'rmse':None}
    y=np.array([r['truth'] for r in rows],dtype=np.float64)
    p=np.array([r[method]['prediction'] for r in rows],dtype=np.float64)
    if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite counts')
    loc={k:sum(r[method]['localization'][k] for r in rows) for k in ('tp','fp','fn')}
    tp,fp,fn=[loc[k] for k in ('tp','fp','fn')]
    return {'n':len(y),'object_count':int(y.sum()),'truth_min':int(y.min()),'truth_max':int(y.max()),
            'truth_mean':float(y.mean()),'truth_median':float(np.median(y)),
            'mae':float(np.abs(p-y).mean()),'rmse':float(np.sqrt(np.square(p-y).mean())),
            'game1':float(np.mean([r[method]['game1'] for r in rows])),
            'game2':float(np.mean([r[method]['game2'] for r in rows])),
            'localization':{**loc,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)},
            'inference_seconds_mean':float(np.mean([r[method]['inference_seconds'] for r in rows])),
            'inference_seconds_median':float(np.median([r[method]['inference_seconds'] for r in rows])),
            'peak_allocated_mib':max(r[method]['peak_allocated_mib'] for r in rows)}


def intervals(rows,unit,replicates=2000):
    if not rows:return {'n':0,'unit':unit,'mae_delta_95ci':None,'rmse_delta_95ci':None}
    e=np.array([r['final']['prediction']-r['truth'] for r in rows],dtype=float)
    b=np.array([r['b0']['prediction']-r['truth'] for r in rows],dtype=float)
    units=np.arange(len(rows)) if unit=='image' else np.array([r['class'] for r in rows])
    _,inv=np.unique(units,return_inverse=True);counts=np.bincount(inv);k=len(counts)
    ae=np.bincount(inv,weights=np.abs(e)-np.abs(b));se=np.bincount(inv,weights=e**2);bs=np.bincount(inv,weights=b**2)
    weights=np.random.default_rng(42).multinomial(k,np.full(k,1/k),size=replicates);den=weights@counts
    dm=weights@ae/den;dr=np.sqrt(weights@se/den)-np.sqrt(weights@bs/den)
    return {'unit':unit,'units':k,'replicates':replicates,'seed':42,'n':len(rows),
            'mae_delta':float(np.mean(np.abs(e)-np.abs(b))),
            'rmse_delta':float(np.sqrt(np.mean(e**2))-np.sqrt(np.mean(b**2))),
            'mae_delta_95ci':np.quantile(dm,[.025,.975]).tolist(),'rmse_delta_95ci':np.quantile(dr,[.025,.975]).tolist()}


def success(b,f,bd,fd,ci,rules):
    checks={'rmse_at_least_2pct_lower':f['rmse']<=rules['rmse_ratio_max']*b['rmse'],
            'rmse_image_ci_below_zero':ci['rmse_delta_95ci'][1]<rules['rmse_image_ci_upper_strict_max'],
            'dense_mae_at_least_5pct_lower':bool(bd['n'] and fd['mae']<=rules['dense_mae_ratio_max']*bd['mae']),
            'mae_increase_at_most_2pct':f['mae']<=rules['mae_ratio_max']*b['mae']}
    passed=all(checks.values())
    return {'checks':checks,'passed':passed,'conclusion':'success' if passed else (
        'point_estimate_improvement_but_uncertain_or_incomplete' if f['rmse']<b['rmse'] else 'not_improved')}


def report(rows,cfg,dataset):
    groups={'overall':rows,'<50':[r for r in rows if r['truth']<50],
            '50-299':[r for r in rows if 50<=r['truth']<300],'>=300':[r for r in rows if r['truth']>=300]}
    groups.update({'size_'+g:[r for r in rows if r['size_group']==g] for g in ['small','medium','large']})
    result={k:{'b0':aggregate(v,'b0'),'final':aggregate(v,'final'),
               'paired':{u:intervals(v,u,cfg['bootstrap']['replicates']) for u in ['image','class']}} for k,v in groups.items()}
    classes={c:{m:aggregate([r for r in rows if r['class']==c],m) for m in ['b0','final']} for c in sorted({r['class'] for r in rows})}
    output={'dataset':dataset,'n':len(rows),'class_count':len(classes),'groups':result,'classes':classes,
            'triggered_images':sum(r['detail']['triggered'] for r in rows),
            'trigger_fraction':sum(r['detail']['triggered'] for r in rows)/len(rows),
            'human_occlusion_clutter_evaluation':'not_performed_per_owner_amendment','cohens_kappa':None}
    if dataset=='fsc147-test':
        output['success']=success(result['overall']['b0'],result['overall']['final'],result['>=300']['b0'],
                                  result['>=300']['final'],result['overall']['paired']['image'],cfg['success'])
    else:output['success']=None
    return output


def qualitative_selection(rows,cfg):
    error=lambda r:abs(r['final']['prediction']-r['truth'])
    delta=lambda r:error(r)-abs(r['b0']['prediction']-r['truth'])
    return {'random_six':cfg['qualitative']['random_six'],
            'worst_final':[r['image'] for r in sorted(rows,key=lambda r:(-error(r),r['image']))[:6]],
            'most_improved':[r['image'] for r in sorted(rows,key=lambda r:(delta(r),r['image']))[:3]],
            'most_regressed':[r['image'] for r in sorted(rows,key=lambda r:(-delta(r),r['image']))[:3]]}
