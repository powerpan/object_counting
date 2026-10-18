"""Independent scalar selection, admission and routing checks for the R3-4 plan."""
import math


def scores(rows):
    errors=[r['prediction']-r['truth'] for r in rows]
    return {'mae':sum(abs(e) for e in errors)/len(errors),
            'rmse':math.sqrt(sum(e*e for e in errors)/len(errors))}


def independent_choice(ids,variants,base,order):
    values={v:scores([variants[v][n] for n in ids]) for v in order}
    reference=scores([base[n] for n in ids])
    eligible=[v for v in order if values[v]['rmse']<=reference['rmse']]
    keys=('mae','rmse') if eligible else ('rmse','mae')
    choice=min(eligible or order,key=lambda v:(values[v][keys[0]],values[v][keys[1]],order.index(v)))
    dense=[n for n in ids if base[n]['truth']>=300]
    accept=bool(dense) and values[choice]['mae']<=1.02*reference['mae'] and values[choice]['rmse']<reference['rmse']
    if accept:accept=scores([variants[choice][n] for n in dense])['mae']<scores([base[n] for n in dense])['mae']
    return choice,accept,values,eligible,reference['rmse']


def audit_plan(plan,names,folds,labels,base,variants,calibration,config):
    if set(plan['decisions'])!={'A','B','future-only'} or set(plan['routes'])!={'A','B','future-only'}:
        raise ValueError('Unexpected plan roles')
    if set(folds['A'])&set(folds['B']) or set(folds['A']+folds['B'])!=set(names):raise ValueError('Overlapping image folds')
    if {labels[n] for n in folds['A']}&{labels[n] for n in folds['B']}:raise ValueError('Overlapping classes')
    orders={'m3':config['m3']['candidate_order'],'tta':config['tta']['candidate_order'],'mask':config['m3']['mask_candidate_order']}
    required=set();needs_cal=False
    for fit,assess,ids,evaluate in (('A','B',folds['A'],folds['B']),('B','A',folds['B'],folds['A']),('all-val','future-only',names,names)):
        d=plan['decisions'][assess]
        if d['fit_fold']!=fit or d['assessment_fold']!=assess or d['fit_images']!=ids:raise ValueError('Wrong decision fold')
        for role,order in orders.items():
            choice,accept,values,eligible,b_rmse=independent_choice(ids,variants,base,order)
            saved=d[role]
            if saved['selected']!=choice or saved['admitted']!=accept or saved['selection']['selected']!=choice:
                raise ValueError('Module selection or admission differs')
            if saved['selection']['rmse_eligible']!=eligible:raise ValueError('RMSE eligibility differs')
            rule='mae then rmse then candidate order' if eligible else 'rmse then mae then candidate order'
            if saved['selection']['selection_rule']!=rule:raise ValueError('Wrong selection rule')
            if not math.isclose(saved['selection']['baseline_rmse'],b_rmse,rel_tol=1e-12,abs_tol=1e-10):raise ValueError('Wrong fitting baseline')
            for v in order:
                for key in values[v]:
                    if not math.isclose(values[v][key],saved['selection']['metrics'][v][key],rel_tol=1e-12,abs_tol=1e-10):
                        raise ValueError('Wrong fitting metrics')
        cal=calibration['final_decision'] if fit=='all-val' else calibration['outer_decisions'][assess]
        if cal['fit_images']!=ids or d['calibration']['selected']!=cal['selected'] or d['calibration']['admitted']!=cal['admitted']:
            raise ValueError('Calibration decision changed')
        needs_cal=needs_cal or cal['admitted']
        expected={}
        for n in evaluate:
            if d['m3']['admitted'] and variants[d['m3']['selected']][n]['triggered']:
                target=d['m3']['selected']
            else:
                if d['tta']['admitted']:
                    target=d['tta']['selected']
                    if d['mask']['admitted']:target+='-refine-'+d['mask']['selected'].rsplit('-',1)[1]
                elif d['mask']['admitted']:target=d['mask']['selected']
                else:target='B0'
            expected[n]=target
            if target not in variants and target!='B0':required.add(target)
        if plan['routes'][assess]!=expected:raise ValueError('Routing differs')
    if plan['missing_inference_variants']!=sorted(required) or plan['requires_calibration_refit']!=needs_cal:
        raise ValueError('Missing-work inventory differs')
