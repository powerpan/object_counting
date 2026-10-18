"""Fit-fold module decisions and image routing; no new inference or calibration."""
from planb.round2.fusion import select_candidate
from planb.round3_selection import candidate_gate


def select_module(fit_names, variants, baseline, order):
    choice, selection = select_candidate(
        {v:[variants[v][n]['prediction'] for n in fit_names] for v in order},
        [baseline[n]['truth'] for n in fit_names],
        [baseline[n]['prediction'] for n in fit_names], order)
    return {'selected':choice, 'selection':selection,
            'admitted':candidate_gate([variants[choice][n] for n in fit_names],baseline)}


def route_image(name, decision, variants):
    """Route using only fixed decisions and the original B0 trigger, never truth."""
    m3 = decision['m3']
    if m3['admitted'] and variants[m3['selected']][name]['triggered']:
        return m3['selected']
    tta, mask = decision['tta'], decision['mask']
    if tta['admitted']:
        return tta['selected']+'-refine-'+mask['selected'].rsplit('-',1)[1] if mask['admitted'] else tta['selected']
    return mask['selected'] if mask['admitted'] else 'B0'


def make_plan(names, folds, labels, baseline, variants, calibration, config):
    if len(names)!=len(set(names)) or set(folds['A']) & set(folds['B']) or set(folds['A']+folds['B'])!=set(names):
        raise ValueError('Invalid outer image partition')
    if {labels[n] for n in folds['A']} & {labels[n] for n in folds['B']}:
        raise ValueError('Outer categories overlap')
    orders={'m3':config['m3']['candidate_order'], 'tta':config['tta']['candidate_order'],
            'mask':config['m3']['mask_candidate_order']}
    for v in [x for order in orders.values() for x in order]:
        if set(variants[v])!=set(names):raise ValueError('Missing source images')
    g2,g3=orders['m3']
    if any(type(variants[g2][n]['triggered']) is not bool or variants[g2][n]['triggered']!=variants[g3][n]['triggered'] for n in names):
        raise ValueError('Original M3 triggers differ')
    decisions={};routes={};required=set()
    for fit_fold,assess_fold,fit_names,assessment in (
            ('A','B',folds['A'],folds['B']),('B','A',folds['B'],folds['A']),
            ('all-val','future-only',names,names)):
        cal = calibration['final_decision'] if fit_fold=='all-val' else calibration['outer_decisions'][assess_fold]
        if cal['fit_images']!=fit_names:raise ValueError('Calibration decision has a different fitting fold')
        decision={'fit_fold':fit_fold,'assessment_fold':assess_fold,'fit_images':list(fit_names),
                  **{role:select_module(fit_names,variants,baseline,order) for role,order in orders.items()},
                  'calibration':{'selected':cal['selected'],'admitted':cal['admitted'],
                                 'source':'audited_B0_calibration_inner_class_predictions'}}
        decisions[assess_fold]=decision
        mapping={n:route_image(n,decision,variants) for n in assessment}
        routes[assess_fold]=mapping
        required.update(v for v in mapping.values() if v not in variants and v!='B0')
    return {'status':'planned','scope':'fit-fold module admission and routing; no combination scores or test',
            'decisions':decisions,'routes':routes,'missing_inference_variants':sorted(required),
            'requires_calibration_refit':any(d['calibration']['admitted'] for d in decisions.values()),
            'final_route_scope':'all-val route inventory only; not an out-of-fold score',
            'all_module_admission_uses_fitting_labels_only':True}
