"""Category-fold choices are made without access to the assessment labels."""
import numpy as np
from planb.round2.fusion import select_candidate


def crossfit_choices(names,folds,variants,baseline,order):
    if set(folds['A'])&set(folds['B']) or set(names)!=set(folds['A']+folds['B']):
        raise ValueError('Invalid outer folds')
    selected={};records={}
    for fit,assess in [('A','B'),('B','A')]:
        train=folds[fit]
        choice,detail=select_candidate(
            {k:[variants[k][n]['prediction'] for n in train] for k in order},
            [baseline[n]['truth'] for n in train],[baseline[n]['prediction'] for n in train],order)
        records[assess]={'fit_fold':fit,'assessment_fold':assess,'fit_n':len(train),
                         'assessment_n':len(folds[assess]),**detail}
        for name in folds[assess]:selected[name]=choice
    return selected,records


def candidate_gate(rows,baseline):
    y=np.array([r['truth'] for r in rows],dtype=float)
    p=np.array([r['prediction'] for r in rows],dtype=float)
    b=np.array([baseline[r['image']]['prediction'] for r in rows],dtype=float)
    if not len(y) or not (y>=300).any():return False
    return bool(np.abs(p-y).mean()<=1.02*np.abs(b-y).mean()
                and np.mean((p-y)**2)<np.mean((b-y)**2)
                and np.abs(p[y>=300]-y[y>=300]).mean()<np.abs(b[y>=300]-y[y>=300]).mean())
