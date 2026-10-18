"""Registered CPU-only Poisson calibration and strictly nested class selection."""
import numpy as np
from scipy.optimize import minimize


def edges(height, width):
    if min(height, width) < 8:
        raise ValueError('Image smaller than the fixed grid')
    return np.linspace(0, height, 9, dtype=int), np.linspace(0, width, 9, dtype=int)


def masses(density):
    d = np.asarray(density, dtype=np.float64)
    if d.ndim != 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('Invalid density')
    ys, xs = edges(*d.shape)
    return np.array([[d[ys[j]:ys[j+1], xs[i]:xs[i+1]].sum()
                      for i in range(8)] for j in range(8)])


def point_grid(points, height, width):
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ys, xs = edges(height, width)
    if not np.isfinite(p).all() or (p < 0).any() or (p[:, 0] >= width).any() or (p[:, 1] >= height).any():
        raise ValueError('Invalid point coordinates')
    ids = (np.searchsorted(ys, p[:, 1], side='right')-1)*8 + np.searchsorted(xs, p[:, 0], side='right')-1
    return ids, np.bincount(ids, minlength=64).reshape(8, 8)


def features(mass, boxes, exemplars, height, width, base_scale):
    d = np.asarray(mass, dtype=np.float64).reshape(8, 8)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    e = np.asarray(exemplars, dtype=np.float64).reshape(-1, 4)
    if not all(np.isfinite(x).all() for x in (d, b, e)) or (d < 0).any() or len(e) != 3:
        raise ValueError('Invalid calibration inputs')
    if not np.isfinite(base_scale) or base_scale <= 0 or (e[:, 2:] <= e[:, :2]).any() or (b[:, 2:] < b[:, :2]).any():
        raise ValueError('Invalid scale or boxes')
    exemplar_side = float(np.median((e[:, 2:]-e[:, :2]).mean(axis=1)))
    ids, counts = point_grid((b[:, :2]+b[:, 2:])/2, height, width)
    sums = np.bincount(ids, weights=(b[:, 2:]-b[:, :2]).mean(axis=1), minlength=64).reshape(8, 8)
    mean_side = np.divide(sums, counts, out=np.full((8, 8), exemplar_side), where=counts > 0)
    if (mean_side <= 0).any():
        raise ValueError('Log feature has nonpositive box side')
    ys, xs = edges(height, width)
    area = np.diff(ys)[:, None]*np.diff(xs)[None, :]*base_scale**2
    x = np.stack((np.log1p(d), np.full((8, 8), np.log(exemplar_side)), np.log(mean_side), d/area), axis=-1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite calibration features')
    return x


def fitting_arrays(records):
    if not records:
        raise ValueError('Empty fitting fold')
    x = np.stack([r['features'] for r in records]).reshape(-1, 4).astype(np.float64)
    d = np.stack([r['mass8'] for r in records]).reshape(-1).astype(np.float64)
    y = np.stack([r['truth8'] for r in records]).reshape(-1).astype(np.float64)
    if not all(np.isfinite(a).all() for a in (x, d, y)) or (d < 0).any() or (y < 0).any():
        raise ValueError('Invalid fitting arrays')
    valid = d > 0
    if not valid.any():
        raise ValueError('No positive-mass cells in fitting fold')
    return x[valid], d[valid], y[valid]


def poisson_objective(beta, design, d, y, l2):
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        eta = np.log(d)+design@beta
        mu = np.exp(eta)
        loss = float(np.mean(mu-y*eta)+l2*np.sum(beta[1:]**2))
        gradient = design.T@(mu-y)/len(y)
        gradient[1:] += 2*l2*beta[1:]
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite Poisson objective')
    return loss, gradient


def fit(records, kind, config):
    x, d, y = fitting_arrays(records)
    common = {'kind':kind, 'fit_images':[r['image'] for r in records],
              'positive_mass_cells':len(d), 'zero_mass_cells_excluded':len(records)*64-len(d),
              'gain_bounds':list(config['gain_bounds'])}
    if kind == 'global':
        total = float(y.sum())
        intercept = float(np.log(total/d.sum())) if total > 0 else None
        return {**common, 'intercept':intercept, 'zero_target_boundary':total == 0,
                'target_sum':total, 'mass_sum':float(d.sum()), 'converged':True}
    if kind != 'regional':
        raise ValueError('Unregistered calibration kind')
    mean, std = x.mean(axis=0), x.std(axis=0)
    constant = np.all(x == x[0], axis=0)
    std[constant | (std == 0)] = 1
    design = np.column_stack((np.ones(len(x)), (x-mean)/std))
    result = minimize(poisson_objective, np.zeros(5, dtype=np.float64),
                      args=(design, d, y, config['l2']), jac=True, method='L-BFGS-B',
                      options={'maxiter':config['maxiter'], 'ftol':config['ftol'], 'gtol':config['gtol']})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError('Registered Poisson optimizer did not converge')
    objective, gradient = poisson_objective(result.x, design, d, y, config['l2'])
    return {**common, 'mean':mean.tolist(), 'std':std.tolist(), 'constant_features':constant.tolist(),
            'coefficients':result.x.tolist(), 'objective':objective,
            'gradient_max_abs':float(np.abs(gradient).max()), 'iterations':int(result.nit),
            'optimizer_message':str(result.message), 'converged':True}


def gain(feature_array, model):
    x = np.asarray(feature_array, dtype=np.float64)
    if x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Invalid prediction features')
    lo, hi = model['gain_bounds']
    if (lo, hi) != (.5, 2.):
        raise ValueError('Unregistered gain bounds')
    if model['kind'] == 'global':
        if model['zero_target_boundary']:
            return np.full(x.shape[:-1], lo, dtype=np.float64)
        log_gain = np.full(x.shape[:-1], model['intercept'], dtype=np.float64)
    else:
        beta = np.asarray(model['coefficients'])
        std = np.asarray(model['std'])
        if beta.shape != (5,) or (std <= 0).any():
            raise ValueError('Invalid fitted model')
        log_gain = beta[0]+((x-np.asarray(model['mean']))/std)@beta[1:]
    if not np.isfinite(log_gain).all():
        raise ValueError('Nonfinite correction')
    # Algebraically identical to clipping exp(log_gain), avoiding overflow only at prediction.
    return np.exp(np.clip(log_gain, np.log(lo), np.log(hi)))


def game_cells(mass, truth, level):
    side = 2**level
    if side not in (1, 2, 4, 8):
        raise ValueError('Unsupported grid')
    a = np.asarray(mass).reshape(side, 8//side, side, 8//side).sum(axis=(1, 3))
    b = np.asarray(truth).reshape(side, 8//side, side, 8//side).sum(axis=(1, 3))
    return float(np.abs(a-b).sum())


def predict(record, model):
    gains = gain(record['features'], model)
    mass = np.asarray(record['mass8'], dtype=np.float64)*gains
    if not np.isfinite(mass).all():
        raise ValueError('Nonfinite corrected mass')
    return {'image':record['image'], 'truth':record['truth'], 'prediction':float(mass.sum()),
            'mass8':mass.tolist(), 'gains8':gains.tolist(), 'game1':game_cells(mass, record['truth8'], 1),
            'game2':game_cells(mass, record['truth8'], 2), 'calibration':model['kind']}


def apply_map(density, gains):
    d = np.asarray(density, dtype=np.float64)
    g = np.asarray(gains, dtype=np.float64).reshape(8, 8)
    if not np.isfinite(g).all() or (g < .5).any() or (g > 2.).any():
        raise ValueError('Invalid regional gains')
    original = masses(d);ys, xs = edges(*d.shape);result = np.empty_like(d)
    for j in range(8):
        for i in range(8):
            result[ys[j]:ys[j+1], xs[i]:xs[i+1]] = d[ys[j]:ys[j+1], xs[i]:xs[i+1]]*g[j, i]
    if not np.allclose(masses(result), original*g, rtol=1e-12, atol=1e-8):
        raise ValueError('Corrected density mass mismatch')
    return result


def metrics(rows):
    if not rows:
        raise ValueError('Empty assessment')
    errors = np.array([r['prediction']-r['truth'] for r in rows], dtype=np.float64)
    if not np.isfinite(errors).all():
        raise ValueError('Nonfinite assessment')
    return {'n':len(rows), 'mae':float(np.abs(errors).mean()), 'rmse':float(np.sqrt(np.mean(errors**2)))}


def admitted(rows, baseline):
    dense = [r for r in rows if r['truth'] >= 300]
    if not dense:
        return False
    current = metrics(rows);base = metrics([baseline[r['image']] for r in rows])
    return bool(current['mae'] <= 1.02*base['mae'] and current['rmse'] < base['rmse']
                and metrics(dense)['mae'] < metrics([baseline[r['image']] for r in dense])['mae'])


def inner_folds(names, labels):
    if len(names) != len(set(names)):
        raise ValueError('Duplicate fitting images')
    classes = sorted({labels[n] for n in names})
    if len(classes) < 2:
        raise ValueError('Too few fitting classes for nested selection')
    left = set(classes[::2]);right = set(classes[1::2])
    return {'A':[n for n in names if labels[n] in left], 'B':[n for n in names if labels[n] in right]}


def choose_inner(names, records, labels, baseline, config):
    folds = inner_folds(names, labels)
    predicted = {k:{} for k in ('regional', 'global')};models = {}
    for fit_fold, assess_fold in (('A', 'B'), ('B', 'A')):
        models[assess_fold] = {}
        for kind in predicted:
            model = fit([records[n] for n in folds[fit_fold]], kind, config)
            models[assess_fold][kind] = model
            for n in folds[assess_fold]:
                predicted[kind][n] = predict(records[n], model)
    rows = {k:[v[n] for n in names] for k,v in predicted.items()}
    scores = {k:metrics(v) for k,v in rows.items()}
    choice = 'regional' if scores['regional']['rmse'] < scores['global']['rmse'] else 'global'
    return {'selected':choice, 'admitted':admitted(rows[choice], baseline), 'metrics':scores,
            'fit_images':list(names), 'inner_folds':folds, 'inner_models':models,
            'inner_predictions':rows, 'selection_scope':'fitting_classes_only'}


def crossfit(names, folds, records, labels, baseline, config):
    if set(folds['A']) & set(folds['B']) or set(folds['A']+folds['B']) != set(names):
        raise ValueError('Invalid outer image partition')
    if {labels[n] for n in folds['A']} & {labels[n] for n in folds['B']}:
        raise ValueError('Outer classes overlap')
    output = {k:{} for k in ('regional', 'global', 'selected', 'admitted')};models={};decisions={}
    for fit_fold, assess_fold in (('A', 'B'), ('B', 'A')):
        fit_names = folds[fit_fold]
        decisions[assess_fold] = choose_inner(fit_names, records, labels, baseline, config)
        models[assess_fold] = {k:fit([records[n] for n in fit_names], k, config) for k in ('regional','global')}
        choice = decisions[assess_fold]['selected']
        for n in folds[assess_fold]:
            for k in ('regional', 'global'):
                output[k][n] = {**predict(records[n], models[assess_fold][k]), 'assessment_fold':assess_fold}
            output['selected'][n] = output[choice][n]
            output['admitted'][n] = output[choice][n] if decisions[assess_fold]['admitted'] else {
                'image':n, 'truth':records[n]['truth'], 'prediction':records[n]['prediction'],
                'mass8':records[n]['mass8'], 'gains8':np.ones((8,8)).tolist(),
                'game1':records[n]['game1'], 'game2':records[n]['game2'],
                'calibration':'none', 'assessment_fold':assess_fold}
    final_decision = choose_inner(names, records, labels, baseline, config)
    final_models = {k:fit([records[n] for n in names], k, config) for k in ('regional', 'global')}
    return {'rows':{k:[v[n] for n in names] for k,v in output.items()}, 'outer_models':models,
            'outer_decisions':decisions, 'final_decision':final_decision, 'final_models':final_models}
