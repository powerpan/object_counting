"""Fold-local width selection and sigmoid blending; no validation policy search."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def select_candidate(predictions, truth, baseline, order):
    truth, baseline = np.asarray(truth), np.asarray(baseline)
    base_rmse = float(np.sqrt(np.mean((baseline-truth)**2)))
    metrics = {}
    for key in order:
        errors = np.asarray(predictions[key]) - truth
        metrics[key] = {'mae': float(np.abs(errors).mean()),
                        'rmse': float(np.sqrt(np.mean(errors**2)))}
    eligible = [k for k in order if metrics[k]['rmse'] <= base_rmse]
    keys = eligible if eligible else list(order)
    primary, secondary = ('mae', 'rmse') if eligible else ('rmse', 'mae')
    chosen = min(keys, key=lambda k: (metrics[k][primary], metrics[k][secondary], order.index(k)))
    return chosen, {'selected': chosen, 'rmse_eligible': eligible, 'metrics': metrics,
                    'baseline_rmse': base_rmse, 'selection_rule': primary+' then '+secondary+' then candidate order'}


def fit_gate(features, m1, m4, truth, config):
    x = np.asarray(features, dtype=np.float64).reshape(-1, 6)
    m1, m4, truth = [np.asarray(a, dtype=np.float64).reshape(-1) for a in (m1, m4, truth)]
    if not len(x) or len(x) != len(truth) or not all(np.isfinite(a).all() for a in (x, m1, m4, truth)):
        raise ValueError('Invalid fitting arrays')
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1
    z = np.column_stack([np.ones(len(x)), (x-mean)/std])
    delta = m1-m4
    def objective(beta):
        w = expit(z @ beta)
        error = m4+w*delta-truth
        loss = float(np.mean(error**2) + config['l2']*np.sum(beta[1:]**2))
        gradient = 2*z.T@(error*delta*w*(1-w))/len(x)
        gradient[1:] += 2*config['l2']*beta[1:]
        return loss, gradient
    result = minimize(objective, np.zeros(7), jac=True, method='L-BFGS-B',
                      options={'maxiter': config['max_iterations'], 'ftol': config['ftol'], 'gtol': config['gtol']})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError('Registered fusion optimizer did not converge')
    return {'mean': mean.tolist(), 'std': std.tolist(), 'coefficients': result.x.tolist(),
            'objective': float(result.fun), 'iterations': int(result.nit), 'converged': True,
            'fitting_images': len(x)//64, 'fitting_cells': len(x)}


def gate_weights(features, model):
    x = np.asarray(features, dtype=np.float64)
    z = (x-np.asarray(model['mean']))/np.asarray(model['std'])
    beta = np.asarray(model['coefficients'])
    weights = expit(beta[0] + z@beta[1:])
    if not np.isfinite(weights).all():
        raise ValueError('Nonfinite blend weights')
    return weights
