"""Paired external-set summaries. No fitted parameters or method selection."""
from collections import Counter
import numpy as np


def report(rows, replicates=10000, seed=20261109):
    if not rows or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate samples')
    truth = np.asarray([r['truth'] for r in rows], dtype=np.float64)
    predictions = {m: np.asarray([r[m]['prediction'] for r in rows], dtype=np.float64)
                   for m in ('b0', 'final')}
    if not np.isfinite(truth).all() or (truth < 0).any() or not np.equal(truth, np.floor(truth)).all():
        raise ValueError('Invalid ground-truth counts')
    if any(not np.isfinite(p).all() or (p < 0).any() for p in predictions.values()):
        raise ValueError('Invalid predictions')
    summary = {}
    for method, pred in predictions.items():
        error = pred - truth
        summary[method] = {'mae': float(np.abs(error).mean()),
                           'rmse': float(np.sqrt(np.square(error).mean()))}
        for field in ('game1', 'game2', 'inference_seconds'):
            values = np.asarray([r[method][field] for r in rows], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Invalid metric')
            summary[method][field + '_mean'] = float(values.mean())
        memory = np.asarray([r[method]['peak_allocated_mib'] for r in rows], dtype=np.float64)
        if not np.isfinite(memory).all() or (memory < 0).any():
            raise ValueError('Invalid memory record')
        summary[method]['peak_allocated_mib'] = float(memory.max())
    b = predictions['b0'] - truth
    f = predictions['final'] - truth
    units, inverse = np.unique([r['sampling_unit'] for r in rows], return_inverse=True)
    k = len(units)
    n = np.bincount(inverse)
    ae = np.bincount(inverse, weights=np.abs(f) - np.abs(b))
    se_b = np.bincount(inverse, weights=b*b)
    se_f = np.bincount(inverse, weights=f*f)
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError('Invalid bootstrap count')
    rng = np.random.default_rng(seed)
    dm, dr = [], []
    # Batches bound memory without altering RNG draws or the sampling unit.
    for start in range(0, replicates, 256):
        weights = rng.multinomial(k, np.full(k, 1/k), size=min(256, replicates-start))
        den = weights @ n
        dm.extend((weights @ ae / den).tolist())
        dr.extend((np.sqrt(weights @ se_f / den) - np.sqrt(weights @ se_b / den)).tolist())
    ci_mae = np.quantile(dm, [.025, .975]).tolist()
    ci_rmse = np.quantile(dr, [.025, .975]).tolist()
    paired = {'units': k, 'unit': 'source image SHA-256', 'replicates': replicates, 'seed': seed,
              'mae_delta': summary['final']['mae'] - summary['b0']['mae'],
              'rmse_delta': summary['final']['rmse'] - summary['b0']['rmse'],
              'mae_delta_95ci': ci_mae, 'rmse_delta_95ci': ci_rmse}
    return {'samples': len(rows), 'unique_source_images': k,
            'truth': {'total': int(truth.sum()), 'min': int(truth.min()), 'max': int(truth.max()),
                      'mean': float(truth.mean()), 'median': float(np.median(truth))},
            **summary, 'paired': paired,
            'routes': dict(Counter(r['route'] for r in rows)),
            'both_count_point_estimates_improved': paired['mae_delta'] < 0 and paired['rmse_delta'] < 0,
            'both_count_intervals_below_zero': ci_mae[1] < 0 and ci_rmse[1] < 0}
