"""Opt-in equal-weight count/density ensemble; no labels or fitted parameters."""
import numpy as np


def half_count(b0_count, final_count):
    values = np.asarray([b0_count, final_count], dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Counts must be finite nonnegative scalars')
    return float(values[0] / 2 + values[1] / 2)


def half_blend(b0_count, final_count, b0_density, final_density):
    """Return a fractional count and its map in the original image coordinates.

    Inputs can be taken directly from FinalEngine.run_pair. The frozen engine,
    detection boxes and its routing are not changed. This ensemble is a density
    estimator; it does not invent a merged set of integer-count detections.
    """
    count = half_count(b0_count, final_count)
    a, b = (np.asarray(x, dtype=np.float64) for x in (b0_density, final_density))
    if a.ndim != 2 or a.shape != b.shape or not a.size:
        raise ValueError('Density maps must share a nonempty two-dimensional shape')
    for array, expected in ((a, b0_count), (b, final_count)):
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError('Invalid density values')
        if abs(float(array.sum()) - expected) > 1e-6:
            raise ValueError('Input density mass differs from its count')
    blended = a / 2 + b / 2
    if not np.isfinite(blended).all() or abs(float(blended.sum()) - count) > 1e-6:
        raise ValueError('Blended density mass differs from its count')
    return count, blended
