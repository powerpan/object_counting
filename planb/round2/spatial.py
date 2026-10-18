"""Original-image density mass, fixed grids, and detection localization."""
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist
from torch.nn import functional as F


def restore_density(scaled_output, height, width, valid_height, valid_width):
    source = scaled_output.detach().double().cpu().reshape(512, 512)[:valid_height, :valid_width] / 100
    if not torch.isfinite(source).all() or (source < 0).any():
        raise ValueError('Invalid predicted density')
    mass = float(source.sum())
    restored = F.interpolate(source[None, None], size=(height, width), mode='bilinear',
                             align_corners=False)[0, 0].numpy()
    total = float(restored.sum())
    if total > 0:
        restored *= mass / total
    elif mass != 0:
        raise ValueError('Density restoration lost all mass')
    if abs(float(restored.sum()) - mass) > 1e-6:
        raise ValueError('Density restoration changed count')
    return restored


def grid_edges(height, width, side=8):
    if min(height, width) < side:
        raise ValueError('Image too small for the registered grid')
    return np.linspace(0, height, side+1, dtype=int), np.linspace(0, width, side+1, dtype=int)


def cell_mass(density, side=8):
    d = np.asarray(density, dtype=np.float64)
    ys, xs = grid_edges(*d.shape, side)
    return np.array([[d[ys[j]:ys[j+1], xs[i]:xs[i+1]].sum()
                      for i in range(side)] for j in range(side)])


def point_cells(points, height, width, side=8):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ys, xs = grid_edges(height, width, side)
    if len(points) and not ((points >= 0).all() and (points[:, 0] < width).all()
                            and (points[:, 1] < height).all()):
        raise ValueError('Points outside original image')
    col = np.searchsorted(xs, points[:, 0], side='right') - 1
    row = np.searchsorted(ys, points[:, 1], side='right') - 1
    ids = row * side + col
    counts = np.bincount(ids, minlength=side*side).reshape(side, side)
    return ids, counts


def grid_features(boxes, scores, exemplars, density, side=8):
    height, width = density.shape
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    exemplars = np.asarray(exemplars, dtype=np.float64).reshape(-1, 4)
    if len(scores) != len(boxes) or not np.isfinite(scores).all():
        raise ValueError('Invalid detection objectness')
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    ids, count = point_cells(centers, height, width, side)
    denominator = np.maximum(count.reshape(-1), 1)
    mean_scores = np.bincount(ids, weights=scores, minlength=side*side) / denominator
    box_sides = (boxes[:, 2:] - boxes[:, :2]).mean(axis=1)
    mean_sides = np.bincount(ids, weights=box_sides, minlength=side*side) / denominator
    exemplar_side = float(np.median((exemplars[:, 2:] - exemplars[:, :2]).mean(axis=1)))
    ys, xs = grid_edges(height, width, side)
    area = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    features = np.stack([count, cell_mass(density, side), mean_scores.reshape(side, side),
                         mean_sides.reshape(side, side), np.full((side, side), exemplar_side),
                         count / area], axis=-1)
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite fusion features')
    return features


def game_from_cells(mass, truth, level):
    mass = np.asarray(mass, dtype=np.float64).reshape(8, 8)
    truth = np.asarray(truth, dtype=np.float64).reshape(8, 8)
    side = 2 ** level
    if side not in (1, 2, 4, 8):
        raise ValueError('Unsupported nested grid level')
    def coarsen(x):
        return x.reshape(side, 8//side, side, 8//side).sum(axis=(1, 3))
    return float(np.abs(coarsen(mass) - coarsen(truth)).sum())


def localization(boxes, points, exemplars):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    exemplars = np.asarray(exemplars, dtype=np.float64).reshape(-1, 4)
    radius = .5 * float((exemplars[:, 2:] - exemplars[:, :2]).mean())
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Invalid localization radius')
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    if len(centers) and len(points):
        graph = csr_matrix(cdist(centers, points) <= radius)
        match = maximum_bipartite_matching(graph, perm_type='column')
        tp = int((match >= 0).sum())
    else:
        tp = 0
    fp, fn = len(centers)-tp, len(points)-tp
    return {'tp': tp, 'fp': fp, 'fn': fn, 'f1': 2*tp/max(2*tp+fp+fn, 1),
            'radius_original_pixels': radius}


def fused_map(m1, m4, weights):
    if m1.shape != m4.shape:
        raise ValueError('Density shapes disagree')
    weights = np.asarray(weights, dtype=np.float64).reshape(8, 8)
    if not np.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError('Invalid mixture weights')
    result = np.empty_like(m1, dtype=np.float64)
    ys, xs = grid_edges(*m1.shape)
    for j in range(8):
        for i in range(8):
            sl = np.s_[ys[j]:ys[j+1], xs[i]:xs[i+1]]
            result[sl] = weights[j, i]*m1[sl] + (1-weights[j, i])*m4[sl]
    expected = weights*cell_mass(m1) + (1-weights)*cell_mass(m4)
    if abs(float(result.sum()) - float(expected.sum())) > 1e-6:
        raise ValueError('Fusion grid count differs from density integral')
    return result
