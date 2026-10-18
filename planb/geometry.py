"""Pure geometry for preregistered pseudo-exemplar and tile policies."""
import math
import numpy as np


def select_scale_candidates(boxes, exemplar_boxes, relative_scores, similarities, cfg):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    original = np.asarray(exemplar_boxes, dtype=float).reshape(-1, 4)
    sides = boxes[:, 2:] - boxes[:, :2]
    orig_sides = original[:, 2:] - original[:, :2]
    if not (orig_sides > 0).all():
        raise ValueError('Degenerate original exemplar')
    scales = np.sqrt(np.prod(np.maximum(sides, 0), axis=1))
    orig_scales = np.sqrt(np.prod(orig_sides, axis=1))
    ratios = np.maximum(scales[:, None] / orig_scales,
                        orig_scales / np.maximum(scales[:, None], 1e-12))
    relative_scores = np.asarray(relative_scores)
    similarities = np.asarray(similarities)
    valid = ((sides >= cfg['min_side_in_model_pixels']).all(1)
             & (ratios.min(1) >= cfg['min_scale_ratio'])
             & (relative_scores >= cfg['min_relative_objectness'])
             & (similarities >= cfg['min_mean_cosine'])
             & np.isfinite(relative_scores) & np.isfinite(similarities))
    order = np.argsort(-(relative_scores * similarities), kind='stable')
    selected, bins = [], set()
    for i in order:
        if not valid[i]:
            continue
        scale_bin = math.floor(math.log(scales[i] / np.median(orig_scales), cfg['min_scale_ratio']))
        if scale_bin in bins:
            continue
        bins.add(scale_bin)
        selected.append(int(i))
        if len(selected) == cfg['max_extra_exemplars']:
            break
    return selected


def make_tiles(width, height, halo_fraction=.1):
    if width < 2 or height < 2 or not 0 <= halo_fraction < .5:
        raise ValueError('Invalid tile geometry')
    xs, ys = [0, width // 2, width], [0, height // 2, height]
    hx, hy = math.ceil(width * halo_fraction), math.ceil(height * halo_fraction)
    tiles = []
    for j in range(2):
        for i in range(2):
            core = [xs[i], ys[j], xs[i+1], ys[j+1]]
            crop = [max(0, core[0]-hx), max(0, core[1]-hy),
                    min(width, core[2]+hx), min(height, core[3]+hy)]
            tiles.append({'core': core, 'crop': crop})
    return tiles


def restore_tile_boxes(boxes_in_model_pixels, scale, tile):
    boxes = np.asarray(boxes_in_model_pixels, dtype=np.float32).reshape(-1, 4)
    if scale <= 0 or not np.isfinite(boxes).all():
        raise ValueError('Invalid tile output')
    boxes = boxes / scale + np.array(tile['crop'][:2] * 2, dtype=np.float32)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    x0, y0, x1, y1 = tile['core']
    keep = ((centers[:, 0] >= x0) & (centers[:, 0] < x1)
            & (centers[:, 1] >= y0) & (centers[:, 1] < y1))
    return boxes, keep
