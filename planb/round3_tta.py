"""Frozen-model TTA and count-conserving M1 projection; no truth enters this API."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from planb.inference_variants import ExperimentEngine
from planb.round2.adapter import Round2Adapter
from planb.round3_m3 import snapshot, require_equal
from planb.geco_adapter import postprocess
from planb.density import boxes_to_density

VIEWS = {
    'T2': [(1., False), (1., True)],
    'T4': [(1., False), (1., True), (.8, False), (.8, True)],
    'tta-repeat': [(1., False)] * 4,
}


def flip_boxes(boxes, width):
    """Continuous xyxy edges, using the same horizontal flip as the RGB image."""
    result = boxes.clone() if isinstance(boxes, torch.Tensor) else np.array(boxes, copy=True)
    result[..., 0] = width - boxes[..., 2]
    result[..., 2] = width - boxes[..., 0]
    return result


def transformed_input(image, exemplars, base_scale, factor, flip):
    if factor not in (1., .8) or not math.isfinite(base_scale) or base_scale <= 0:
        raise ValueError('Unregistered TTA scale')
    if image.ndim != 3 or image.shape[0] != 3 or exemplars.shape != (3, 4):
        raise ValueError('Invalid RGB/exemplar shape')
    width = image.shape[-1]
    source = torch.flip(image, [-1]) if flip else image
    prompts = flip_boxes(exemplars, width) if flip else exemplars
    scale = base_scale * factor
    resized = F.interpolate(source[None], scale_factor=scale, mode='bilinear', align_corners=False)
    h, w = resized.shape[-2:]
    if min(h, w) < 1 or max(h, w) > 1024:
        raise ValueError('Invalid registered TTA canvas')
    padded = F.pad(resized, (0, 1024-w, 0, 1024-h))[0]
    prompts = prompts * torch.tensor([scale]*4, dtype=prompts.dtype, device=prompts.device)
    return padded, prompts, scale, (h, w)


def project_boxes(normalized, scale, width, height, flip):
    # Preserve B0's float32 arithmetic before the inverse view transform.
    original = np.asarray(normalized, dtype=np.float32).reshape(-1, 4) * (1024/scale)
    if not flip:
        return original
    original = flip_boxes(original.astype(np.float64), width)
    if not np.isfinite(original).all():
        raise ValueError('Nonfinite inverse projection')
    # A zero-width box centred exactly at flipped x=0 maps to x=W.
    # Preserve it with a bounded coordinate-tail correction, not by deleting it.
    centers = (original[:, :2] + original[:, 2:]) / 2
    bounds = np.array([width, height], dtype=np.float64)
    tol = 8*np.spacing(bounds)
    if ((centers < -tol) | (centers > bounds+tol)).any():
        raise ValueError('Inverse projection centre outside image')
    target = np.clip(centers, 0., np.nextafter(bounds, 0.))
    shift = target-centers
    original[:, :2] += shift
    original[:, 2:] += shift
    return original


def mean_density(maps, counts, repeat=False):
    if not maps or len(maps) != len(counts):
        raise ValueError('Missing TTA views')
    shape = np.asarray(maps[0]).shape
    for density, count in zip(maps, counts):
        density = np.asarray(density)
        if density.shape != shape or not np.isfinite(density).all() or (density < 0).any():
            raise ValueError('Invalid view density')
        if not math.isfinite(count) or count < 0 or abs(float(density.sum())-count) > 1e-6:
            raise ValueError('View mass does not equal count')
    if repeat:
        if any(not np.array_equal(maps[0], d) for d in maps[1:]) or len(set(counts)) != 1:
            raise ValueError('Repeated M1 differs from B0')
        result = maps[0] + np.mean([d-maps[0] for d in maps], axis=0, dtype=np.float64)
    else:
        result = np.mean(maps, axis=0, dtype=np.float64)
    count = float(np.mean(counts, dtype=np.float64))
    if not np.isfinite(result).all() or (result < 0).any() or abs(float(result.sum())-count) > 1e-6:
        raise ValueError('Average density changed count')
    return result, count


class TTAEngine(ExperimentEngine):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.adapter = Round2Adapter(model)

    def run(self, image, exemplars, variant):
        if variant not in VIEWS:
            raise ValueError('Unknown TTA variant')
        height, width = image.shape[-2:]
        base_x, base_prompts, base_scale = self.prepare(image, exemplars)
        maps, counts, details = [], [], []
        initial = None
        for index, (factor, flip) in enumerate(VIEWS[variant]):
            if factor == 1. and not flip:
                x, prompts, scale = base_x, base_prompts, base_scale
                valid = (int(height*scale), int(width*scale))
            else:
                padded, p, scale, valid = transformed_input(image, exemplars, base_scale, factor, flip)
                x, prompts = self.normalize(padded)[None].cuda(), p[None].cuda()
                del padded, p
            output, features, appearance = self.adapter.forward(x, prompts)
            if not all(torch.isfinite(output[k]).all() for k in ('pred_boxes', 'box_v', 'scores')):
                raise ValueError('Nonfinite TTA output')
            raw = snapshot(output)
            detected, objectness = postprocess(output, valid[1], valid[0])
            boxes = project_boxes(detected.cpu().numpy(), scale, width, height, flip)
            scores = objectness.cpu().numpy()
            if initial is None:
                initial, first_boxes, first_scores = raw, boxes.copy(), scores.copy()
            if variant == 'tta-repeat':
                require_equal(initial, raw)
                if not np.array_equal(boxes, first_boxes) or not np.array_equal(scores, first_scores):
                    raise ValueError('Repeated detections differ from B0')
            density = boxes_to_density(boxes, height, width)
            maps.append(density); counts.append(len(boxes))
            details.append({'scale_factor':factor, 'flipped':flip, 'effective_scale':scale,
                            'valid_height':valid[0], 'valid_width':valid[1],
                            'count':len(boxes), 'boxes_original_xyxy':boxes.tolist(),
                            'boxes_normalized_xyxy':detected.cpu().numpy().tolist(),
                            'objectness':scores.tolist(), 'refinement':dict(self.adapter.last_refinement)})
            self.check_memory()
            del output, features, appearance, detected, objectness, raw, x, prompts
        density, count = mean_density(maps, counts, repeat=variant=='tta-repeat')
        return density, count, first_boxes, first_scores, {
            'base_scale':base_scale, 'views':details, 'view_count_std':float(np.std(counts, ddof=0)),
            'repeat_bitwise_equal':True if variant=='tta-repeat' else None,
            'localization_and_box_features':'original_view_only',
            'density_projection':'inverse_box_geometry_then_original_grid_M1',
        }
