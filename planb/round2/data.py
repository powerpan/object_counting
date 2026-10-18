"""Train-only augmentation and mass-preserving point targets."""
import json
import math
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from planb.density import points_for_game


def point_target(points, valid_height, valid_width, sigma, canvas=512):
    result = np.zeros((canvas, canvas), dtype=np.float64)
    if not 0 < valid_height <= canvas or not 0 < valid_width <= canvas or sigma < 1:
        raise ValueError('Invalid target geometry')
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite points')
    # After resampling truncates dimensions, preserve edge points in the last
    # valid cell. Original-coordinate validation happens before this step.
    bounds = np.array([valid_width, valid_height], dtype=np.float64)
    points = points.clip(0, np.nextafter(bounds, 0))
    for x, y in points:
        x0, x1 = max(0, math.floor(x-3*sigma)), min(valid_width, math.ceil(x+3*sigma)+1)
        y0, y1 = max(0, math.floor(y-3*sigma)), min(valid_height, math.ceil(y+3*sigma)+1)
        xx = (np.arange(x0, x1)-x)/sigma
        yy = (np.arange(y0, y1)-y)/sigma
        kernel = np.exp(-.5*(yy[:, None]**2 + xx[None, :]**2))
        result[y0:y1, x0:x1] += kernel/kernel.sum()
    if not np.isclose(result.sum(), len(points), atol=1e-8, rtol=0):
        raise ValueError('Point target lost mass')
    return result


class Inputs:
    def __init__(self, root, allowed_names):
        self.root = root
        raw = json.loads((root/'data/fsc147/annotations/annotation_FSC147_384.json').read_text())
        self.annotations = {name: raw[name] for name in allowed_names}
        self.normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])

    def prepare(self, name, *, seed=42, epoch=0, index=0, augment=False, with_target=True):
        with Image.open(self.root/'data/fsc147/images_384_VarV2'/name) as im:
            image = T.ToTensor()(im.convert('RGB'))
        height, width = image.shape[-2:]
        annotation = self.annotations[name]
        points = points_for_game(annotation['points'], height, width).copy() if with_target else None
        boxes = torch.tensor(annotation['box_examples_coordinates'], dtype=torch.float32)[:3, [0, 2], :].reshape(-1, 4)
        if boxes.shape!=(3,4) or not torch.isfinite(boxes).all() or not ((boxes[:,2:]-boxes[:,:2])>0).all():
            raise ValueError('Invalid official exemplar boxes')
        multiplier = 1.
        if augment:
            rng = np.random.default_rng(seed + 1000003*epoch + 1009*index)
            if rng.random() < .5:
                image = image.flip(-1)
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                if points is not None:points[:, 0] = width - points[:, 0]
            multiplier = float(rng.uniform(.8, 1.))
        # Match the source arithmetic (not merely an algebraically equivalent
        # expression), so augmentation-disabled gate remains bitwise equal.
        scaling = 1024 / max(height, width)
        scaled_boxes = boxes * scaling
        mean_side = ((scaled_boxes[:, 2]-scaled_boxes[:, 0]).mean() +
                     (scaled_boxes[:, 3]-scaled_boxes[:, 1]).mean()) / 2
        scaling = min(1., 80/mean_side.item()) * scaling * multiplier
        resized = F.interpolate(image.unsqueeze(0), scale_factor=scaling, mode='bilinear', align_corners=False)
        ch, cw = resized.shape[-2:]
        padded = F.pad(resized, (0, 1024-cw, 0, 1024-ch))[0]
        boxes = boxes * torch.tensor([scaling]*4)
        vh, vw = math.ceil(ch/2), math.ceil(cw/2)
        sigma = max(float((boxes[:, 2:]-boxes[:, :2]).mean())/8, 1.)
        target = point_target(points*scaling/2, vh, vw, sigma) if points is not None else None
        mask = torch.zeros(1, 512, 512)
        mask[:, :vh, :vw] = 1
        return {'image': self.normalize(padded).unsqueeze(0), 'boxes': boxes.unsqueeze(0),
                'target': torch.from_numpy(target.astype(np.float32)).unsqueeze(0) if target is not None else None, 'mask': mask,
                'count': len(points) if points is not None else None, 'name': name, 'height': height, 'width': width,
                'valid_height': vh, 'valid_width': vw, 'scale': scaling,
                'sigma': sigma, 'target_mass_float64': float(target.sum()) if target is not None else None}
