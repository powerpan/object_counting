"""Frozen GeCo2 query interface with explicit, reusable exemplar appearance.

The math mirrors pinned GECO2/models/counter_infer.py. No upstream files or
weights are changed. A real-GPU equivalence gate precedes experiments.
"""
import torch
from torchvision.ops import roi_align, nms


class QueryAdapter:
    def __init__(self, model):
        self.model = model

    def encode(self, image):
        return self.model.backbone(image)

    def appearance(self, features, boxes):
        # Experiments are sequential, with one image and arbitrary exemplars.
        if boxes.ndim != 3 or boxes.shape[0] != 1 or boxes.shape[-1] != 4:
            raise ValueError('Expected one image with N exemplar boxes')
        rois = torch.cat([torch.zeros((boxes.shape[1], 1), device=boxes.device), boxes[0]], dim=1)
        maps = [features['vision_features'], *features['backbone_fpn'][:2]]
        return [roi_align(f, rois, output_size=1, spatial_scale=f.shape[-1] / 1024,
                          aligned=True).permute(0, 2, 3, 1).reshape(1, boxes.shape[1], -1)
                for f in maps]

    def decode(self, features, appearance, sizes):
        from utils.box_ops import boxes_with_scores
        m = self.model
        shape = m.shape_or_objectness(sizes).reshape(1, -1, m.emb_dim)
        prototypes = [torch.cat([a, shape], dim=1) for a in appearance]
        adapted, _ = m.adapt_features(
            image_embeddings=features['vision_features'],
            image_pe=m.sam_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototypes[0], hq_features=features['backbone_fpn'],
            hq_prototypes=prototypes[1:], hq_pos=features['vision_pos_enc'])
        bs, c, w, h = adapted.shape
        flat = adapted.view(bs, m.emb_dim, -1).permute(0, 2, 1)
        centerness = m.class_embed(flat).view(bs, w, h, 1).permute(0, 3, 1, 2)
        coords = m.bbox_embed(flat).sigmoid().view(bs, w, h, 4).permute(0, 3, 1, 2)
        outputs, _ = boxes_with_scores(centerness, coords, sort=False, validate=True)
        masks, ious, corrected = m.sam_mask(features, outputs)
        for i, out in enumerate(outputs):
            out['scores'] = ious[i]
            out['pred_boxes'] = corrected[i].to(out['pred_boxes'].device).unsqueeze(0) / 1024
        del masks
        return outputs[0]

    def forward(self, image, boxes):
        features = self.encode(image)
        appearance = self.appearance(features, boxes)
        output = self.decode(features, appearance, boxes[:, :, 2:] - boxes[:, :, :2])
        return output, features, appearance


def postprocess(output, width, height):
    raw, v = output['pred_boxes'].reshape(-1, 4), output['box_v'].reshape(-1)
    if not torch.isfinite(raw).all() or not torch.isfinite(v).all():
        raise ValueError('Nonfinite output')
    if not len(raw):
        return raw, v
    selected = v > v.max() * .11
    boxes, scores = raw[selected], v[selected]
    keep = nms(boxes, scores, .5)
    boxes, scores = boxes[keep].clamp(0, 1), scores[keep]
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    valid = (centers[:, 0] * 1024 < width) & (centers[:, 1] * 1024 < height)
    return boxes[valid], scores[valid]
