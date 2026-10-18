"""Dense feature access and explicit SAM policy for the pinned GeCo2 model.

The dense arithmetic follows the MIT-licensed upstream counter_infer.py;
see THIRD_PARTY_NOTICES.md. Auto mode delegates to the unchanged SAM module.
"""
import torch
from planb.geco_adapter import QueryAdapter


class Round2Adapter(QueryAdapter):
    def dense(self, features, appearance, sizes):
        m = self.model
        shape = m.shape_or_objectness(sizes).reshape(1, -1, m.emb_dim)
        prototypes = [torch.cat([a, shape], dim=1) for a in appearance]
        adapted, _ = m.adapt_features(
            image_embeddings=features['vision_features'],
            image_pe=m.sam_prompt_encoder.get_dense_pe(),
            prototype_embeddings=prototypes[0], hq_features=features['backbone_fpn'],
            hq_prototypes=prototypes[1:], hq_pos=features['vision_pos_enc'])
        bs, c, h, w = adapted.shape
        if (bs, c, h, w) != (1, 256, 512, 512):
            raise ValueError('Unexpected dense feature shape')
        flat = adapted.view(bs, m.emb_dim, -1).permute(0, 2, 1)
        centerness = m.class_embed(flat).view(bs, h, w, 1).permute(0, 3, 1, 2)
        coords = m.bbox_embed(flat).sigmoid().view(bs, h, w, 4).permute(0, 3, 1, 2)
        return adapted, centerness, coords

    def refine(self, features, outputs, mode):
        if mode not in ('auto', 'on', 'off'):
            raise ValueError('Unknown refinement mode')
        out = outputs[0]
        n = out['pred_boxes'].shape[1]
        self.last_refinement = {'requested': mode, 'pre_refinement_candidates': n,
                               'refinement_executed': mode == 'on' or (mode == 'auto' and n <= 800)}
        if mode == 'off':
            out['scores'] = out['box_v']
            return out
        if mode == 'auto' or n <= 800:
            masks, ious, corrected = self.model.sam_mask(features, outputs)
            out['scores'] = ious[0]
            out['pred_boxes'] = corrected[0].to(out['pred_boxes'].device).unsqueeze(0) / 1024
            del masks
            return out
        # 800 is a multiple of the upstream 50-box internal batch. Only the
        # >800 early-exit policy changes; the mask arithmetic is unchanged.
        boxes, scores = [], []
        for start in range(0, n, 800):
            chunk = {'pred_boxes': out['pred_boxes'][:, start:start+800],
                     'box_v': out['box_v'][:, start:start+800]}
            masks, ious, corrected = self.model.sam_mask(features, [chunk])
            boxes.append(corrected[0].to(out['pred_boxes'].device).reshape(-1, 4))
            scores.append(ious[0].reshape(-1))
            del masks
        out['pred_boxes'] = torch.cat(boxes).unsqueeze(0) / 1024
        out['scores'] = torch.cat(scores).unsqueeze(0)
        return out

    def decode(self, features, appearance, sizes, refine='auto'):
        from utils.box_ops import boxes_with_scores
        adapted, centerness, coords = self.dense(features, appearance, sizes)
        outputs, _ = boxes_with_scores(centerness, coords, sort=False, validate=True)
        del adapted, centerness, coords
        return self.refine(features, outputs, refine)

    def density_features(self, image, boxes):
        features = self.encode(image)
        appearance = self.appearance(features, boxes)
        adapted, centerness, coords = self.dense(features, appearance, boxes[:, :, 2:] - boxes[:, :, :2])
        del coords
        return torch.cat([adapted, centerness], dim=1)
