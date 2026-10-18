"""M2/M3 inference policies; ground truth is deliberately absent from this API."""
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.ops import nms
from .geco_adapter import QueryAdapter, postprocess
from .geometry import make_tiles, restore_tile_boxes, select_scale_candidates


class ExperimentEngine:
    def __init__(self, model, config):
        self.adapter = QueryAdapter(model)
        self.config = config
        self.normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])

    def prepare(self, image, exemplar_boxes):
        from utils.data import resize_and_pad
        image, boxes, scale = resize_and_pad(image, exemplar_boxes)
        return self.normalize(image).unsqueeze(0).cuda(), boxes.unsqueeze(0).cuda(), float(scale)

    def check_memory(self):
        if torch.cuda.max_memory_allocated() / 2**20 > self.config['stop']['memory_allocated_mib']:
            raise RuntimeError('Predeclared memory budget exceeded')

    def run(self, image, exemplars, variant):
        height, width = image.shape[-2:]
        x, original_boxes, scale = self.prepare(image, exemplars)
        output, features, appearance = self.adapter.forward(x, original_boxes)
        detected, scores = postprocess(output, int(width * scale), int(height * scale))
        base = detected.cpu().numpy() * (1024 / scale)
        boxes = base
        meta = {'base_count': len(base), 'm2_triggered': False, 'm3_triggered': False,
                'backbone_passes': 1, 'decoder_passes': 1, 'pseudo_exemplars': []}
        prompt_boxes = original_boxes
        self.check_memory()

        if variant in ('m2', 'm2-repeat', 'm2-m3') and len(detected):
            candidate_boxes = (detected * 1024).unsqueeze(0)
            candidate_appearance = self.adapter.appearance(features, candidate_boxes)
            similarities = torch.stack([
                F.normalize(c[0], dim=-1) @ F.normalize(a[0], dim=-1).T
                for c, a in zip(candidate_appearance, appearance)
            ]).mean(0).max(1).values
            relative = scores / output['box_v'].max().clamp_min(1e-12)
            selected = select_scale_candidates(candidate_boxes[0].cpu().numpy(),
                                               original_boxes[0].cpu().numpy(), relative.cpu().numpy(),
                                               similarities.cpu().numpy(), self.config['m2'])
            meta['selected_candidate_count'] = len(selected)
            for i in selected:
                meta['pseudo_exemplars'].append({'box_xyxy': base[i].tolist(),
                    'cosine': float(similarities[i]), 'relative_objectness': float(relative[i])})
            if selected:
                if variant == 'm2-repeat':
                    ids = torch.arange(len(selected), device=x.device) % original_boxes.shape[1]
                    extra = original_boxes[:, ids]
                else:
                    extra = candidate_boxes[:, selected]
                prompt_boxes = torch.cat([original_boxes, extra], dim=1)
                appearance = self.adapter.appearance(features, prompt_boxes)
                second = self.adapter.decode(features, appearance, prompt_boxes[:, :, 2:] - prompt_boxes[:, :, :2])
                detected2, _ = postprocess(second, int(width * scale), int(height * scale))
                boxes = detected2.cpu().numpy() * (1024 / scale)
                meta.update(m2_triggered=True, decoder_passes=2, second_count=len(boxes))
                del second, detected2
                self.check_memory()
            del candidate_appearance, candidate_boxes, similarities, relative
        del output, detected, scores

        cfg = self.config['m3']
        median_side = float(torch.sqrt(torch.prod(original_boxes[0, :, 2:] - original_boxes[0, :, :2], dim=1)).median())
        triggered = median_side <= cfg['trigger_median_exemplar_side'] or len(boxes) >= cfg['trigger_predicted_count']
        meta['median_exemplar_side_model_pixels'] = median_side
        if variant in ('m3', 'm3-repeat', 'm2-m3') and triggered:
            meta['m3_triggered'] = True
            if variant == 'm3-repeat':
                for _ in range(4):
                    repeated, f, a = self.adapter.forward(x, original_boxes)
                    detected3, _ = postprocess(repeated, int(width * scale), int(height * scale))
                    boxes = detected3.cpu().numpy() * (1024 / scale)
                    if not np.array_equal(boxes, base):
                        raise RuntimeError('Repeated baseline changed unexpectedly')
                    del repeated, f, a, detected3
                    self.check_memory()
                meta['backbone_passes'] += 4
                meta['decoder_passes'] += 4
            else:
                # Only the small appearance tensors survive across tiles.
                # No requirement that exemplar coordinates lie within a crop.
                del features, x
                tile_boxes, tile_scores, details = [], [], []
                sizes_original = (prompt_boxes[:, :, 2:] - prompt_boxes[:, :, :2]) / scale
                for tile in make_tiles(width, height, cfg['halo_fraction_of_full_image']):
                    x0, y0, x1, y1 = tile['crop']
                    crop = image[:, y0:y1, x0:x1]
                    tile_scale = min(1024 / max(y1-y0, x1-x0), 80 / float(sizes_original.mean()))
                    resized = F.interpolate(crop.unsqueeze(0), scale_factor=tile_scale,
                                            mode='bilinear', align_corners=False)
                    ch, cw = resized.shape[-2:]
                    padded = F.pad(resized, (0, 1024-cw, 0, 1024-ch))[0]
                    tile_x = self.normalize(padded).unsqueeze(0).cuda()
                    tile_features = self.adapter.encode(tile_x)
                    tile_output = self.adapter.decode(tile_features, appearance, sizes_original * tile_scale)
                    td, ts = postprocess(tile_output, cw, ch)
                    restored, keep = restore_tile_boxes(td.cpu().numpy() * 1024, tile_scale, tile)
                    normalized = (ts / tile_output['box_v'].max().clamp_min(1e-12)).cpu().numpy() if len(ts) else np.zeros(0)
                    tile_boxes.append(restored[keep]); tile_scores.append(normalized[keep])
                    details.append({**tile, 'scale': tile_scale, 'raw_count': len(td), 'owned_count': int(keep.sum())})
                    del tile_x, tile_features, tile_output, td, ts, padded, resized
                    self.check_memory()
                boxes = np.concatenate(tile_boxes, axis=0)
                all_scores = np.concatenate(tile_scores, axis=0)
                before_nms = len(boxes)
                if len(boxes):
                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
                    keep = nms(torch.from_numpy(boxes), torch.from_numpy(all_scores).to(torch.float32), cfg['merge_nms_iou']).numpy()
                    boxes = boxes[keep]
                meta.update(tiles=details, pre_merge_count=before_nms,
                            cross_tile_removed=before_nms-len(boxes))
                meta['backbone_passes'] += 4
                meta['decoder_passes'] += 4
        return boxes, base, meta
