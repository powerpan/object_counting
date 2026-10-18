"""Registered M3-v2 geometry and inference. This API never accepts truth."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms
from planb.inference_variants import ExperimentEngine
from planb.round2.adapter import Round2Adapter
from planb.geco_adapter import postprocess
from planb.geometry import restore_tile_boxes


def tiles(width, height, grid, halo=.1):
    if grid not in (2, 3) or min(width, height) < grid or not 0 <= halo < .5:
        raise ValueError('Invalid registered tile geometry')
    xs = [i * width // grid for i in range(grid+1)]
    ys = [j * height // grid for j in range(grid+1)]
    hx, hy = math.ceil(width*halo), math.ceil(height*halo)
    return [{'core':[xs[i], ys[j], xs[i+1], ys[j+1]],
             'crop':[max(0,xs[i]-hx), max(0,ys[j]-hy),
                     min(width,xs[i+1]+hx), min(height,ys[j+1]+hy)]}
            for j in range(grid) for i in range(grid)]


def threshold_output(output, width, height, maximum, relative=.11, iou=.5):
    """Same B0 order/device arithmetic, with a supplied common maximum."""
    raw, v = output['pred_boxes'].reshape(-1,4), output['box_v'].reshape(-1)
    if len(raw) != len(v) or not all(torch.isfinite(x).all() for x in output.values()):
        raise ValueError('Invalid raw output')
    if not len(v):
        return raw, v
    if not math.isfinite(maximum):
        raise ValueError('Nonfinite objectness maximum')
    keep = v > torch.as_tensor(maximum, dtype=v.dtype, device=v.device)*relative
    boxes, scores = raw[keep], v[keep]
    keep = nms(boxes, scores, iou)
    boxes, scores = boxes[keep].clamp(0,1), scores[keep]
    centers = (boxes[:,:2]+boxes[:,2:])/2
    valid = (centers[:,0]*1024 < width) & (centers[:,1]*1024 < height)
    return boxes[valid], scores[valid]


def snapshot(output):
    return {key:output[key].detach().cpu().clone() for key in ('pred_boxes','box_v','scores')}


def require_equal(a, b):
    if any(not torch.equal(a[k].cpu(),b[k].cpu()) for k in ('pred_boxes','box_v','scores')):
        raise ValueError('Repeated raw model output differs from B0')


class M3Engine(ExperimentEngine):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.adapter = Round2Adapter(model)

    def run(self, image, exemplars, variant):
        if variant not in self.config['m3']['runs']:
            raise ValueError('Unregistered M3 variant')
        height,width = image.shape[-2:]
        x,prompts,scale = self.prepare(image,exemplars)
        output,features,appearance = self.adapter.forward(x,prompts)
        initial = snapshot(output)
        branch = dict(self.adapter.last_refinement)
        detected,objectness = postprocess(output,int(width*scale),int(height*scale))
        base = detected.cpu().numpy()*(1024/scale)
        boxes,scores = base,objectness.cpu().numpy()
        sizes = prompts[:,:,2:]-prompts[:,:,:2]
        median = float(torch.sqrt(sizes[0].prod(1)).median())
        cfg = self.config['m3']
        triggered = median <= cfg['trigger_median_exemplar_side'] or len(base) >= cfg['trigger_predicted_count']
        meta = {'base_count':len(base),'base_scale':scale,'base_refinement':branch,
                'triggered':triggered,'tiling_applied':False,'backbone_passes':1,'decoder_passes':1,
                'median_exemplar_side_model_pixels':median,'tiles':[],'cross_tile_removed':0}
        self.check_memory()
        del output,detected,objectness
        if variant.startswith('b0-refine-'):
            mode = variant.removeprefix('b0-refine-')
            forced = self.adapter.decode(features,appearance,sizes,refine=mode)
            d,v = postprocess(forced,int(width*scale),int(height*scale))
            boxes,scores = d.cpu().numpy()*(1024/scale),v.cpu().numpy()
            if (mode=='on' and branch['refinement_executed']) or (mode=='off' and not branch['refinement_executed']):
                require_equal(initial,forced)
                if not np.array_equal(boxes,base):
                    raise ValueError('Matching forced/auto SAM branches changed boxes')
            meta.update(forced_refinement=dict(self.adapter.last_refinement),decoder_passes=2)
            del forced,d,v
        elif triggered and '-repeat-' in variant:
            grid = int(variant[-1])
            del features,appearance
            for _ in range(grid*grid):
                repeated,f,a = self.adapter.forward(x,prompts)
                require_equal(initial,repeated)
                d,v = postprocess(repeated,int(width*scale),int(height*scale))
                if not np.array_equal(d.cpu().numpy()*(1024/scale),base):
                    raise ValueError('Repeat final boxes differ from B0')
                del repeated,f,a,d,v
                self.check_memory()
            meta.update(repeat_bitwise_equal=True,backbone_passes=1+grid*grid,decoder_passes=1+grid*grid)
        elif triggered:
            grid = 3 if variant=='m3-v2-g3' else 2
            local = variant=='m3-v2-branch-only'
            mode = 'on' if branch['refinement_executed'] else 'off'
            original_sizes = sizes/scale
            del features,x
            cached = []
            for tile in tiles(width,height,grid,cfg['halo_fraction_of_full_image']):
                x0,y0,x1,y1 = tile['crop']
                crop = image[:,y0:y1,x0:x1]
                ts = min(1024/max(y1-y0,x1-x0),80/float(original_sizes.mean()))
                resized = F.interpolate(crop[None],scale_factor=ts,mode='bilinear',align_corners=False)
                ch,cw = resized.shape[-2:]
                padded = F.pad(resized,(0,1024-cw,0,1024-ch))[0]
                tx = self.normalize(padded)[None].cuda()
                f = self.adapter.encode(tx)
                o = self.adapter.decode(f,appearance,original_sizes*ts,refine=mode)
                if not all(torch.isfinite(o[k]).all() for k in ('pred_boxes','box_v','scores')):
                    raise ValueError('Nonfinite tile output')
                detail = {**tile,'scale':ts,'valid_width':cw,'valid_height':ch,
                          **dict(self.adapter.last_refinement),
                          'maximum_objectness':float(o['box_v'].max()) if o['box_v'].numel() else None}
                cached.append((snapshot(o),detail))
                del tx,f,o,padded,resized
                self.check_memory()
            maxima = [d['maximum_objectness'] for _,d in cached if d['maximum_objectness'] is not None]
            maximum = max(maxima) if maxima else None
            all_boxes,all_raw,all_normalized = [],[],[]
            for raw,detail in cached:
                reference = detail['maximum_objectness'] if local else maximum
                gpu = {k:v.cuda() for k,v in raw.items()}
                d,v = threshold_output(gpu,detail['valid_width'],detail['valid_height'],
                                       reference if reference is not None else 0.,cfg['relative_threshold'],cfg['nms_iou'])
                restored,keep = restore_tile_boxes(d.cpu().numpy()*1024,detail['scale'],detail)
                score = v.cpu().numpy()
                # Match v1's positive-denominator guard; never divide an empty tile.
                norm = (v/max(reference,1e-12)).cpu().numpy() if len(v) else score
                all_boxes.append(restored[keep]);all_raw.append(score[keep]);all_normalized.append(norm[keep])
                detail.update(threshold_maximum=reference,post_threshold_count=len(d),owned_count=int(keep.sum()))
                meta['tiles'].append(detail)
                del gpu,d,v
            boxes,scores,rank = np.concatenate(all_boxes),np.concatenate(all_raw),np.concatenate(all_normalized)
            before = len(boxes)
            if before:
                boxes[:,[0,2]] = boxes[:,[0,2]].clip(0,width)
                boxes[:,[1,3]] = boxes[:,[1,3]].clip(0,height)
                keep = nms(torch.from_numpy(boxes),torch.from_numpy(rank).float(),cfg['nms_iou']).numpy()
                boxes,scores = boxes[keep],scores[keep]
            meta.update(tiling_applied=True,grid=grid,threshold_scope='tile' if local else 'all_tiles',
                        global_maximum_objectness=maximum,pre_merge_count=before,
                        cross_tile_removed=before-len(boxes),backbone_passes=1+grid*grid,decoder_passes=1+grid*grid)
        self.check_memory()
        return boxes,scores,base,meta
