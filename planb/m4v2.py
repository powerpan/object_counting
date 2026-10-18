"""Frozen M1 residual model and preregistered, testable M4-v2 rules."""
import math
import numpy as np
import torch
from torch import nn
from planb.density import boxes_to_density
from planb.geco_adapter import postprocess


class ResidualHead(nn.Module):
    def __init__(self, channels=(256,256,256,128,64), groups=32):
        super().__init__()
        blocks=[]; previous=258
        for width in channels:
            blocks.extend([nn.Conv2d(previous,width,3,padding=1,bias=False),
                           nn.GroupNorm(groups,width),nn.GELU()])
            previous=width
        self.blocks=nn.Sequential(*blocks)
        self.final=nn.Conv2d(previous,1,1)
        nn.init.zeros_(self.final.weight);nn.init.zeros_(self.final.bias)

    def forward(self, features, mask):
        # Final convolution retains S's scaled-output units; M1 is unscaled.
        return (features[:,257:258]+self.final(self.blocks(features))/100)*mask


def residual_loss(raw, target, mask, counts):
    raw=raw*mask; target=target*mask
    pixels=((100*(raw-target))**2).flatten(1).sum(1)/mask.flatten(1).sum(1).clamp_min(1)
    raw_counts=raw.flatten(1).sum(1)
    negative=(-raw).clamp_min(0).flatten(1).sum(1)
    denominator=counts.clamp_min(1).sqrt()
    count_term=.1*(raw_counts-counts).abs()/denominator
    negative_term=negative/denominator
    loss=(pixels+count_term+negative_term).mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual loss')
    return loss, {k:float(v.detach().mean()) for k,v in
                  [('pixel_mse',pixels),('count_term',count_term),('negative_term',negative_term),
                   ('negative_mass',negative),('raw_count',raw_counts),
                   ('clamped_count',raw.clamp_min(0).flatten(1).sum(1))]}


def epoch_indices(names, counts, seed, epoch):
    indices=list(range(len(names)))
    indices += [i for i,n in enumerate(names) if counts[n]>=300 for _ in range(3)]
    return np.random.default_rng(seed+epoch).permutation(indices).tolist()


def expand_holdout(original, counts, labels):
    fit=original['train_fit'];hold=list(original['train_holdout'])
    ranked=sorted(original['train_fit_classes'],
                  key=lambda c:(-sum(counts[n]>=300 for n in fit if labels[n]==c),c))
    added=[]
    for category in ranked[1::2]:
        if sum(counts[n]>=300 for n in hold)>=10:break
        added.append(category);hold.extend(n for n in fit if labels[n]==category)
    if sum(counts[n]>=300 for n in hold)<10:raise ValueError('Fixed holdout rule cannot reach ten dense images')
    return [n for n in fit if labels[n] not in added],hold,added


def safety_stop(epoch, rmse, baseline_rmse):
    if not all(math.isfinite(x) and x>=0 for x in (rmse,baseline_rmse)):
        raise FloatingPointError('Invalid safety metric')
    return epoch==10 and rmse>baseline_rmse


def diagnostic_label(ratio):
    if not math.isfinite(ratio) or ratio<0:raise ValueError('Invalid diagnostic ratio')
    return 'fit_problem_signal' if ratio<.7 else ('generalization_problem_signal' if ratio>=.9 else 'mixed')


def render_m1_grid(normalized_boxes, valid_height, valid_width, canvas=512):
    boxes=np.asarray(normalized_boxes,dtype=np.float64).reshape(-1,4)*512
    m1=boxes_to_density(boxes,valid_height,valid_width)
    result=np.zeros((canvas,canvas),dtype=np.float32)
    result[:valid_height,:valid_width]=m1
    if abs(float(result.sum(dtype=np.float64))-len(boxes))>1e-4:
        raise ValueError('M1 float32 conversion lost too much mass')
    return result


def frozen_forward(adapter, sample):
    """One unchanged dense/SAM pass provides features and final detections."""
    from utils.box_ops import boxes_with_scores
    x=sample['image'].cuda();prompts=sample['boxes'].cuda()
    features=adapter.encode(x);appearance=adapter.appearance(features,prompts)
    adapted,centerness,coords=adapter.dense(features,appearance,prompts[:,:,2:]-prompts[:,:,:2])
    outputs,_=boxes_with_scores(centerness,coords,sort=False,validate=True)
    output=adapter.refine(features,outputs,'auto')
    final,scores=postprocess(output,int(sample['width']*sample['scale']),int(sample['height']*sample['scale']))
    normalized=final.detach().cpu().numpy()
    m1=render_m1_grid(normalized,sample['valid_height'],sample['valid_width'])
    tensor=torch.from_numpy(m1)[None,None].to(device=adapted.device)
    density_features=torch.cat([adapted,centerness,tensor],dim=1)
    return density_features, output, normalized*(1024/sample['scale']),scores.detach().cpu().numpy()
