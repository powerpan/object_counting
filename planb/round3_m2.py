"""Frozen M2-v2 with reusable backbone features and a saved unfiltered second pass."""
import numpy as np
import torch
from torch.nn import functional as F
from .inference_variants import ExperimentEngine
from .geco_adapter import postprocess
from .round3_m2_policy import select_candidates,should_fallback


class M2Engine(ExperimentEngine):
    def __init__(self,model,base_config,m2_config):
        super().__init__(model,base_config);self.m2_config=m2_config

    def run(self,image,exemplars,candidate,repeat=False):
        if candidate not in self.m2_config['candidates']:raise ValueError('Unknown registered M2 candidate')
        policy={**self.m2_config['policy'],**self.m2_config['candidates'][candidate]}
        h,w=image.shape[-2:];x,original,scale=self.prepare(image,exemplars)
        output,features,appearance=self.adapter.forward(x,original)
        detected,scores=postprocess(output,int(w*scale),int(h*scale))
        base=detected.cpu().numpy()*(1024/scale);base_scores=scores.cpu().numpy()
        second_boxes,second_scores=base,base_scores
        detail={'candidate':candidate,'repeat':bool(repeat),'base_count':len(base),'base_scale':scale,
                'first_detection_boxes_normalized_xyxy':detected.cpu().tolist(),
                'first_detection_boxes_model_pixels':(detected*1024).cpu().tolist(),
                'first_detection_objectness':base_scores.tolist(),'original_exemplars_model_pixels':original[0].cpu().tolist(),
                'base_refinement':'on' if output['pred_boxes'].reshape(-1,4).shape[0]<=800 else 'off',
                'first_relative_objectness':[],'candidate_cosine':[],'selection':{
                    'selected_indices':[],'other_centers':[],'eligible_before_single_object_filter':[],
                    'rejected_by_single_object_filter':[]},
                'pseudo_exemplars':[],'backbone_passes':1,'decoder_passes':1,'triggered':False}
        self.check_memory()
        if len(detected):
            candidate_boxes=(detected*1024).unsqueeze(0)
            candidate_appearance=self.adapter.appearance(features,candidate_boxes)
            cosine=torch.stack([F.normalize(c[0],dim=-1)@F.normalize(a[0],dim=-1).T
                                for c,a in zip(candidate_appearance,appearance)]).mean(0).max(1).values
            relative=scores/output['box_v'].max().clamp_min(1e-12)
            selection=select_candidates(candidate_boxes[0].cpu().numpy(),original[0].cpu().numpy(),
                                        relative.cpu().numpy(),cosine.cpu().numpy(),policy)
            detail.update(first_relative_objectness=relative.cpu().tolist(),candidate_cosine=cosine.cpu().tolist(),selection=selection)
            chosen=selection['selected_indices']
            for i in chosen:
                detail['pseudo_exemplars'].append({'first_detection_index':i,'box_xyxy':base[i].tolist(),
                    'cosine':float(cosine[i]),'relative_objectness':float(relative[i]),
                    'other_detection_centers':selection['other_centers'][i]})
            if chosen:
                if repeat:
                    ids=torch.arange(len(chosen),device=x.device)%original.shape[1]
                    extra=original[:,ids]
                else:extra=candidate_boxes[:,chosen]
                prompts=torch.cat([original,extra],dim=1)
                second_appearance=self.adapter.appearance(features,prompts)
                second=self.adapter.decode(features,second_appearance,prompts[:,:,2:]-prompts[:,:,:2])
                found,confidence=postprocess(second,int(w*scale),int(h*scale))
                second_boxes=found.cpu().numpy()*(1024/scale);second_scores=confidence.cpu().numpy()
                detail.update(triggered=True,decoder_passes=2,appended_exemplars_model_pixels=extra[0].cpu().tolist(),
                              second_refinement='on' if second['pred_boxes'].reshape(-1,4).shape[0]<=800 else 'off')
                del second,found,confidence,second_appearance,prompts,extra
                self.check_memory()
            del candidate_boxes,candidate_appearance,cosine,relative
        fallback=detail['triggered'] and should_fallback(len(base),len(second_boxes),policy['fallback_relative_count_difference'])
        detail.update(second_count=len(second_boxes),fallback=bool(fallback),second_boxes_xyxy=second_boxes.tolist(),
                      second_objectness=second_scores.tolist())
        chosen_boxes,chosen_scores=(base,base_scores) if fallback else (second_boxes,second_scores)
        return chosen_boxes,chosen_scores,base,base_scores,detail
