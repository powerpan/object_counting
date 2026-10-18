import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.round3_m2_policy import select_candidates,should_fallback
from planb.round3_m2 import M2Engine
from audit_round3_m2 import independent_selection,diagnostics
from planb.geometry import select_scale_candidates


CONFIG=json.loads((Path(__file__).resolve().parents[1]/'configs/round3_m2_v1.json').read_text())
EX=np.array([[0,0,10,10],[30,0,40,10],[60,0,70,10]],dtype=np.float32)


class M2PolicyTests(unittest.TestCase):
    def choose(self,boxes,relative=None,cosine=None,**overrides):
        cfg={**CONFIG['policy'],**CONFIG['candidates']['m2-v2-e3-c075'],**overrides}
        r=[1.]*len(boxes) if relative is None else relative;c=[1.]*len(boxes) if cosine is None else cosine
        actual=select_candidates(boxes,EX,r,c,cfg)
        self.assertEqual(actual,independent_selection(boxes,EX,r,c,cfg))
        return actual

    def test_own_center_does_not_count(self):
        result=self.choose([[100,100,120,120]])
        self.assertEqual(result['other_centers'],[0]);self.assertEqual(result['selected_indices'],[0])

    def test_one_other_allowed_two_others_rejected(self):
        result=self.choose([[100,100,120,120],[101,101,103,103]])
        self.assertIn(0,result['selected_indices']);self.assertEqual(result['other_centers'][0],1)
        result=self.choose([[100,100,120,120],[101,101,103,103],[117,117,119,119]])
        self.assertEqual(result['other_centers'][0],2);self.assertIn(0,result['rejected_by_single_object_filter'])

    def test_boundary_centers_are_inside_and_self_only_removed_once(self):
        result=self.choose([[100,100,120,120],[99,109,101,111],[119,109,121,111]])
        self.assertEqual(result['other_centers'][0],2);self.assertNotIn(0,result['selected_indices'])

    def test_rejection_happens_before_scale_bin_competition(self):
        result=self.choose([[100,100,120,120],[101,101,103,103],[117,117,119,119],[200,200,220,220]],
                           relative=[1.,.1,.1,.9])
        self.assertEqual(result['selected_indices'],[3])

    def test_same_bin_and_score_ties_keep_original_order(self):
        result=self.choose([[100,100,120,120],[200,200,220,220]])
        self.assertEqual(result['selected_indices'],[0])

    def test_distinct_bins_and_maximum_extra_count(self):
        boxes=[[100,100,120,120],[200,200,240,240],[300,300,302,302]]
        self.assertEqual(self.choose(boxes)['selected_indices'],[0,1,2])
        self.assertEqual(self.choose(boxes,max_extra_exemplars=1)['selected_indices'],[0])

    def test_original_geometry_similarity_and_confidence_thresholds(self):
        boxes=[[100,100,115,115],[200,200,201,201],[300,300,310,310]]
        self.assertEqual(self.choose(boxes,relative=[.5,1,1],cosine=[.75,1,1])['selected_indices'],[0])
        self.assertEqual(self.choose(boxes,relative=[.4999,1,1])['selected_indices'],[])
        self.assertEqual(self.choose(boxes,cosine=[.7499,1,1])['selected_indices'],[])

    def test_without_multi_center_boxes_matches_unchanged_v1_policy(self):
        rng=np.random.default_rng(83);sizes=np.array([2,5,15,20,40,60,80])
        boxes=np.array([[i*200,0,i*200+s,s] for i,s in enumerate(sizes)])
        cfg={**CONFIG['policy'],**CONFIG['candidates']['m2-v2-e3-c075']}
        for _ in range(20):
            r=rng.uniform(.4,1,len(boxes));c=rng.uniform(.7,1,len(boxes))
            self.assertEqual(self.choose(boxes,r,c)['selected_indices'],select_scale_candidates(boxes,EX,r,c,cfg))

    def test_empty_candidates_and_nonfinite_or_wrong_shapes(self):
        self.assertEqual(self.choose([])['selected_indices'],[])
        cfg={**CONFIG['policy'],**CONFIG['candidates']['m2-v2-e1-c075']}
        for f in (select_candidates,independent_selection):
            with self.assertRaises(ValueError):f([[1,1,20,20]],EX,[1],[float('nan')],cfg)
            with self.assertRaises(ValueError):f([[1,1,20,20]],EX,[],[1],cfg)
            with self.assertRaises(ValueError):f([[20,1,1,20]],EX,[1],[1],cfg)

    def test_fallback_is_strictly_more_than_half_in_both_directions(self):
        self.assertFalse(should_fallback(10,15));self.assertFalse(should_fallback(10,5))
        self.assertTrue(should_fallback(10,16));self.assertTrue(should_fallback(10,4))
        self.assertFalse(should_fallback(0,0));self.assertTrue(should_fallback(0,1))
        with self.assertRaises(ValueError):should_fallback(-1,0)
        with self.assertRaises(ValueError):should_fallback(1.,2)

    def test_no_fallback_diagnostic_does_not_claim_actual_fallback(self):
        row={'image':'3481.jpg','truth':100,'prediction':4,'detail':{
            'triggered':True,'base_count':4,'second_count':7,'fallback':True,'pseudo_exemplars':[],
            'selection':{'eligible_before_single_object_filter':[],'rejected_by_single_object_filter':[]}}}
        self.assertEqual(diagnostics([row])['fallback_images'],1)
        raw={**row,'prediction':7,'diagnostic_only':True};d=diagnostics([raw])
        self.assertEqual(d['fallback_images'],0);self.assertEqual(d['fallback_rule_triggered_images'],1)
        self.assertFalse(d['case_3481']['fallback']);self.assertEqual(d['case_3481']['prediction'],7)


class FakeAdapter:
    def __init__(self,second_count):self.encodes=0;self.decodes=0;self.prompts=[];self.second_count=second_count
    def output(self,n):
        boxes=torch.tensor([[100+i*40,100,120+i*40,120] for i in range(n)],dtype=torch.float32)/1024
        return {'pred_boxes':boxes,'box_v':torch.ones(n)}
    def appearance(self,features,boxes):
        self.prompts.append(boxes.clone());return [torch.ones(1,boxes.shape[1],2) for _ in range(3)]
    def forward(self,x,boxes):
        self.encodes+=1;return self.output(4),{},self.appearance({},boxes)
    def decode(self,features,appearance,sizes):self.decodes+=1;return self.output(self.second_count)


class M2EngineTests(unittest.TestCase):
    def run_fake(self,repeat,second_count):
        engine=M2Engine.__new__(M2Engine);engine.adapter=FakeAdapter(second_count);engine.m2_config=CONFIG
        engine.prepare=lambda im,ex:(im.unsqueeze(0),ex.unsqueeze(0),1.)
        engine.check_memory=lambda:None
        with torch.inference_mode(),patch('planb.round3_m2.postprocess',lambda o,w,h:(o['pred_boxes'],o['box_v'])):
            result=engine.run(torch.zeros(3,384,384),torch.tensor(EX),'m2-v2-e1-c075',repeat)
        return engine,result

    def test_single_cached_backbone_second_pass_and_preserved_raw_fallback(self):
        engine,(boxes,scores,first,first_scores,d)=self.run_fake(False,7)
        self.assertEqual(engine.adapter.encodes,1);self.assertEqual(engine.adapter.decodes,1)
        self.assertTrue(d['fallback']);self.assertEqual(len(boxes),4);self.assertEqual(len(d['second_boxes_xyxy']),7)
        self.assertEqual(d['appended_exemplars_model_pixels'],[[100.,100.,120.,120.]])

    def test_repeat_copies_original_prompt_with_same_policy_and_count(self):
        _,main=self.run_fake(False,6);_,repeat=self.run_fake(True,6)
        self.assertEqual(main[-1]['selection'],repeat[-1]['selection']);self.assertFalse(repeat[-1]['fallback'])
        self.assertEqual(len(repeat[0]),6);self.assertEqual(repeat[-1]['appended_exemplars_model_pixels'],EX[:1].tolist())


if __name__=='__main__':unittest.main()
