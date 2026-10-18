import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from planb.geometry import make_tiles,restore_tile_boxes
from planb.geco_adapter import postprocess
from planb.round3_m3 import tiles,threshold_output,require_equal
from planb.round3_selection import crossfit_choices,candidate_gate
from planb.round3_accounting import StageLedger
from planb.round2.adapter import Round2Adapter


class M3Tests(unittest.TestCase):
    def test_grid2_preserves_existing_crops(self):
        for w,h in [(101,77),(384,697),(1024,384)]:self.assertEqual(tiles(w,h,2),make_tiles(w,h))

    def test_core_partition_including_internal_boundaries(self):
        for g in (2,3):
            w,h=101,77;owners=np.zeros((h,w),dtype=int)
            for t in tiles(w,h,g):
                x0,y0,x1,y1=t['core'];owners[y0:y1,x0:x1]+=1
                self.assertLessEqual(t['crop'][0],x0);self.assertGreaterEqual(t['crop'][2],x1)
            self.assertTrue(np.all(owners==1))

    def test_supplied_max_equals_baseline_and_rejects_weak_tile(self):
        out={'pred_boxes':torch.tensor([[[.1,.1,.2,.2],[.6,.6,.7,.7]]]),
             'box_v':torch.tensor([[1.,.1]]),'scores':torch.tensor([[.8,.7]])}
        a=postprocess(out,1024,1024);b=threshold_output(out,1024,1024,1.)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertEqual(len(threshold_output(out,1024,1024,20.)[0]),0)

    def test_threshold_precedes_nms_and_padding_filter(self):
        out={'pred_boxes':torch.tensor([[[.1,.1,.3,.3],[.11,.11,.31,.31],[.8,.8,.9,.9]]]),
             'box_v':torch.tensor([[1.,.9,.8]]),'scores':torch.ones(1,3)}
        boxes,scores=threshold_output(out,512,512,1.)
        self.assertEqual(len(boxes),1);self.assertEqual(float(scores[0]),1.)

    def test_empty_and_nonfinite_outputs(self):
        o={'pred_boxes':torch.empty(1,0,4),'box_v':torch.empty(1,0),'scores':torch.empty(1,0)}
        self.assertEqual(len(threshold_output(o,1024,1024,0)[0]),0)
        o={'pred_boxes':torch.zeros(1,1,4),'box_v':torch.ones(1,1),'scores':torch.tensor([[float('nan')]])}
        with self.assertRaises(ValueError):threshold_output(o,1024,1024,1)

    def test_repeat_checks_raw_not_only_final_count(self):
        a={k:torch.ones(2) for k in ('pred_boxes','box_v','scores')};b={k:v.clone() for k,v in a.items()}
        b['box_v'][0]+=1e-4
        with self.assertRaises(ValueError):require_equal(a,b)

    def test_forced_sam_chunks_801_without_changing_order(self):
        class Model:
            def __init__(self):self.calls=[]
            def sam_mask(self,features,outputs):
                o=outputs[0];self.calls.append(o['pred_boxes'].shape[1])
                return [torch.empty(0)],[o['box_v']],[o['pred_boxes'][0]*1024]
        model=Model();adapter=Round2Adapter(model)
        boxes=torch.arange(801*4,dtype=torch.float32).reshape(1,801,4)/10000
        scores=torch.arange(801,dtype=torch.float32)[None]
        out=adapter.refine({},[{'pred_boxes':boxes.clone(),'box_v':scores.clone()}],'on')
        self.assertEqual(model.calls,[800,1]);self.assertTrue(torch.equal(out['pred_boxes'],boxes))
        self.assertTrue(torch.equal(out['scores'],scores))
        adapter.refine({},[{'pred_boxes':boxes.clone(),'box_v':scores.clone()}],'off')
        self.assertEqual(model.calls,[800,1])

    def test_assessment_values_cannot_change_its_choice(self):
        names=['a','b','c','d'];folds={'A':['a','b'],'B':['c','d']}
        base={n:{'truth':10,'prediction':8} for n in names}
        variants={'x':{n:{'prediction':10} for n in names},'y':{n:{'prediction':9} for n in names}}
        chosen,record=crossfit_choices(names,folds,variants,base,['x','y'])
        for n in folds['B']:
            base[n]['truth']=1000;variants['x'][n]['prediction']=-999
        again,_=crossfit_choices(names,folds,variants,base,['x','y'])
        self.assertEqual(chosen['c'],again['c']);self.assertEqual(chosen['d'],again['d'])
        self.assertEqual(record['B']['fit_fold'],'A')

    def test_no_dense_images_cannot_pass_gate(self):
        rows=[{'image':'a','truth':1,'prediction':1}]
        self.assertFalse(candidate_gate(rows,{'a':{'prediction':2}}))


class BudgetTests(unittest.TestCase):
    def test_failure_is_charged_and_no_retry(self):
        with tempfile.TemporaryDirectory() as t:
            ledger=StageLedger.create(Path(t)/'l.json',{'a':100,'b':50},150)
            ledger.begin('first','a',90,'now');ledger.finish('first',30,False,'then')
            self.assertEqual(ledger.remaining('a'),70)
            with self.assertRaises(RuntimeError):ledger.begin('retry','a',90,'later')

    def test_stage_cannot_borrow_and_active_cannot_restart(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'l.json';ledger=StageLedger.create(path,{'a':50,'b':100},150)
            self.assertEqual(ledger.begin('first','a',100,'now'),50)
            with self.assertRaises(RuntimeError):StageLedger(path).begin('second','b',50,'later')
            self.assertFalse(ledger.finish('first',51,True,'then'))
            self.assertEqual(ledger.remaining('b'),99)

    def test_duplicate_settlement_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            ledger=StageLedger.create(Path(t)/'l.json',{'a':100},100)
            ledger.begin('first','a',90,'now');ledger.finish('first',20,True,'then')
            with self.assertRaises(ValueError):ledger.finish('first',20,True,'again')
            with self.assertRaises(FileExistsError):ledger.begin('first','a',90,'again')


if __name__=='__main__':unittest.main()
