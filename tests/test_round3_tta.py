import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
import torch
from planb.round3_tta import flip_boxes, transformed_input, project_boxes, mean_density, VIEWS
from planb.density import boxes_to_density
from planb.round3_accounting import StageLedger
from planb.round3_followup import inherit_ledger
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_round3_tta import rebuild_view,correlation


class TTATests(unittest.TestCase):
    def test_independent_empty_view_reconstruction_and_zero_variance_diagnostic(self):
        view={'boxes_normalized_xyxy':[], 'boxes_original_xyxy':[], 'objectness':[],
              'effective_scale':1., 'flipped':True, 'count':0}
        density,boxes=rebuild_view(view,13,17)
        self.assertEqual(boxes.shape,(0,4));self.assertEqual(float(density.sum()),0.)
        rows=[{'detail':{'view_count_std':0.},'prediction':n,'truth':n+1} for n in (1,2)]
        self.assertIsNone(correlation(rows)['pearson_r'])

    def test_identity_projection_preserves_baseline_float32_arithmetic(self):
        boxes=np.array([[.123,.045,.345,.567]],np.float32);scale=1.78397212543554
        expected=boxes*(1024/scale)
        actual=project_boxes(boxes,scale,574,384,False)
        self.assertEqual(actual.dtype,expected.dtype)
        self.assertEqual(actual.tobytes(),expected.tobytes())

    def test_flip_exemplars_is_involution_and_preserves_size(self):
        boxes=torch.tensor([[0.,1.,4.,7.],[3.,2.,11.,5.],[8.,0.,13.,9.]])
        transformed=flip_boxes(boxes,13)
        self.assertTrue(torch.equal(flip_boxes(transformed,13),boxes))
        self.assertTrue(torch.equal(transformed[:,2:]-transformed[:,:2],boxes[:,2:]-boxes[:,:2]))
        self.assertEqual(transformed[0,0],9)

    def test_scale_applies_after_upstream_limit_and_padding_stays_right_bottom(self):
        image=torch.zeros(3,100,200);image[:,:,0:50]=1
        prompts=torch.tensor([[0.,0.,10.,10.],[20.,0.,30.,10.],[40.,0.,50.,10.]])
        padded,p,scale,valid=transformed_input(image,prompts,2.,.8,True)
        self.assertEqual(scale,1.6);self.assertEqual(valid,(160,320));self.assertEqual(padded.shape,(3,1024,1024))
        self.assertEqual(float(padded[:,160:,:].sum()+padded[:,:,320:].sum()),0.)
        self.assertTrue(torch.equal(p,flip_boxes(prompts,200)*torch.tensor([1.6]*4)))
        self.assertEqual(float(padded[0,30,10]),0.);self.assertEqual(float(padded[0,30,300]),1.)

    def test_inverse_projection_known_original_rectangle(self):
        # Original [10,20,30,40], flipped at width100 -> [70,20,90,40].
        normalized=np.array([[70,20,90,40]],np.float32)*2/1024
        restored=project_boxes(normalized,2,100,60,True)
        np.testing.assert_array_equal(restored,[[10.,20.,30.,40.]])
        self.assertAlmostEqual(boxes_to_density(restored,60,100).sum(),1.)

    def test_boundary_degenerate_detection_kept_without_losing_mass(self):
        b=project_boxes(np.array([[0.,.1,0.,.2]],np.float32),1.,100,300,True)
        self.assertLess((b[0,0]+b[0,2])/2,100.)
        self.assertAlmostEqual(boxes_to_density(b,300,100).sum(),1.)

    def test_averaging_counts_and_locations_including_empty_view(self):
        a=boxes_to_density([[1,1,3,3]],12,16);z=np.zeros_like(a)
        result,count=mean_density([a,z],[1,0])
        self.assertEqual(count,.5);np.testing.assert_array_equal(result,a/2)
        self.assertAlmostEqual(result.sum(),.5)

    def test_repeat_map_is_bitwise_identical(self):
        a=boxes_to_density([[.123,.456,5.987,8.765],[4,3,7,9]],13,17)
        result,count=mean_density([a.copy() for _ in range(4)],[2]*4,repeat=True)
        np.testing.assert_array_equal(a,result);self.assertEqual(count,2.)
        altered=a.copy();altered[0,0]+=1e-9
        with self.assertRaises(ValueError):mean_density([a,altered],[2,2],repeat=True)

    def test_nonfinite_negative_and_mass_mismatch_fail(self):
        for a in [np.array([[float('nan')]]),np.array([[-1.]]),np.array([[2.]])]:
            with self.assertRaises(ValueError):mean_density([a],[1])
        with self.assertRaises(ValueError):mean_density([],[])
        self.assertEqual(VIEWS['T4'],[(1.,False),(1.,True),(.8,False),(.8,True)])


class InheritanceTests(unittest.TestCase):
    def make(self,root):
        path=root/'old.json';ledger=StageLedger.create(path,{'pre':100,'next':200},300)
        ledger.begin('old-stage','pre',50,'a');ledger.finish('old-stage',20,True,'b')
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_inherit_once_preserves_old_bytes_and_charged_time(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old,digest=self.make(root);original=old.read_bytes();new=root/'new.json'
            ledger=inherit_ledger(old,new,digest,{'pre':100,'next':200},300)
            self.assertEqual(len(ledger.data['entries']),1);self.assertEqual(ledger.remaining('pre'),80)
            ledger.begin('next-stage','next',100,'c');ledger.finish('next-stage',30,True,'d')
            self.assertEqual(old.read_bytes(),original);self.assertEqual(sum(e['seconds'] for e in ledger.data['entries']),50)
            with self.assertRaises(FileExistsError):inherit_ledger(old,new,digest,{'pre':100,'next':200},300)

    def test_changed_caps_or_unsettled_parent_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);old,digest=self.make(root)
            with self.assertRaises(ValueError):inherit_ledger(old,root/'a.json',digest,{'pre':200,'next':100},300)
            ledger=StageLedger(old);ledger.begin('still-running','next',50,'c')
            with self.assertRaises(ValueError):inherit_ledger(old,root/'b.json',digest,{'pre':100,'next':200},300)
            digest=hashlib.sha256(old.read_bytes()).hexdigest()
            with self.assertRaises(RuntimeError):inherit_ledger(old,root/'b.json',digest,{'pre':100,'next':200},300)


if __name__=='__main__':unittest.main()
