import copy
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.round3_combination import ready,density_from_source,assemble
from planb.density import boxes_to_density,game
from audit_round3_combination import check_lineage


def example():
    box=[4.,4.,8.,8.];points=np.array([[6.,6.]])
    annotation={'points':points.tolist(),'box_examples_coordinates':[[[4,4],[8,4],[8,8],[4,8]]]*3}
    den=boxes_to_density([box],16,16)
    row={'image':'example','truth':1,'prediction':1,'width':16,'height':16,'boxes_xyxy':[box],
         'inference_seconds_per_image':.2,'game1':game(den,points,1),'game2':game(den,points,2)}
    return row,annotation


class CombinationAssemblyTests(unittest.TestCase):
    def test_refuses_unimplemented_work(self):
        p={'missing_inference_variants':[],'requires_calibration_refit':False,'decisions':{'A':{'calibration':{'admitted':False}}}}
        ready(p)
        q=copy.deepcopy(p);q['missing_inference_variants']=['T2-refine-off']
        with self.assertRaises(RuntimeError):ready(q)
        q=copy.deepcopy(p);q['requires_calibration_refit']=True
        with self.assertRaises(RuntimeError):ready(q)
        q=copy.deepcopy(p);q['decisions']['A']['calibration']['admitted']=True
        with self.assertRaises(ValueError):ready(q)

    def test_detector_density_and_lineage(self):
        r,a=example();x=assemble(r,'m3-v2-g2',a,'abc');check_lineage(x,r,'m3-v2-g2','abc')
        self.assertAlmostEqual(x['density_mass'],1)
        self.assertEqual(x['prediction'],1)
        self.assertEqual(np.asarray(x['truth8']).sum(),1)

    def test_tta_preserves_fractional_mean_and_mass(self):
        r,a=example();box=r['boxes_xyxy'][0]
        r['detail']={'views':[{'boxes_original_xyxy':[box],'count':1},{'boxes_original_xyxy':[],'count':0}]}
        r['prediction']=.5;den=density_from_source(r,'T2')
        np.testing.assert_allclose(den,boxes_to_density([box],16,16)/2)
        self.assertAlmostEqual(den.sum(),.5)

    def test_wrong_source_count_and_unknown_variant_rejected(self):
        r,a=example();r['prediction']=2
        with self.assertRaises(ValueError):density_from_source(r,'B0')
        with self.assertRaises(ValueError):density_from_source(r,'T2-refine-on')

    def test_wrong_tta_view_count_and_average_rejected(self):
        r,a=example();r['detail']={'views':[{'boxes_original_xyxy':r['boxes_xyxy'],'count':1}]}
        with self.assertRaises(ValueError):density_from_source(r,'T2')
        r['detail']['views']*=2;r['prediction']=2
        with self.assertRaises(ValueError):density_from_source(r,'T2')

    def test_immutable_truth_and_game_checks(self):
        r,a=example();wrong=copy.deepcopy(a);wrong['points']=[]
        with self.assertRaises(ValueError):assemble(r,'B0',wrong,'abc')
        r['game2']+=1
        with self.assertRaises(ValueError):assemble(r,'B0',a,'abc')

    def test_audit_rejects_changed_source_and_count(self):
        r,a=example();x=assemble(r,'B0',a,'abc')
        with self.assertRaises(ValueError):check_lineage(x,r,'T2','abc')
        with self.assertRaises(ValueError):check_lineage(x,r,'B0','wrong')
        x['prediction']+=1
        with self.assertRaises(ValueError):check_lineage(x,r,'B0','abc')


if __name__=='__main__':unittest.main()
