import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.round3_combination_plan import make_plan,route_image
from audit_round3_combination_plan import audit_plan


def fixture():
    names=['a','b','c','d'];folds={'A':['a','b'],'B':['c','d']};labels={n:n for n in names}
    truth=dict(zip(names,[500,20,500,20]))
    base={n:{'image':n,'truth':truth[n],'prediction':truth[n]+10} for n in names}
    cfg={'m3':{'candidate_order':['m3-v2-g2','m3-v2-g3'],'mask_candidate_order':['b0-refine-off','b0-refine-on']},
         'tta':{'candidate_order':['T2','T4']}}
    variants={v:{n:{'image':n,'truth':truth[n],'prediction':truth[n]+offset,'triggered':n in ('a','c')} for n in names}
              for v,offset in [('m3-v2-g2',1),('m3-v2-g3',2),('T2',3),('T4',4),('b0-refine-off',5),('b0-refine-on',6)]}
    cal={'outer_decisions':{assess:{'fit_images':folds[fit],'selected':'global','admitted':False}
                            for fit,assess in [('A','B'),('B','A')]},
         'final_decision':{'fit_images':names,'selected':'global','admitted':False}}
    return names,folds,labels,base,variants,cal,cfg


class CombinationPlanTests(unittest.TestCase):
    def test_fixed_routing_and_mask_tta_inventory(self):
        args=fixture();p=make_plan(*args);audit_plan(p,*args)
        self.assertEqual(p['routes']['A'],{'a':'m3-v2-g2','b':'T2-refine-off'})
        self.assertEqual(p['missing_inference_variants'],['T2-refine-off'])
        self.assertFalse(p['requires_calibration_refit'])

    def test_assessment_truth_does_not_select_its_modules(self):
        args=fixture();p=make_plan(*args);changed=copy.deepcopy(args)
        for n in changed[1]['A']:
            changed[3][n]['truth']+=1000
            for rows in changed[4].values():rows[n]['truth']+=1000
        q=make_plan(*changed)
        self.assertEqual(p['decisions']['A'],q['decisions']['A'])
        self.assertEqual(p['routes']['A'],q['routes']['A'])

    def test_routing_never_reads_truth(self):
        args=fixture();p=make_plan(*args)
        for rows in args[4].values():
            for row in rows.values():del row['truth']
        self.assertEqual(route_image('b',p['decisions']['A'],args[4]),'T2-refine-off')

    def test_failed_modules_fall_back_to_baseline(self):
        args=fixture()
        for rows in args[4].values():
            for n,row in rows.items():row['prediction']=args[3][n]['prediction']
        p=make_plan(*args);audit_plan(p,*args)
        self.assertEqual(set(p['routes']['future-only'].values()),{'B0'})
        self.assertEqual(p['missing_inference_variants'],[])

    def test_mask_without_tta_reuses_saved_full_image(self):
        args=fixture();p=make_plan(*args);d=p['decisions']['A'];d['tta']['admitted']=False
        self.assertEqual(route_image('b',d,args[4]),'b0-refine-off')
        self.assertEqual(route_image('a',d,args[4]),'m3-v2-g2')

    def test_tta_without_mask_reuses_saved_tta(self):
        args=fixture();p=make_plan(*args);d=p['decisions']['A'];d['mask']['admitted']=False
        self.assertEqual(route_image('b',d,args[4]),'T2')

    def test_rejects_class_overlap_and_trigger_disagreement(self):
        args=fixture();args[2]['c']='a'
        with self.assertRaises(ValueError):make_plan(*args)
        args=fixture();args[4]['m3-v2-g3']['a']['triggered']=False
        with self.assertRaises(ValueError):make_plan(*args)

    def test_rejects_wrong_calibration_fit_fold(self):
        args=fixture();args[5]['outer_decisions']['A']['fit_images']=args[1]['A']
        with self.assertRaises(ValueError):make_plan(*args)

    def test_audit_rejects_changed_admission_and_route(self):
        args=fixture();p=make_plan(*args);p['decisions']['A']['tta']['admitted']=False
        with self.assertRaises(ValueError):audit_plan(p,*args)
        p=make_plan(*args);p['routes']['A']['a']='T4'
        with self.assertRaises(ValueError):audit_plan(p,*args)

    def test_audit_rejects_hidden_required_work(self):
        args=fixture();p=make_plan(*args);p['missing_inference_variants']=[]
        with self.assertRaises(ValueError):audit_plan(p,*args)


if __name__=='__main__':unittest.main()
