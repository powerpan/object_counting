import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from planb.final_metrics import size_group,success,intervals,qualitative_selection,report
from planb.final_inference import trigger
from planb.round3_accounting import StageLedger


class FinalChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/final_v1.json').read_text())
        self.rows=[]
        for i,(n,delta) in enumerate([(10,2),(20,-2),(300,40),(500,-50)]):
            row={'image':str(i)+'.jpg','class':str(i//2),'truth':n,'size_group':'small','detail':{'triggered':i>1}}
            for m,d in [('b0',delta),('final',delta*.5)]:
                row[m]={'prediction':n+d,'localization':{'tp':n,'fp':0,'fn':0},'game1':abs(d),'game2':abs(d),
                        'inference_seconds':.2,'peak_allocated_mib':100}
            self.rows.append(row)

    def test_trigger_inclusive_boundaries(self):
        self.assertTrue(trigger(32,1));self.assertTrue(trigger(33,200));self.assertFalse(trigger(32.00001,199))

    def test_size_mean_area_not_median_side(self):
        p=[[0,0,10,10],[0,0,10,10],[0,0,40,40]]
        r,g=size_group(p,100,100,[.2,.3]);self.assertAlmostEqual(r,np.sqrt(.06));self.assertEqual(g,'medium')

    def test_size_boundaries(self):
        self.assertEqual(size_group([[0,0,10,10]]*3,100,100,[.1,.2])[1],'small')
        self.assertEqual(size_group([[0,0,20,20]]*3,100,100,[.1,.2])[1],'medium')

    def test_success_all_conditions_required(self):
        b={'rmse':10,'mae':5};f={'rmse':9,'mae':5};d={'n':2,'mae':30};fd={'n':2,'mae':28}
        ci={'rmse_delta_95ci':[-2,-.1]};rules=self.cfg['success']
        self.assertTrue(success(b,f,d,fd,ci,rules)['passed'])
        for ff,dd,cc in [({'rmse':9.9,'mae':5},fd,ci),(f,{'n':2,'mae':29},ci),
                          (f,fd,{'rmse_delta_95ci':[-2,0]}),({'rmse':9,'mae':5.2},fd,ci)]:
            self.assertFalse(success(b,ff,d,dd,cc,rules)['passed'])

    def test_empty_dense_group_cannot_pass(self):
        self.assertFalse(success({'rmse':10,'mae':5},{'rmse':9,'mae':4},{'n':0,'mae':None},
                          {'n':0,'mae':None},{'rmse_delta_95ci':[-2,-1]},self.cfg['success'])['passed'])

    def test_identity_bootstrap_exact_zero(self):
        rows=copy.deepcopy(self.rows)
        for r in rows:r['final']=copy.deepcopy(r['b0'])
        for u in ['image','class']:
            ci=intervals(rows,u);self.assertEqual(ci['rmse_delta_95ci'],[0.,0.]);self.assertEqual(ci['mae_delta_95ci'],[0.,0.])

    def test_bootstrap_direction_and_unit(self):
        for u,k in [('image',4),('class',2)]:
            ci=intervals(self.rows,u);self.assertLess(ci['rmse_delta_95ci'][1],0);self.assertEqual(ci['units'],k)

    def test_qualitative_tie_break_and_frozen_random(self):
        selected=qualitative_selection(list(reversed(self.rows)),self.cfg)
        self.assertEqual(selected['random_six'],self.cfg['qualitative']['random_six'])
        self.assertEqual(selected['worst_final'][:2],['3.jpg','2.jpg'])
        self.assertEqual(selected['worst_final'][2:],['0.jpg','1.jpg'])

    def test_counts_and_group_sizes(self):
        r=report(self.rows,self.cfg,'fsc147-test');self.assertEqual(r['groups']['>=300']['b0']['n'],2)
        self.assertEqual(r['groups']['overall']['b0']['object_count'],830);self.assertEqual(r['groups']['50-299']['b0']['n'],0)

    def test_budget_no_retry_or_cross_stage_borrow(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=StageLedger.create(Path(d)/'l.json',{'a':20,'b':30},50)
            self.assertEqual(ledger.begin('one','a',100,'now'),20)
            self.assertFalse(ledger.finish('one',21,True,'end'))
            with self.assertRaises(RuntimeError):ledger.begin('two','b',30,'now')


if __name__=='__main__':unittest.main()
