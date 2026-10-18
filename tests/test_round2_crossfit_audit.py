import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_round2_crossfit import check_partitions,check_scaler,choose_width


class CrossfitAuditTests(unittest.TestCase):
    def test_overlapping_class_is_rejected_even_with_distinct_images(self):
        splits={'val_A':['a'],'val_B':['b']}
        folds={'A':{'fit_images':['a'],'assess_images':['b']},'B':{'fit_images':['b'],'assess_images':['a']}}
        check_partitions(folds,splits,{'a':'class-a','b':'class-b'})
        with self.assertRaises(AssertionError):check_partitions(folds,splits,{'a':'same','b':'same'})
        wrong=copy.deepcopy(folds);wrong['A']['assess_images']=['a']
        with self.assertRaises(AssertionError):check_partitions(wrong,splits,{'a':'class-a','b':'class-b'})

    def test_scaler_from_another_fold_is_rejected(self):
        x=np.arange(128*6,dtype=float).reshape(2,8,8,6)
        model={'mean':x.reshape(-1,6).mean(0).tolist(),'std':x.reshape(-1,6).std(0).tolist(),
               'fitting_cells':128,'fitting_images':2}
        check_scaler(x,model)
        with self.assertRaises(AssertionError):check_scaler(x+100,model)

    def test_selection_uses_declared_fit_names_only(self):
        baseline={'a':{'prediction':3,'truth':0},'b':{'prediction':3,'truth':0}}
        rows={'S':{'a':{'prediction':1},'b':{'prediction':100}},
              'L':{'a':{'prediction':2},'b':{'prediction':0}}}
        self.assertEqual(choose_width(['a'],rows,baseline),'S')
        self.assertEqual(choose_width(['b'],rows,baseline),'L')


if __name__=='__main__':unittest.main()
