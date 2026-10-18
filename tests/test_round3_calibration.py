import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_round3_calibration import audit_model,audit_decision,reconstruct_prediction
from planb.round3_calibration import (features, fitting_arrays, poisson_objective, fit, gain,
                                    apply_map, masses, game_cells, inner_folds, choose_inner,
                                    crossfit, predict)

CONFIG={'l2':1.,'maxiter':1000,'ftol':1e-12,'gtol':1e-8,'gain_bounds':[.5,2.]}


def record(name, factor=1., mass_value=5.):
    d=np.full((8,8),mass_value);y=d*factor
    x=np.stack([np.log1p(d),np.full((8,8),.1),np.full((8,8),.3),d/256],axis=-1)
    return {'image':name,'mass8':d.tolist(),'truth8':y.tolist(),'features':x.tolist(),
            'truth':float(y.sum()),'prediction':float(d.sum()),
            'game1':float(abs(d-y).sum()),'game2':float(abs(d-y).sum())}


class CalibrationTests(unittest.TestCase):
    def test_original_pixel_sides_and_canvas_area(self):
        d=np.zeros((8,8));d[1,0]=6;d[0,0]=2
        x=features(d,[[0,0,2,4]],[[0,0,4,4]]*3,16,24,2.)
        self.assertAlmostEqual(x[1,0,0],np.log(7))
        self.assertAlmostEqual(x[1,0,1],np.log(4))
        self.assertAlmostEqual(x[1,0,2],np.log(3))
        self.assertAlmostEqual(x[0,0,2],np.log(4))
        self.assertAlmostEqual(x[1,0,3],6/(2*3*4))

    def test_invalid_feature_geometry_rejected(self):
        for boxes in ([[0,0,0,0]],[[-4,0,-2,2]]):
            with self.assertRaises(ValueError):features(np.ones((8,8)),boxes,[[0,0,4,4]]*3,16,16,1.)

    def test_positive_mass_support_only(self):
        r=record('a',1.5);r['mass8'][0][0]=0;r['truth8'][0][0]=999;r['features'][0][0]=[1e6]*4
        x,d,y=fitting_arrays([r]);self.assertEqual(len(d),63);self.assertLess(x.max(),10);self.assertLess(y.max(),10)
        g=fit([r],'global',CONFIG);self.assertAlmostEqual(g['intercept'],np.log(1.5));self.assertEqual(g['zero_mass_cells_excluded'],1)

    def test_poisson_gradient_and_unclipped_fit_link(self):
        rng=np.random.default_rng(7);z=np.column_stack((np.ones(9),rng.normal(size=(9,4))))
        d=np.arange(1,10,dtype=float);y=d*.8;beta=np.array([3.,.1,-.2,.05,.03])
        value,g=poisson_objective(beta,z,d,y,1.)
        numeric=[]
        for i in range(5):
            step=np.zeros(5);step[i]=1e-6
            numeric.append((poisson_objective(beta+step,z,d,y,1.)[0]-poisson_objective(beta-step,z,d,y,1.)[0])/2e-6)
        np.testing.assert_allclose(g,numeric,rtol=1e-6,atol=1e-6)
        self.assertGreater(g[0],10)
        self.assertTrue(np.isfinite(value))

    def test_global_analytic_clip_and_zero_target(self):
        r=record('a',4.);m=fit([r],'global',CONFIG)
        np.testing.assert_array_equal(gain(r['features'],m),np.full((8,8),2.))
        r=record('a',0.);m=fit([r],'global',CONFIG)
        self.assertIsNone(m['intercept']);self.assertTrue(m['zero_target_boundary'])
        np.testing.assert_array_equal(gain(r['features'],m),np.full((8,8),.5))

    def test_constant_decimal_features_and_known_solution(self):
        r=record('a',1.5);m=fit([r],'regional',CONFIG)
        np.testing.assert_array_equal(m['std'],np.ones(4))
        np.testing.assert_allclose(m['coefficients'][1:],np.zeros(4),atol=1e-10)
        self.assertAlmostEqual(m['coefficients'][0],np.log(1.5),places=6)

    def test_failed_optimizer_and_nonfinite_stop(self):
        with patch('planb.round3_calibration.minimize',return_value=SimpleNamespace(success=False)):
            with self.assertRaises(RuntimeError):fit([record('a')],'regional',CONFIG)
        r=record('a');r['features'][0][0][0]=float('nan')
        with self.assertRaises(ValueError):fit([r],'global',CONFIG)
        with self.assertRaises(FloatingPointError):poisson_objective(np.array([1000.,0,0,0,0]),np.ones((1,5)),np.ones(1),np.ones(1),1.)

    def test_map_mass_and_zero_support(self):
        d=np.arange(16*24,dtype=float).reshape(16,24)/100;d[:2,:3]=0
        g=np.linspace(.5,2,64).reshape(8,8);corrected=apply_map(d,g)
        np.testing.assert_allclose(masses(corrected),masses(d)*g,rtol=1e-12,atol=1e-12)
        self.assertEqual(float(corrected[:2,:3].sum()),0.)
        self.assertAlmostEqual(game_cells(masses(corrected),np.zeros((8,8)),2),corrected.sum())

    def test_inner_classes_alphabetical_with_original_image_order(self):
        names=['x','y','z','w'];labels={'x':'pear','y':'apple','z':'pear','w':'banana'}
        self.assertEqual(inner_folds(names,labels),{'A':['x','y','z'],'B':['w']})

    def test_tie_selects_global_and_no_gain_fails_admission(self):
        records={str(i):record(str(i)) for i in range(4)};labels={str(i):str(i) for i in range(4)}
        d=choose_inner(list(records),records,labels,records,CONFIG)
        self.assertEqual(d['selected'],'global');self.assertFalse(d['admitted'])

    def test_outer_labels_cannot_change_fit_fold_decisions(self):
        records={str(i):record(str(i),1.2 if i%2 else .8,5.+i/10) for i in range(8)}
        labels={n:'class'+n for n in records};names=list(records);folds={'A':names[:4],'B':names[4:]}
        first=crossfit(names,folds,records,labels,records,CONFIG)
        changed=copy.deepcopy(records)
        for n in folds['B']:
            changed[n]['truth8']=(np.asarray(changed[n]['truth8'])*2).tolist();changed[n]['truth']*=2
        second=crossfit(names,folds,changed,labels,changed,CONFIG)
        self.assertEqual(first['outer_models']['B'],second['outer_models']['B'])
        self.assertEqual(first['outer_decisions']['B'],second['outer_decisions']['B'])
        self.assertEqual([r['prediction'] for r in first['rows']['regional'][4:]],
                         [r['prediction'] for r in second['rows']['regional'][4:]])

    def test_outer_class_overlap_rejected(self):
        records={str(i):record(str(i)) for i in range(4)};labels={str(i):'same' for i in range(4)}
        with self.assertRaises(ValueError):crossfit(list(records),{'A':['0','1'],'B':['2','3']},records,labels,records,CONFIG)

    def test_prediction_count_does_not_use_assessment_truth(self):
        r=record('a',1.1);model=fit([r],'regional',CONFIG);before=predict(r,model)
        changed=copy.deepcopy(r);changed['truth8']=np.full((8,8),999).tolist();changed['truth']=999*64
        after=predict(changed,model)
        self.assertEqual(before['prediction'],after['prediction']);self.assertEqual(before['gains8'],after['gains8'])

    def test_independent_audit_rejects_changed_coefficient_or_fit_scope(self):
        records={'a':record('a',1.2),'b':record('b',.8)}
        m=fit(list(records.values()),'regional',CONFIG);audit_model(m,list(records),records,CONFIG)
        bad=copy.deepcopy(m);bad['coefficients'][0]+=.1
        with self.assertRaises(ValueError):audit_model(bad,list(records),records,CONFIG)
        bad=copy.deepcopy(m);bad['fit_images']=['a']
        with self.assertRaises(ValueError):audit_model(bad,list(records),records,CONFIG)

    def test_independent_audit_rejects_changed_inner_choice(self):
        records={str(i):record(str(i)) for i in range(4)};labels={n:'class'+n for n in records}
        d=choose_inner(list(records),records,labels,records,CONFIG)
        audit_decision(d,list(records),records,labels,records,CONFIG)
        d['selected']='regional'
        with self.assertRaises(ValueError):audit_decision(d,list(records),records,labels,records,CONFIG)

    def test_independent_predictions_match_for_nonconstant_features(self):
        r=record('a',1.2);r['features']=np.random.default_rng(42).normal(size=(8,8,4)).tolist()
        model=fit([r],'regional',CONFIG);a=predict(r,model);b=reconstruct_prediction(r,model)
        for k in ('prediction','mass8','gains8','game1','game2'):np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-8)


if __name__=='__main__':unittest.main()
