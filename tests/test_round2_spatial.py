import unittest
import numpy as np
import torch
from planb.density import game
from planb.round2.spatial import restore_density, cell_mass, point_cells, game_from_cells, localization, fused_map
from planb.round2.fusion import select_candidate, fit_gate, gate_weights


class SpatialTests(unittest.TestCase):
    def test_restore_mass_and_nested_grid_on_odd_image_dimensions(self):
        q = torch.ones(1,1,512,512)
        density = restore_density(q, 17, 19, 6, 7)
        self.assertAlmostEqual(float(density.sum()), .42, places=12)
        points = np.array([[0.,0.],[18.99,16.99],[7.1,9.2]])
        truth = point_cells(points,17,19)[1]
        for level in (1,2):
            self.assertAlmostEqual(game(density,points,level),game_from_cells(cell_mass(density),truth,level),places=12)

    def test_spatial_fusion_matches_cell_weighted_mass(self):
        first = np.arange(17*19,dtype=float).reshape(17,19)/1000
        second = first[::-1].copy()*2
        weights = np.linspace(0,1,64).reshape(8,8)
        result = fused_map(first,second,weights)
        np.testing.assert_allclose(cell_mass(result),weights*cell_mass(first)+(1-weights)*cell_mass(second),atol=1e-12)

    def test_matching_uses_maximum_cardinality_instead_of_greedy_nearest(self):
        boxes = [[1,1,3,3],[0,1,2,3]]
        points = [[2,2],[4,2]]
        report = localization(boxes,points,[[0,0,4,4]]*3)
        self.assertEqual(report['tp'],2)
        self.assertEqual(report['f1'],1.)

    def test_selection_applies_rmse_constraint_before_mae(self):
        chosen, _ = select_candidate({'S':[0,0,0,8],'L':[2,2,2,2]},[0]*4,[3]*4,['S','L'])
        self.assertEqual(chosen,'L')
        chosen, _ = select_candidate({'S':[5]*4,'L':[4]*4},[0]*4,[3]*4,['S','L'])
        self.assertEqual(chosen,'L')

    def test_constant_feature_gate_recovers_known_mixture_without_refitting(self):
        config = {'l2':.001,'max_iterations':1000,'ftol':1e-12,'gtol':1e-8}
        x = np.zeros((2,8,8,6)); m1 = np.full((2,8,8),10.); m4=np.zeros_like(m1)
        model = fit_gate(x,m1,m4,np.full_like(m1,3.),config)
        before = repr(model)
        np.testing.assert_allclose(gate_weights(x,model),.3,atol=1e-6)
        gate_weights(np.full((1,8,8,6),10000.),model)
        self.assertEqual(before,repr(model))
        self.assertEqual(model['std'],[1.]*6)

    def test_decimal_constant_features_use_unit_scale_despite_roundoff(self):
        config = {'l2':.001,'max_iterations':1000,'ftol':1e-12,'gtol':1e-8}
        x = np.full((2,8,8,6),.1)
        m1 = np.full((2,8,8),10.);m4 = np.zeros_like(m1)
        model = fit_gate(x,m1,m4,np.full_like(m1,3.),config)
        self.assertEqual(model['std'],[1.]*6)
        np.testing.assert_allclose(gate_weights(x,model),.3,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
