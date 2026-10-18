import unittest
import numpy as np
from planb.density import boxes_to_density, game, points_for_game


class DensityTests(unittest.TestCase):
    def test_mass_at_edges_and_overlap(self):
        boxes = [[0,0,1,1], [0,0,1,1], [30,18,32,20], [2,3,22,17]]
        d = boxes_to_density(boxes,20,32)
        self.assertAlmostEqual(d.sum(),4.,places=12)
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue((d>=0).all())

    def test_empty(self):
        d = boxes_to_density([],8,8)
        self.assertEqual(d.sum(),0)
        self.assertEqual(game(d,[],2),0)

    def test_game_exposes_spatial_mismatch(self):
        d = np.zeros((8,8)); d[1,1] = 1
        self.assertEqual(game(d,[[6,6]],0),0)
        self.assertEqual(game(d,[[6,6]],1),2)

    def test_invalid_centre(self):
        with self.assertRaises(ValueError):
            boxes_to_density([[10,10,12,12]],8,8)

    def test_annotation_edge_roundoff_preserves_count_and_cell(self):
        # Actual FSC val 1989.jpg coordinate, plus exact far-edge points.
        original = np.array([[408.00000000000006, 350.832], [408., 384.], [0., 0.]])
        points = points_for_game(original, 384, 408)
        self.assertEqual(len(points), 3)
        self.assertTrue((points >= 0).all())
        self.assertTrue((points < [408, 384]).all())
        self.assertEqual(original[0, 0], 408.00000000000006)
        d = np.zeros((384, 408)); d[350, 407] = 1; d[383, 407] = 1; d[0, 0] = 1
        self.assertEqual(game(d, points, 2), 0)

    def test_annotation_real_overflow_and_nonfinite_rejected(self):
        for point in ([408.000001, 350.832], [-0.000001, 20], [1, 384.000001], [np.nan, 2], [np.inf, 2]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                points_for_game([point], 384, 408)


if __name__ == '__main__':
    unittest.main()
