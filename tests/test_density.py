import unittest
import numpy as np
from planb.density import boxes_to_density, game


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


if __name__ == '__main__':
    unittest.main()
