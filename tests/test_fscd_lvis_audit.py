from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_fscd_lvis_external import rebuilt_density, grid_error
from planb.density import boxes_to_density, game, points_for_game


class ExternalAuditTests(unittest.TestCase):
    def test_tta_rebuild_and_false_count_rejection(self):
        b = [[1., 1., 5., 5.]]
        f = [[10., 10., 15., 15.], [2., 10., 5., 15.]]
        row = {'height': 20, 'width': 20, 'route': 'T2',
               'b0': {'prediction': 1, 'boxes_xyxy': b}, 'final': {'prediction': 1.5},
               'detail': {'method_detail': {'views': [
                   {'flipped': False, 'scale_factor': 1., 'count': 1, 'boxes_original_xyxy': b},
                   {'flipped': True, 'scale_factor': 1., 'count': 2, 'boxes_original_xyxy': f}]}}}
        expected = (boxes_to_density(b, 20, 20)+boxes_to_density(f, 20, 20))/2
        np.testing.assert_array_equal(rebuilt_density(row, 'final'), expected)
        row['final']['prediction'] = 1
        with self.assertRaises(ValueError): rebuilt_density(row, 'final')

    def test_m3_count_and_unknown_route_rejected(self):
        row = {'height': 20, 'width': 20, 'route': 'M3-v2-g3',
               'final': {'prediction': 2, 'boxes_xyxy': [[1, 1, 4, 4]]}}
        with self.assertRaises(ValueError): rebuilt_density(row, 'final')
        row['route'] = 'unknown'
        with self.assertRaises(ValueError): rebuilt_density(row, 'final')

    def test_separate_grid_implementation_on_odd_sizes_and_edges(self):
        density = boxes_to_density([[0, 0, 2, 2], [14, 9, 16, 12]], 13, 17)
        points = points_for_game([[0, 0], [17, 13], [8, 7]], 13, 17)
        for level in (1, 2):
            self.assertAlmostEqual(grid_error(density, points, level), game(density, points, level), places=12)


if __name__ == '__main__':
    unittest.main()
