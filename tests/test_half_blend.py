from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from planb.half_blend import half_count, half_blend
from planb.density import game
from run_half_blend_replay import count_summary, independent_counts


class HalfBlendTests(unittest.TestCase):
    def test_fractional_count_is_not_rounded(self):
        self.assertEqual(half_count(2, 3.5), 2.75)

    def test_different_locations_preserve_mass_without_mutation(self):
        a = np.zeros((9, 13)); a[1, 2] = 2
        b = np.zeros_like(a); b[8, 12] = 3.5
        aa, bb = a.copy(), b.copy()
        count, density = half_blend(2, 3.5, a, b)
        self.assertEqual(count, 2.75)
        self.assertEqual(density.sum(), count)
        self.assertEqual(density[1, 2], 1)
        self.assertEqual(density[8, 12], 1.75)
        np.testing.assert_array_equal(a, aa); np.testing.assert_array_equal(b, bb)

    def test_identical_inputs_reproduce_baseline(self):
        a = np.arange(20, dtype=float).reshape(4, 5)/8
        n, out = half_blend(a.sum(), a.sum(), a, a)
        self.assertEqual(n, a.sum())
        np.testing.assert_array_equal(out, a)

    def test_invalid_counts(self):
        for bad in (-1, np.inf, np.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError): half_count(bad, 2)

    def test_bad_maps_rejected(self):
        a = np.ones((4, 4))
        for bad in (np.ones((2, 8)), np.ones((16,)), np.full((4, 4), np.nan),
                    -a, a*2, np.full((4, 4), np.inf)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                half_blend(16, 16, a, bad)
        with self.assertRaises(ValueError): half_blend(0, 0, np.zeros((0, 0)), np.zeros((0, 0)))

    def test_grid_error_must_be_computed_after_mixing(self):
        a = np.zeros((4, 4)); a[0, 0] = 2
        b = np.zeros_like(a); b[3, 3] = 2
        points = np.array([[0., 0.], [3., 3.]])
        _, blended = half_blend(2, 2, a, b)
        self.assertEqual(game(blended, points, 1), 0)
        self.assertEqual((game(a, points, 1)+game(b, points, 1))/2, 2)

    @staticmethod
    def rows():
        return [{'sample_id': str(i), 'sampling_unit': str(i), 'truth': 10,
                 'b0': {'prediction': 8}, 'original_final': {'prediction': 12},
                 'half_blend': {'prediction': 10}} for i in range(3)]

    def test_metrics_and_paired_interval_on_known_case(self):
        rows = self.rows()
        result = count_summary(rows, 40, 1)
        self.assertEqual(result['methods']['b0'], {'mae': 2., 'rmse': 2.})
        self.assertEqual(result['methods']['half_blend'], {'mae': 0., 'rmse': 0.})
        self.assertEqual(result['paired_half_minus_b0']['rmse_delta_95ci'], [-2., -2.])
        self.assertTrue(result['both_point_estimates_not_worse'])
        independent_counts(rows, result)

    def test_missing_duplicate_or_modified_prediction_rejected(self):
        with self.assertRaises(ValueError): count_summary([], 10)
        rows = self.rows()
        with self.assertRaises(ValueError): count_summary(rows+rows, 10)
        result = count_summary(rows, 10)
        rows[0]['half_blend']['prediction'] = 10.5
        with self.assertRaises(ValueError): independent_counts(rows, result)

    def test_independent_metric_audit_rejects_wrong_summary(self):
        rows = self.rows(); result = count_summary(rows, 10)
        result['methods']['half_blend']['mae'] = 1
        with self.assertRaises(ValueError): independent_counts(rows, result)


if __name__ == '__main__':
    unittest.main()
