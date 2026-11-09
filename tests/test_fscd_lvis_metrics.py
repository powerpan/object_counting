import copy
import unittest
from planb.fscd_lvis_metrics import report


def row(i, b, f, unit=None):
    value = {'sample_id': str(i), 'sampling_unit': str(i) if unit is None else unit,
             'truth': 10, 'route': 'T2'}
    for name, count in [('b0', b), ('final', f)]:
        value[name] = {'prediction': count, 'game1': 1., 'game2': 2.,
                       'inference_seconds': .3, 'peak_allocated_mib': 4000.}
    return value


class ExternalMetricsTests(unittest.TestCase):
    def test_exact_improvement_and_reversal(self):
        rows = [row(i, 12, 11) for i in range(4)]
        r = report(rows, replicates=300)
        self.assertEqual(r['b0']['mae'], 2)
        self.assertEqual(r['final']['rmse'], 1)
        self.assertEqual(r['paired']['mae_delta_95ci'], [-1, -1])
        self.assertTrue(r['both_count_intervals_below_zero'])
        reversed_rows = copy.deepcopy(rows)
        for item in reversed_rows:
            item['b0'], item['final'] = item['final'], item['b0']
        reverse = report(reversed_rows, replicates=300)
        self.assertEqual(reverse['paired']['rmse_delta_95ci'], [1, 1])
        self.assertFalse(reverse['both_count_point_estimates_improved'])

    def test_repeated_source_is_one_unit_and_ties_do_not_pass(self):
        r = report([row(1, 9, 11, 'same'), row(2, 8, 12, 'same')], replicates=300)
        self.assertEqual(r['unique_source_images'], 1)
        self.assertEqual(r['paired']['mae_delta_95ci'], [0, 0])
        self.assertFalse(r['both_count_intervals_below_zero'])

    def test_invalid_samples_rejected(self):
        with self.assertRaises(ValueError):
            report([row(1, 12, float('nan'))])
        with self.assertRaises(ValueError):
            report([row(1, 12, 11), row(1, 12, 11)])


if __name__ == '__main__':
    unittest.main()
