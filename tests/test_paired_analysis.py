from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_m2_m3 import paired_intervals


class PairedAnalysisTests(unittest.TestCase):
    def test_identical_predictions_have_zero_paired_interval(self):
        rows = [{'image': str(i), 'truth': 10, 'prediction': i} for i in range(6)]
        ref = {r['image']: r for r in rows}
        labels = {str(i): 'a' if i < 4 else 'b' for i in range(6)}
        for unit in ('image', 'class'):
            result = paired_intervals(rows, ref, labels, 100, unit)
            self.assertEqual(result['mae_delta_95ci'], [0., 0.])
            self.assertEqual(result['rmse_delta_95ci'], [0., 0.])

    def test_class_point_estimate_keeps_image_weighting(self):
        rows = [{'image': 'a1', 'truth': 10, 'prediction': 10},
                {'image': 'a2', 'truth': 10, 'prediction': 10},
                {'image': 'b1', 'truth': 10, 'prediction': 13}]
        ref = {r['image']: dict(r, prediction=10) for r in rows}
        result = paired_intervals(rows, ref, {'a1': 'a', 'a2': 'a', 'b1': 'b'}, 100, 'class')
        self.assertEqual(result['mae_delta'], 1.)
        self.assertEqual(result['units'], 2)


if __name__ == '__main__':
    unittest.main()
