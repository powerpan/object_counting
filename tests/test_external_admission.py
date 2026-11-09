import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fscd_lvis_admission import admission_seconds


class AdmissionTests(unittest.TestCase):
    def test_fixed_time_boundary_without_accuracy_input(self):
        boundary = (7200-300)*20/(1.5*2242)
        self.assertAlmostEqual(admission_seconds(boundary, 20, 2242), 7200)
        self.assertLess(admission_seconds(boundary-1, 20, 2242), 7200)
        self.assertGreater(admission_seconds(boundary+1, 20, 2242), 7200)

    def test_invalid_runtime_evidence_is_rejected(self):
        for seconds in (math.nan, math.inf, -1, 0):
            with self.assertRaises(ValueError): admission_seconds(seconds, 20, 2242)
        for sample_count in (0, -1, 1.2):
            with self.assertRaises(ValueError): admission_seconds(30, sample_count, 2242)


if __name__ == '__main__':
    unittest.main()
