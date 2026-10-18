import unittest
import numpy as np
from planb.geometry import select_scale_candidates, make_tiles, restore_tile_boxes


class GeometryTests(unittest.TestCase):
    def test_every_pixel_has_one_owner_with_odd_dimensions(self):
        tiles = make_tiles(101, 79)
        owners = np.zeros((79, 101), dtype=int)
        for tile in tiles:
            x0, y0, x1, y1 = tile['core']
            owners[y0:y1, x0:x1] += 1
            a, b, c, d = tile['crop']
            self.assertTrue(a <= x0 < x1 <= c and b <= y0 < y1 <= d)
        self.assertTrue((owners == 1).all())

    def test_restore_and_seam_ownership(self):
        tiles = make_tiles(100, 80)
        global_boxes = np.array([[45, 35, 55, 45], [44, 34, 54, 44]], dtype=np.float32)
        owner_count = np.zeros(2, dtype=int)
        for tile in tiles:
            local = (global_boxes - np.array(tile['crop'][:2] * 2)) * 2.5
            restored, keep = restore_tile_boxes(local, 2.5, tile)
            np.testing.assert_allclose(restored, global_boxes)
            owner_count += keep
        np.testing.assert_array_equal(owner_count, [1, 1])

    def test_only_novel_confident_scales_are_selected(self):
        cfg = dict(min_side_in_model_pixels=2, min_scale_ratio=1.5,
                   min_relative_objectness=.5, min_mean_cosine=.75, max_extra_exemplars=3)
        original = [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]]
        boxes = [[0, 0, s, s] for s in [10, 5, 5.2, 20, 40, 1]]
        chosen = select_scale_candidates(boxes, original, [1, .9, .8, .9, .9, 1],
                                         [1, .9, .9, .2, .9, 1], cfg)
        self.assertEqual(chosen, [1, 4])
        self.assertEqual(select_scale_candidates([], original, [], [], cfg), [])


if __name__ == '__main__':
    unittest.main()
