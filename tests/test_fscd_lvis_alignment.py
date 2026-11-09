import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fscd_lvis_common import align_annotations


def fixture():
    count = {'images': [{'id': 9, 'file_name': 'one.jpg', 'width': 100, 'height': 80}],
             'annotations': [{'id': 9, 'image_id': 21, 'file_name': 'one.jpg',
                              'boxes': [[1.5, 2, 5, 6], [10, 10, 5, 6], [20, 20, 5, 6]],
                              'points': [[3, 4], [12, 12], [22, 22]]}]}
    instances = {'images': [{'id': 21, 'file_name': 'one.jpg', 'width': 100, 'height': 80}],
                 'annotations': [{'id': i, 'image_id': 21, 'category_id': 6} for i in [1, 2, 3]]}
    return count, instances


class AlignmentTests(unittest.TestCase):
    def test_joins_count_id_to_instance_id_and_preserves_floats(self):
        c, i = fixture()
        result = align_annotations(c, i, 'test')[0]
        self.assertEqual(result['image_id'], 21)
        self.assertEqual(result['exemplars_xyxy'][0], [1.5, 2., 6.5, 8.])
        self.assertEqual(result['truth'], 3)

    def test_wrong_file_count_class_or_id_is_rejected(self):
        for field in ['file', 'count', 'class', 'id']:
            c, i = fixture()
            if field == 'file': i['images'][0]['file_name'] = 'wrong.jpg'
            if field == 'count': i['annotations'].pop()
            if field == 'class': i['annotations'][0]['category_id'] = 8
            if field == 'id': c['annotations'][0]['id'] = 10
            with self.assertRaises(ValueError):
                align_annotations(c, i, 'test')

    def test_outside_prompt_or_point_is_rejected(self):
        c, i = fixture(); c['annotations'][0]['boxes'][0] = [98, 2, 5, 6]
        with self.assertRaises(ValueError): align_annotations(c, i, 'test')
        c, i = fixture(); c['annotations'][0]['points'][0] = [102, 2]
        with self.assertRaises(ValueError): align_annotations(c, i, 'test')


if __name__ == '__main__':
    unittest.main()
