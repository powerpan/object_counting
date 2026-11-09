import sys
from pathlib import Path
import stat
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fscd_lvis_assets import metadata_members


class MetadataSafetyTest(unittest.TestCase):
    def test_selects_only_metadata(self):
        entries = [zipfile.ZipInfo('FSCD_LVIS/annotations/count_test.json'),
                   zipfile.ZipInfo('FSCD_LVIS/images/one.jpg')]
        self.assertEqual(metadata_members(entries), entries[:1])

    def test_rejects_paths_duplicates_and_links(self):
        for name in ['../count.json', '/count.json', 'folder\\count.json']:
            with self.assertRaises(ValueError):
                metadata_members([zipfile.ZipInfo(name)])
        item = zipfile.ZipInfo('count.json')
        with self.assertRaises(ValueError):
            metadata_members([item, item])
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError):
            metadata_members([item])

    def test_rejects_unexpected_size_or_missing_metadata(self):
        item = zipfile.ZipInfo('count.json')
        item.file_size = 129 * 1024**2
        with self.assertRaises(ValueError):
            metadata_members([item])
        with self.assertRaises(ValueError):
            metadata_members([zipfile.ZipInfo('image.jpg')])


if __name__ == '__main__':
    unittest.main()
