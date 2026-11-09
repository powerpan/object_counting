import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hpc4_port', ROOT / 'scripts/hpc4_port.py')
PORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PORT)


class MigrationSafetyTests(unittest.TestCase):
    def test_receipt_rejects_source_mutation_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'provenance').mkdir()
            source = root / 'entry.py'
            source.write_text('original')
            receipt = {'remote_main_verified': True, 'files_sha256': {
                'entry.py': hashlib.sha256(b'original').hexdigest()}}
            rp = root / 'provenance/hpc4_source_receipt_v1.json'
            rp.write_text(json.dumps(receipt))
            PORT.sources(root)
            source.write_text('modified')
            with self.assertRaises(ValueError):
                PORT.sources(root)
            receipt['files_sha256'] = {'../outside.py': 'unused'}
            rp.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                PORT.sources(root)

    def test_existing_evidence_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'evidence.json'
            PORT.write_new(path, {'status': 'first'})
            with self.assertRaises(FileExistsError):
                PORT.write_new(path, {'status': 'replacement'})
            self.assertEqual(json.loads(path.read_text())['status'], 'first')

    def test_only_registered_validation_assets_are_selected(self):
        records = PORT.asset_records(ROOT)
        images = [x for x in records if x.get('kind') == 'image']
        self.assertEqual({Path(x['path']).name for x in images}, {'190.jpg', '191.jpg', '809.jpg'})
        self.assertEqual(len(images), 3)
        self.assertEqual(len(records), 8)
        self.assertTrue(all(x['url'].startswith('https://') for x in records))


if __name__ == '__main__':
    unittest.main()
