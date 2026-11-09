"""Prepare a private image/annotation manifest without model inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile
from PIL import Image
from fscd_lvis_common import checked_sources, align_annotations, annotation_path, read, sha, write_new


def prepare(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'amd':
        raise RuntimeError('Use a CPU allocation')
    receipt = checked_sources(root, 'fscd_lvis_prepare_receipt_v1.json')
    cfg = read(root / 'configs/fscd_lvis_external_v1.json')
    assets = read(root / 'results/fscd-lvis-assets-v1/summary.json')
    if assets['status'] != 'passed' or assets['archive_sha256'] != cfg['archive']['sha256']:
        raise ValueError('Asset verification has not passed')
    out = root / 'results/fscd-lvis-preparation-v1'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        inventory = read(root / 'results/fscd-lvis-assets-v1/archive_inventory.json')
        splits, metadata_hashes = {}, {}
        for split in ('train', 'test'):
            c = annotation_path(root, inventory, f'unseen_count_{split}.json')
            i = annotation_path(root, inventory, f'unseen_instances_{split}.json')
            splits[split] = align_annotations(read(c), read(i), split)
            metadata_hashes[c.name], metadata_hashes[i.name] = sha(c), sha(i)
        train_classes = {r['category_id'] for r in splits['train']}
        test_classes = {r['category_id'] for r in splits['test']}
        shared_classes = train_classes & test_classes
        if shared_classes:
            raise ValueError('Declared unseen split has shared target category IDs')
        train_names = {r['file_name'] for r in splits['train']}
        test_names = {r['file_name'] for r in splits['test']}
        archive = root / 'data/fscd-lvis-v1/FSCD_LVIS.zip'
        if sha(archive) != cfg['archive']['sha256']:
            raise ValueError('Archive changed')
        selected = {'smoke': splits['train'][:cfg['smoke']['images']], 'full': splits['test']}
        fsc_hashes = set()
        for name in ('fsc147_train.json', 'fsc147_val.json', 'fsc147_test_v1.json'):
            for entry in read(root/'manifests'/name)['files']:
                if entry.get('kind') == 'image' or str(entry.get('path', '')).lower().endswith('.jpg'):
                    fsc_hashes.add(entry['sha256'])
        image_cache = {}
        with zipfile.ZipFile(archive) as z:
            members = {item.filename: item for item in z.infolist() if not item.is_dir()}
            for records in selected.values():
                for record in records:
                    filename = record['file_name']
                    if filename not in image_cache:
                        # Match the cited reader's images/<file_name> path exactly.
                        # The mirror also has train/test subdirectories with duplicates.
                        member = str(PurePosixPath('FSCD_LVIS/images') / filename)
                        if member not in members:
                            raise ValueError('Required canonical image missing')
                        data = z.read(members[member])  # ZIP CRC is checked while reading.
                        digest = hashlib.sha256(data).hexdigest()
                        relative = Path('data/fscd-lvis-v1/images') / filename
                        path = root/relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('xb') as stream:
                            stream.write(data)
                        with Image.open(path) as image:
                            image.load()
                            size = image.size
                        image_cache[filename] = {'image_path': str(relative), 'image_sha256': digest,
                                                 'actual_size': list(size), 'exact_fsc_match': digest in fsc_hashes}
                    record.update(image_cache[filename])
                    if record['actual_size'] != [record['width'], record['height']]:
                        raise ValueError('Decoded image dimensions disagree with metadata')
        manifest = {'version': cfg['version'], 'metadata_sha256': metadata_hashes,
                    'selection': cfg['smoke']['selection'], 'splits': selected}
        manifest_path = root/'data/fscd-lvis-v1/evaluation_manifest.json'
        write_new(manifest_path, manifest)
        summary = {'status': 'passed', 'source_commit': receipt['code_commit'],
                   'manifest_sha256': sha(manifest_path), 'metadata_sha256': metadata_hashes,
                   'train_records': len(splits['train']), 'test_records': len(splits['test']),
                   'smoke_records': len(selected['smoke']), 'target_class_overlap_train_test': len(shared_classes),
                   'same_filename_train_test': len(train_names & test_names),
                   'unique_extracted_images': len(image_cache), 'fsc_reference_hashes': len(fsc_hashes),
                   'test_exact_fsc_image_matches': sum(r['exact_fsc_match'] for r in selected['full']),
                   'test_unique_source_images': len({r['image_sha256'] for r in selected['full']}),
                   'test_objects': sum(r['truth'] for r in selected['full']),
                   'elapsed_seconds': time.monotonic()-start, 'gpu_used': False,
                   'predictions_made': False, 'public_manifest_contains_annotations': False}
        write_new(out/'summary.json', summary)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_new(out/'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'message': str(exc), 'elapsed_seconds': time.monotonic()-start, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    prepare(a.root.resolve())
