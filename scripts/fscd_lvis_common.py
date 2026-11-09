"""Explicit metadata joins and provenance for the fixed external evaluation."""
import json
from pathlib import Path, PurePosixPath
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fscd_lvis_assets import verify_entry
from hpc4_port import read, sha, write_new
from planb.density import points_for_game


def checked_sources(root, name):
    verify_entry(root)
    receipt = read(root / 'provenance' / name)
    if not receipt['remote_main_verified']:
        raise ValueError('Entry was not published')
    for name, expected in receipt['files_sha256'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or sha(root/name) != expected:
            raise ValueError('Source differs from published entry')
    return receipt


def keyed(records, key):
    result = {}
    for record in records:
        value = record[key]
        if not isinstance(value, int) or isinstance(value, bool) or value in result:
            raise ValueError('Invalid or duplicate identifier')
        result[value] = record
    return result


def align_annotations(count, instances, split):
    count_images = keyed(count['images'], 'id')
    count_annotations = keyed(count['annotations'], 'id')
    instance_images = keyed(instances['images'], 'id')
    if count_images.keys() != count_annotations.keys():
        raise ValueError('Count-image and count-annotation IDs differ')
    boxes = {key: [] for key in instance_images}
    keyed(instances['annotations'], 'id')
    for annotation in instances['annotations']:
        if annotation['image_id'] not in boxes:
            raise ValueError('Unmapped instance annotation')
        boxes[annotation['image_id']].append(annotation)
    records = {}
    for count_id, annotation in count_annotations.items():
        original = count_images[count_id]
        image_id = annotation['image_id']
        if image_id in records or image_id not in instance_images:
            raise ValueError('Ambiguous target image ID')
        instance = instance_images[image_id]
        filename = original['file_name']
        rel = PurePosixPath(filename)
        if rel.is_absolute() or '..' in rel.parts or '\\' in filename:
            raise ValueError('Unsafe image filename')
        if instance['file_name'] != filename or annotation.get('file_name', filename) != filename:
            raise ValueError('Image filename disagreement')
        width, height = int(original['width']), int(original['height'])
        if (width, height) != (instance['width'], instance['height']) or min(width, height) < 1:
            raise ValueError('Image size disagreement')
        prompts = np.asarray(annotation['boxes'][:3], dtype=np.float64)
        if prompts.shape != (3, 4) or not np.isfinite(prompts).all():
            raise ValueError('Missing or nonfinite official exemplars')
        prompts[:, 2:] += prompts[:, :2]
        if not ((prompts[:, 2:] > prompts[:, :2]).all() and (prompts >= 0).all()
                and (prompts[:, [0, 2]] <= width).all() and (prompts[:, [1, 3]] <= height).all()):
            raise ValueError('Official exemplar outside image or empty')
        points = points_for_game(annotation['points'], height, width)
        target = boxes[image_id]
        if len(points) != len(target) or len(points) < 3:
            raise ValueError('Point/instance target count disagreement')
        categories = sorted({x['category_id'] for x in target})
        if len(categories) != 1:
            raise ValueError('Multiple target categories in one record')
        records[image_id] = {'sample_id': f'{split}-{image_id}', 'image_id': image_id,
                             'file_name': filename, 'width': width, 'height': height,
                             'exemplars_xyxy': prompts.tolist(), 'points': points.tolist(),
                             'truth': len(points), 'category_id': categories[0]}
    if records.keys() != instance_images.keys():
        raise ValueError('Unpaired instance images')
    return [records[key] for key in sorted(records)]


def annotation_path(root, inventory, filename):
    found = [r for r in inventory['metadata'] if PurePosixPath(r['member']).name == filename]
    if len(found) != 1:
        raise ValueError('Required annotation is missing or ambiguous')
    record = found[0]
    path = root / 'data/fscd-lvis-v1/metadata' / record['member']
    if sha(path) != record['sha256']:
        raise ValueError('Metadata changed after extraction')
    return path
