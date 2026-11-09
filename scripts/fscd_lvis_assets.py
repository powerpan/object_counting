"""Bounded CPU-only download and metadata inventory; never runs a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
import zipfile

from hpc4_port import sources, read, sha, write_new


def verify_entry(root):
    base = sources(root)
    receipt = read(root / 'provenance/fscd_lvis_assets_receipt_v1.json')
    if not receipt['remote_main_verified'] or receipt['base_code_commit'] != base['code_commit']:
        raise ValueError('Unpublished or mismatched entry')
    for name, expected in receipt['files_sha256'].items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or sha(root / name) != expected:
            raise ValueError('Entry source mismatch')
    return receipt


def metadata_members(infos):
    seen = set()
    selected = []
    for item in infos:
        path = PurePosixPath(item.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
            raise ValueError('Unsafe archive name')
        if item.filename in seen:
            raise ValueError('Duplicate archive member')
        seen.add(item.filename)
        mode = item.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError('Archive symlinks are not allowed')
        if not item.is_dir() and path.suffix.lower() == '.json':
            if item.file_size > 128 * 1024**2:
                raise ValueError('Unexpected metadata size')
            selected.append(item)
    if not selected or sum(x.file_size for x in selected) > 512 * 1024**2:
        raise ValueError('Missing or excessive metadata')
    return selected


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_JOB_PARTITION') != 'amd':
        raise RuntimeError('Submit this preparation to the CPU partition')
    receipt = verify_entry(root)
    cfg = read(root / 'configs/fscd_lvis_external_v1.json')
    out = root / 'results/fscd-lvis-assets-v1'
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if shutil.disk_usage(root).free < 32 * 1024**3:
            raise RuntimeError('Less than 32 GiB free for archive and verified ranges')
        archive = root / 'data/fscd-lvis-v1/FSCD_LVIS.zip'
        archive.parent.mkdir(parents=True, exist_ok=True)
        spec = cfg['archive']
        subprocess.run([sys.executable, str(root / 'scripts/download_verified_ranges.py'),
                        '--url', spec['url'], '--bytes', str(spec['bytes']),
                        '--sha256', spec['sha256'], '--output', str(archive),
                        '--workers', str(cfg['download_workers'])],
                       check=True, timeout=cfg['download_cpu_seconds'] - 120)
        if archive.stat().st_size != spec['bytes'] or sha(archive) != spec['sha256']:
            raise ValueError('Archive identity mismatch')
        inventory = []
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            for item in metadata_members(infos):
                data = z.read(item)
                parsed = json.loads(data)
                dest = archive.parent / 'metadata' / item.filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as stream:
                    stream.write(data)
                inventory.append({'member': item.filename, 'bytes': len(data),
                                  'sha256': hashlib.sha256(data).hexdigest(),
                                  'keys': list(parsed) if isinstance(parsed, dict) else None,
                                  'images': len(parsed.get('images', [])) if isinstance(parsed, dict) else None,
                                  'annotations': len(parsed.get('annotations', [])) if isinstance(parsed, dict) else None})
            images = [x for x in infos if PurePosixPath(x.filename).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            write_new(out / 'archive_inventory.json', {'files': len(infos), 'image_files': len(images),
                      'metadata': inventory, 'image_uncompressed_bytes': sum(x.file_size for x in images)})
        write_new(out / 'summary.json', {'status': 'passed', 'stage': 'archive and metadata only',
                  'source_commit': receipt['code_commit'], 'base_code_commit': receipt['base_code_commit'],
                  'archive_bytes': spec['bytes'], 'archive_sha256': spec['sha256'],
                  'mirror_revision': spec['mirror_revision'], 'official_archive_equivalence_verified': False,
                  'metadata_files': len(inventory), 'elapsed_seconds': time.monotonic() - started,
                  'gpu_used': False, 'next': 'validate annotation alignment and freeze evaluation manifest'})
        print('FSCD_LVIS_METADATA_READY', flush=True)
    except BaseException as exc:
        write_new(out / 'failure.json', {'status': 'failed', 'error_type': type(exc).__name__,
                  'elapsed_seconds': time.monotonic() - started, 'automatic_retry': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())
