"""Download only official validation inputs and verify each against a manifest.

The mirror is a transport source, not the authority for dataset identity.
Expected SHA-256 values are computed from the official FSC-147 archive/files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

import requests


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--mirror', default='https://hf-mirror.com')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest['split'] != 'val' or manifest['n'] != 1286 or sum(r['kind'] == 'image' for r in manifest['files']) != 1286:
        raise ValueError('Expected exactly the official validation split')
    for record in manifest['files']:
        path = Path(record['path'])
        if path.is_absolute() or '..' in path.parts or not str(path).startswith('data/fsc147/'):
            raise ValueError('Invalid input path in manifest')
    local = threading.local()

    def download(record):
        dest = args.root / record['path']
        if dest.is_file() and dest.stat().st_size == record['bytes'] and digest(dest) == record['sha256']:
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        if record['kind'] == 'image':
            url = args.mirror.rstrip('/') + '/datasets/isentropic/FSC147/resolve/' + manifest['mirror_revision'] + '/images_384_VarV2/' + dest.name
        else:
            url = record['source']
        for attempt in range(3):
            try:
                with local.session.get(url, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    partial = dest.with_suffix(dest.suffix + '.partial')
                    with partial.open('wb') as stream:
                        for block in response.iter_content(1024 * 1024):
                            stream.write(block)
                if partial.stat().st_size != record['bytes'] or digest(partial) != record['sha256']:
                    raise ValueError('Content differs from official source: ' + record['path'])
                partial.replace(dest)
                return
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(download, record) for record in manifest['files']]
        for done, future in enumerate(as_completed(tasks), 1):
            future.result()
            if done % 100 == 0 or done == len(tasks):
                print(json.dumps({'verified_files': done, 'total': len(tasks),
                                  'elapsed_seconds': round(time.monotonic() - started, 1)}), flush=True)
    report = {'status': 'complete', 'verified_files': len(tasks), 'validation_images': 1286,
              'manifest_sha256': digest(args.manifest), 'transport': args.mirror,
              'mirror_revision': manifest['mirror_revision'],
              'elapsed_seconds': time.monotonic() - started}
    output = args.root / 'provenance/validation_download.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
