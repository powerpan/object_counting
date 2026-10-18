"""Fetch public author assets; keep binaries outside Git and record provenance."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import gdown


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--part', choices=['geco', 'sam', 'images', 'annotations'], required=True)
    args = parser.parse_args()
    assets = {
        'geco': [('weights/GECO2FSCD.pth', '1OG6-tTP2Egvx-O5oK6lxPzvQO1LWgHbb')],
        'sam': [('cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt',
                 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt')],
        'images': [('data/fsc147/images.zip', '1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S')],
        'annotations': [('data/fsc147/annotations/' + name,
            'https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/' + name)
            for name in ['annotation_FSC147_384.json', 'Train_Test_Val_FSC_147.json', 'ImageClasses_FSC147.txt']],
    }
    records = []
    for relative, source in assets[args.part]:
        path = args.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        url = source if source.startswith('https://') else 'https://drive.google.com/file/d/' + source + '/view'
        if not path.exists():
            partial = path.with_suffix(path.suffix + '.partial')
            if source.startswith('https://'):
                with urllib.request.urlopen(source, timeout=60) as response, partial.open('wb') as f:
                    while block := response.read(4 * 1024 * 1024):
                        f.write(block)
            else:
                result = gdown.download(id=source, output=str(partial), quiet=True,
                                        use_cookies=False, resume=True, timeout=(20, 90), retries=1)
                if result is None:
                    raise RuntimeError('Official asset unavailable: ' + relative)
            partial.replace(path)
        record = {'path': relative, 'source': url, 'bytes': path.stat().st_size,
                  'sha256': sha256(path)}
        records.append(record)
        print(json.dumps(record), flush=True)
    dest = args.root / 'provenance'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / (args.part + '.json')).write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
