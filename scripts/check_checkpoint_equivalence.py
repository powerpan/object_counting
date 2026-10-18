"""Verify author HF weights against tensor digests of the README's Drive file."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    expected_sha = '945c5d838a058aa970582291b12325e0819bbf445768f433162b2c8b608f512b'
    actual_sha = digest(args.checkpoint)
    if actual_sha != expected_sha:
        raise ValueError('File differs from pinned author Hugging Face LFS object')
    expected = json.loads(args.manifest.read_text())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = saved.get('model', saved)
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    differences = []
    if set(state) != set(expected['tensors']):
        differences.append({'missing': sorted(set(expected['tensors']) - set(state)),
                            'extra': sorted(set(state) - set(expected['tensors']))})
    for key in sorted(set(state) & set(expected['tensors'])):
        tensor = state[key].detach().cpu().contiguous()
        original = expected['tensors'][key]
        if (tensor.dtype != torch.float32 or list(tensor.shape) != original['shape'] or
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest() != original['sha256']):
            differences.append({'tensor': key})
    result = {'status': 'passed' if not differences else 'failed',
              'hf_revision': 'ed3c8ff3753e731fd7074862c0ea49d908785335',
              'hf_file_sha256': actual_sha, 'original_file_sha256': expected['original_file_sha256'],
              'checked_tensors': len(state), 'differences': differences,
              'scope': 'Exact float32 model tensor equality; optimizer/metadata excluded'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if differences:
        raise ValueError('Author checkpoints are not model-equivalent; stop baseline execution')


if __name__ == '__main__':
    main()
