"""Fetch a public object directly with bounded HTTP ranges and a final SHA-256."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time

import requests


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--bytes', type=int, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-prefix', type=Path)
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists():
        if a.output.stat().st_size == a.bytes and digest(a.output) == a.sha256:
            print('ALREADY_VERIFIED', flush=True)
            return
        raise FileExistsError('Preserve or remove the different destination before retrying')
    prefix_size = a.resume_prefix.stat().st_size if a.resume_prefix else 0
    if not 0 <= prefix_size <= a.bytes or not 1 <= a.workers <= 16:
        raise ValueError('Invalid prefix size or worker limit')
    parts = a.output.with_suffix(a.output.suffix + '.ranges')
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(start, min(start + 8 * 1024 * 1024, a.bytes) - 1)
            for start in range(prefix_size, a.bytes, 8 * 1024 * 1024)]

    def fetch(bounds):
        start, end = bounds
        dest = parts / f'{start:012d}-{end:012d}'
        if dest.exists() and dest.stat().st_size == end - start + 1:
            return dest
        for attempt in range(3):
            try:
                response = requests.get(a.url, headers={'Range': f'bytes={start}-{end}'}, timeout=(10, 90))
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{a.bytes}':
                    raise ValueError('Server did not return the exact requested range')
                if len(response.content) != end - start + 1:
                    raise ValueError('Incomplete range response')
                partial = dest.with_suffix('.partial')
                partial.write_bytes(response.content)
                partial.replace(dest)
                return dest
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending = [pool.submit(fetch, job) for job in jobs]
        for count, future in enumerate(as_completed(pending), 1):
            future.result()
            print(f'Ranges verified structurally: {count}/{len(jobs)}', flush=True)
    partial = a.output.with_suffix(a.output.suffix + '.assembling')
    with partial.open('wb') as out:
        paths = ([a.resume_prefix] if a.resume_prefix else []) + [parts / f'{start:012d}-{end:012d}' for start, end in jobs]
        for path in paths:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    out.write(block)
    if partial.stat().st_size != a.bytes or digest(partial) != a.sha256:
        raise ValueError('Final object hash mismatch; output not promoted')
    partial.replace(a.output)
    print('SHA256_VERIFIED ' + a.sha256, flush=True)


if __name__ == '__main__':
    main()
