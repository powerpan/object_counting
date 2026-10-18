"""Run the bounded B0 checks only after verified installation and asset transfer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-minutes', type=int, default=90)
    args = parser.parse_args()
    root = args.root.resolve()
    results = root / 'results'
    results.mkdir(exist_ok=True)
    lock = (results / '.b0.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_file = results / 'b0_workflow.json'

    def status(stage, **values):
        record = {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}
        tmp = status_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(status_file)
        print(json.dumps(record), flush=True)

    stage = 'waiting_for_environment_and_assets'
    try:
        status(stage)
        direct_manifest = root / 'provenance/validation_manifest.json'
        use_direct = direct_manifest.exists()
        required = [json.loads((root / 'provenance' / f'{p}.json').read_text())[0]
                    for p in (['geco', 'sam'] if use_direct else ['geco', 'sam', 'validation_bundle'])]
        verified = set()
        deadline = time.monotonic() + args.wait_minutes * 60
        while time.monotonic() < deadline:
            exit_file = root / 'logs/bootstrap.exit'
            if exit_file.exists() and exit_file.read_text().strip() != '0':
                raise RuntimeError('Environment installation failed; inspect private bootstrap log')
            env_ready = exit_file.exists() and 'BOOTSTRAP_COMPLETE' in (root / 'logs/bootstrap.log').read_text()
            for asset in required:
                path = root / asset['path']
                if str(path) in verified:
                    continue
                candidates = [path, path.with_suffix(path.suffix + '.partial')]
                for candidate in candidates:
                    if candidate.exists() and candidate.stat().st_size == asset['bytes']:
                        if sha(candidate) != asset['sha256']:
                            raise ValueError('Asset hash mismatch: ' + asset['path'])
                        if candidate != path:
                            candidate.replace(path)
                        verified.add(str(path))
                        break
            data_ready = not use_direct or (root / 'provenance/validation_download.json').exists()
            if env_ready and data_ready and len(verified) == len(required):
                break
            time.sleep(15)
        else:
            raise TimeoutError('Preparation deadline exceeded; no evaluation started')

        if use_direct:
            stage = 'verifying_downloaded_validation_inputs'
            status(stage)
            receipt = json.loads((root / 'provenance/validation_download.json').read_text())
            manifest = json.loads(direct_manifest.read_text())
            if receipt['status'] != 'complete' or receipt['manifest_sha256'] != sha(direct_manifest):
                raise ValueError('Validation download receipt does not match manifest')
            if manifest['split'] != 'val' or manifest['n'] != 1286:
                raise ValueError('Unexpected validation manifest')
            for record in manifest['files']:
                path = root / record['path']
                if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
                    raise ValueError('Validation input changed after download')
        else:
            stage = 'extracting_validation_inputs'
            status(stage)
            with tarfile.open(root / 'validation-inputs.tar.gz') as archive:
                for member in archive.getmembers():
                    path = Path(member.name)
                    if not member.isfile() or path.is_absolute() or '..' in path.parts or not member.name.startswith('data/fsc147/'):
                        raise ValueError('Unexpected archive member')
                archive.extractall(root)
        python = str(root / '.venv/bin/python')
        env = dict(os.environ, TORCH_HOME=str(root / 'cache/torch'), PYTHONUNBUFFERED='1')

        def run(stage_name, command, timeout):
            nonlocal stage
            stage = stage_name
            status(stage)
            with (root / 'logs' / f'{stage}.log').open('x') as log:
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=timeout)

        run('cuda_operator_check', [python, str(root / 'scripts/check_cuda_ops.py'),
            '--upstream', str(root / 'external/GECO2'), '--output', str(results / 'cuda_ops.json')], 300)
        run('m1_unit_tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 60)
        runner = [python, str(root / 'scripts/evaluate_val.py'), '--root', str(root)]
        run('b0_smoke_b1', runner + ['--output', str(results / 'b0-smoke-b1'), '--limit', '20', '--batch-size', '1'], 900)
        smoke = json.loads((results / 'b0-smoke-b1/summary.json').read_text())
        if smoke['status'] != 'complete' or smoke['overall']['n'] != 20:
            raise RuntimeError('Smoke evaluation incomplete')
        # Full validation uses the measured batch 1 setting. Larger batches and
        # training are separate experiments; a failure is never silently retried.
        run('b0_full_validation', runner + ['--output', str(results / 'b0-val-b1'), '--limit', '0', '--batch-size', '1'], 5400)
        summary = json.loads((results / 'b0-val-b1/summary.json').read_text())
        if summary['status'] != 'complete' or summary['overall']['n'] != 1286:
            raise RuntimeError('Full validation incomplete')
        status('complete', scope='B0 full validation and M1 checks; no training')
    except Exception as error:
        # Raw tracebacks and paths stay in private logs; publish only a checked
        # diagnostic summary after inspecting the failed stage.
        status('failed', failed_stage=stage, error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
