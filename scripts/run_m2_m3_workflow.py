"""One bounded, locked run of gate -> smoke -> full validation; never trains."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    args = p.parse_args(); root = args.root.resolve()
    lock = (root / 'results/.m2-m3.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    result = root / 'results/m2-m3-v1'
    result.mkdir(exist_ok=True)
    status_file = root / 'results/m2-m3-workflow.json'
    config = json.loads((root / 'configs/m2_m3_v1.json').read_text())
    python = str(root / '.venv/bin/python')
    env = dict(os.environ, PYTHONUNBUFFERED='1', TORCH_HOME=str(root / 'cache/torch'))
    stage = 'unit_tests'

    def status(state, **kwargs):
        record = {'state': state, 'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs}
        tmp = status_file.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2) + '\n'); tmp.replace(status_file)
        print(json.dumps(record), flush=True)

    def execute(command, timeout):
        status('running')
        with (root / 'logs' / ('m2-m3-' + stage + '.log')).open('x') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)

    try:
        execute([python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 60)
        runner = [python, str(root / 'scripts/run_m2_m3.py'), '--root', str(root)]
        stage = 'gate'
        execute(runner + ['--phase', 'gate', '--output', str(result / 'gate')], 900)
        gate = json.loads((result / 'gate/gate.json').read_text())
        assert gate['status'] == 'passed' and gate['n'] == 32
        for phase in ['smoke', 'full']:
            for variant in config['variants']:
                stage = ('val' if phase == 'full' else phase) + '-' + variant
                output = result / stage
                execute(runner + ['--phase', phase, '--variant', variant, '--output', str(output)],
                        config['stop']['variant_timeout_seconds'] if phase == 'full' else 900)
                summary = json.loads((output / 'summary.json').read_text())
                assert summary['status'] == 'complete' and summary['overall']['n'] == (1286 if phase == 'full' else 32)
        stage = 'complete'
        status('complete', scope='Frozen M2/M3 validation only; independent audit and publication still required')
    except Exception as error:
        status('failed', error_type=type(error).__name__)
        raise


if __name__ == '__main__':
    main()
