"""After both training heads finish, validate and audit seed 42 sequentially.

No training, retry, test inference, or automatic adoption of partial outputs.
This entry deliberately imports only the standard library and budget ledger.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.round2.budget import BudgetLedger, seconds


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def exclusive(path):
    with Path(path).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def verify_receipts(root):
    receipts = {}
    for name in ('round2_code_receipt', 'round2_budget_receipt',
                 'round2_analysis_receipt', 'round2_validation_workflow_receipt'):
        path = root/'provenance'/(name + '.json')
        receipt = read(path)
        if receipt.get('remote_main_verified') is not True:
            raise ValueError('Entrypoint has not been verified on main')
        for relative, expected in receipt['files_sha256'].items():
            if sha(root/relative) != expected:
                raise ValueError('Committed source receipt mismatch')
        receipts[name] = {'code_commit': receipt['code_commit'], 'receipt_sha256': sha(path)}
    if not read(root/'provenance/round2_code_receipt.json')['registration_commit'].startswith('ce0fc1a'):
        raise ValueError('Missing original preregistration')
    if not read(root/'provenance/round2_budget_receipt.json')['amendment_commit'].startswith('ee4a066'):
        raise ValueError('Missing approved time amendment')
    return receipts


def check_training(root, cfg, amendment, workflow):
    """Recompute checkpoint selection from all 30 holdout epochs before val."""
    evidence = {}
    per_head_seconds = 0.
    for width in ('S', 'L'):
        label = 'head-' + width + '-seed42'
        folder = root/'results/round2-v1'/label
        if any((folder/name).exists() for name in ('error.json', 'failure.json')):
            raise ValueError('Training has an unresolved failure')
        status, manifest, history = [read(folder/name) for name in
                                       ('status.json', 'checkpoint_manifest.json', 'history.json')]
        epochs = cfg['m4']['epochs']
        if status['status'] != 'complete' or status['epoch'] != epochs or manifest['complete_epochs'] != epochs:
            raise ValueError('Training did not complete all registered epochs')
        if [r['epoch'] for r in history] != list(range(1, epochs + 1)):
            raise ValueError('Missing or repeated training epoch')
        previous = 0.
        for row in history:
            if row['train_n'] != 3457 or row['holdout_n'] != 202:
                raise ValueError('Unexpected training split size')
            for key in ('train_loss', 'holdout_loss', 'holdout_mae', 'elapsed_seconds'):
                seconds(row[key])
            if row['elapsed_seconds'] < previous or row['frozen_state_sha256'] != manifest['frozen_model_sha256']:
                raise ValueError('Invalid timing or changed frozen model')
            previous = row['elapsed_seconds']
        best = min(history, key=lambda row: (row['holdout_mae'], row['epoch']))
        for record in (status, manifest):
            if record['best_epoch'] != best['epoch'] or record['best_holdout_mae'] != best['holdout_mae']:
                raise ValueError('Checkpoint selection differs from registered holdout rule')
        elapsed = seconds(status['elapsed_seconds'])
        if elapsed < previous or elapsed > amendment['m4_' + width + '_single_run_seconds']:
            raise ValueError('Completed training exceeds its approved cap')
        per_head_seconds += elapsed
        for kind in ('best', 'last'):
            if sha(root/'work/round2-v1'/label/(kind+'.pth')) != manifest[kind+'_sha256']:
                raise ValueError('Checkpoint content differs from manifest')
        evidence[label] = {name: sha(folder/name) for name in
                           ('status.json', 'checkpoint_manifest.json', 'history.json')}
    previous_file = root/'results/round2-v1/workflow_before_budget_amendment.json'
    previous_workflow = read(previous_file)
    if previous_workflow['stage'] != 'head-S-seed42' or previous_workflow['state'] != 'running':
        raise ValueError('Unexpected pre-amendment handoff record')
    preflight = seconds(previous_workflow['spent_seconds'])
    if seconds(workflow['spent_seconds']) + 1e-6 < preflight + per_head_seconds:
        raise ValueError('Training controller undercounts its recorded training time')
    evidence['workflow_before_budget_amendment.json'] = sha(previous_file)
    return evidence


def completed_evidence(root, label, kind, expected_n=1286):
    folder = root/'results/round2-v1'/label
    if any((folder/name).exists() for name in ('error.json', 'audit_error.json')):
        raise ValueError('Result contains an unresolved failure')
    if kind == 'audit':
        result = read(folder/'audit.json')
        if result['status'] != 'passed' or result['n'] != expected_n:
            raise ValueError('Independent audit has not passed')
        return {label+'/audit.json': sha(folder/'audit.json')}
    result = read(folder/'summary.json')
    n = result['n'] if kind == 'crossfit' else result['overall']['n']
    if result['status'] != 'complete' or n != expected_n:
        raise ValueError('Incomplete inference result')
    return {label+'/summary.json': sha(folder/'summary.json'),
            label+'/protocol.json': sha(folder/'protocol.json')}


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    # Every stage owns a separate process group; no signal touches the previous
    # training controller or another job. The 10 s allowance is reserved below.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=3)


def run_stage(root, ledger, state, state_file, stage, cap, command, verify):
    started = time.monotonic()
    limit = ledger.begin(stage, cap, utc())
    state.update(state='running', stage=stage, updated_utc=utc())
    write(state_file, state)
    proc = None
    passed, evidence = False, {}
    try:
        # Ten seconds belong to shutdown and settlement, not additional compute.
        child_limit = limit - 10
        with (root/'logs'/('round2-' + stage + '.log')).open('x') as log:
            proc = subprocess.Popen(command(child_limit), cwd=root, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            state['stage_pid'] = proc.pid
            write(state_file, state)
            code = proc.wait(timeout=max(.001, child_limit - (time.monotonic() - started)))
        if code:
            raise RuntimeError('Stage exited unsuccessfully')
        evidence = verify()
        passed = True
    finally:
        stop_process(proc)
        accepted = ledger.finish(stage, time.monotonic() - started, passed, evidence, utc())
        state.pop('stage_pid', None)
        state.update(spent_seconds=ledger.spent, remaining_seconds=ledger.remaining, updated_utc=utc())
        write(state_file, state)
    if not accepted:
        raise TimeoutError('Stage or aggregate runtime cap reached')
    state['completed'].append(stage)
    write(state_file, state)


def workflow(root):
    result = root/'results/round2-v1'
    state_file = result/'validation_workflow_v1.json'
    ledger_file = result/'m4_budget_ledger_v1.json'
    with exclusive(root/'results/.round2-m4.lock'):
        started = time.monotonic()
        if state_file.exists() or ledger_file.exists():
            raise FileExistsError('Inspect existing validation workflow; automatic restart is disabled')
        training_file = result/'workflow_budget_v1.json'
        training = read(training_file)
        # Check readiness before writing anything. A harmless early invocation
        # must not create stopped results or interfere with active training.
        if training.get('state') != 'awaiting_validation':
            raise ValueError('Wait for the active training controller to finish')
        cfg = read(root/'configs/round2_v1.json')
        amendment = read(root/'configs/round2_budget_amendment_v1.json')
        receipts = verify_receipts(root)
        if amendment['m4_total_seconds'] != 108000 or cfg['stop']['m4_total_seconds'] != 108000:
            raise ValueError('Aggregate M4 cap changed')
        ledger = BudgetLedger.create(ledger_file, 108000, training, sha(training_file))
        state = {'state': 'running', 'stage': 'training-handoff-audit', 'completed': [],
                 'started_utc': utc(), 'source_receipts': receipts, 'automatic_retry': False,
                 'scope': 'seed42 train audit, density gate, full val, class crossfit and independent audits'}
        write(state_file, state)
        try:
            ledger.begin('training-handoff-audit', cfg['stop']['single_val_seconds'], utc())
            passed, evidence = False, {}
            try:
                evidence = check_training(root, cfg, amendment, training)
                # Existing partial or complete outputs are never adopted blindly.
                labels = ['m4-crossfit-seed42'] + [prefix+'head-'+w+'-seed42'
                           for w in ('S','L') for prefix in ('gate-density-', 'val-')]
                for label in labels:
                    if (result/label).exists() or (root/'work/round2-v1'/label).exists():
                        raise FileExistsError('Validation outputs already exist')
                passed = True
            finally:
                accepted = ledger.finish('training-handoff-audit', time.monotonic()-started,
                                         passed, evidence, utc())
            if not accepted:
                raise TimeoutError('Handoff audit exceeded available budget')
            state['completed'].append('training-handoff-audit')
            python = str(root/'.venv/bin/python')
            def cmd(script, *extra):
                return [python, str(root/'scripts'/script), '--root', str(root), *extra]
            cap = cfg['stop']['single_val_seconds']
            for width in ('S', 'L'):
                run = 'head-'+width+'-seed42'
                for phase, prefix, n in [('gate', 'gate-density-', 32), ('full', 'val-', 1286)]:
                    label = prefix+run
                    run_stage(root, ledger, state, state_file, label, cap,
                              lambda limit: cmd('round2_m4_validate.py', '--run', run, '--phase', phase,
                                                '--time-limit-seconds', str(limit)),
                              lambda: completed_evidence(root, label, 'density', n))
                    run_stage(root, ledger, state, state_file, 'audit-'+label, cap,
                              lambda limit: cmd('audit_round2_m4.py', '--label', label),
                              lambda: completed_evidence(root, label, 'audit', n))
            label = 'm4-crossfit-seed42'
            run_stage(root, ledger, state, state_file, label, cap,
                      lambda limit: cmd('round2_m4_crossfit.py'),
                      lambda: completed_evidence(root, label, 'crossfit'))
            run_stage(root, ledger, state, state_file, 'audit-'+label, cap,
                      lambda limit: cmd('audit_round2_crossfit.py'),
                      lambda: completed_evidence(root, label, 'audit'))
            state.update(state='awaiting_seed_sensitivity', stage='R2-seeds43-44-required',
                         spent_seconds=ledger.spent, remaining_seconds=ledger.remaining,
                         updated_utc=utc(), all_round2_complete=False)
            write(state_file, state)
        except BaseException as exc:
            state.update(state='stopped', error_type=type(exc).__name__,
                         spent_seconds=ledger.spent, remaining_seconds=ledger.remaining, updated_utc=utc())
            write(state_file, state)
            raise


def interrupted(signum, frame):
    raise InterruptedError('Validation controller interrupted')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    workflow(args.root.resolve())


if __name__ == '__main__':
    main()
