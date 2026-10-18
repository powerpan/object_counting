"""Adopt the unchanged S process, validate S, then train/validate L and crossfit."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.round2.budget import BudgetLedger, seconds
from run_round2_validation_workflow import (
    read, sha, utc, write, verify_receipts, completed_evidence, run_stage, interrupted,
)


def process(pid):
    base = Path('/proc')/str(pid)
    try:
        fields = (base/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'state': fields[0], 'start_ticks': int(fields[19]),
                'cwd': str((base/'cwd').resolve(strict=True)),
                'args': (base/'cmdline').read_bytes().split(b'\0')}
    except FileNotFoundError:
        return None


def script_matches(current, expected):
    """Resolve the actual Python script argument using its process cwd."""
    if len(current['args']) < 2:
        return False
    script = Path(os.fsdecode(current['args'][1]))
    if not script.is_absolute():
        if not current.get('cwd'):
            return False
        script = Path(current['cwd'])/script
    return script.resolve() == Path(expected).resolve()


def stop_identity(pid, expected_ticks):
    """Signal only this verified PID, never its group or unrelated training."""
    current = process(pid)
    if current is None:
        return
    if current['start_ticks'] != expected_ticks:
        raise RuntimeError('PID was reused; no signal sent')
    os.kill(pid, signal.SIGTERM)
    if current['state'] == 'T':
        # A stopped controller must run briefly to receive its pending SIGTERM.
        os.kill(pid, signal.SIGCONT)
    deadline = time.monotonic()+10
    while process(pid) is not None:
        if time.monotonic() >= deadline:
            raise TimeoutError('Verified process did not exit; no dependent work may start')
        time.sleep(.05)


def verify_sources(root):
    receipts = verify_receipts(root)
    path = root/'provenance/round2_s_first_receipt.json'
    receipt = read(path)
    if receipt.get('remote_main_verified') is not True or not receipt['amendment_commit'].startswith('a20c49a'):
        raise ValueError('S-first amendment was not committed and verified before use')
    for name, expected in receipt['files_sha256'].items():
        if sha(root/name) != expected:
            raise ValueError('S-first source mismatch')
    receipts['round2_s_first_receipt'] = {'code_commit': receipt['code_commit'], 'receipt_sha256': sha(path)}
    cfg = read(root/'configs/round2_s_first_amendment_v1.json')
    expected = (21600, 43200, 108000, 5400)
    if tuple(cfg[k] for k in ('single_S_seconds', 'single_L_seconds', 'total_M4_seconds', 'single_validation_seconds')) != expected:
        raise ValueError('Approved budget limits changed')
    if cfg['test_enabled'] or not cfg['S_result_does_not_change_L_settings']:
        raise ValueError('Unexpected experimental scope')
    return cfg, receipts


def check_one_training(root, width):
    if width not in ('S', 'L'):
        raise ValueError('Unknown head')
    label = 'head-'+width+'-seed42'
    folder = root/'results/round2-v1'/label
    if any((folder/name).exists() for name in ('error.json', 'failure.json')):
        raise ValueError('Training has an unresolved failure')
    status, manifest, history = [read(folder/name) for name in
                               ('status.json', 'checkpoint_manifest.json', 'history.json')]
    if status['status'] != 'complete' or status['epoch'] != 30 or manifest['complete_epochs'] != 30:
        raise ValueError('Head has not completed 30 epochs')
    if [r['epoch'] for r in history] != list(range(1,31)):
        raise ValueError('Missing or repeated epoch')
    previous = 0.
    for row in history:
        if row['train_n'] != 3457 or row['holdout_n'] != 202:
            raise ValueError('Training split changed')
        for key in ('train_loss', 'holdout_loss', 'holdout_mae', 'elapsed_seconds'):
            seconds(row[key])
        if row['elapsed_seconds'] < previous or row['frozen_state_sha256'] != manifest['frozen_model_sha256']:
            raise ValueError('Changed frozen model or invalid elapsed time')
        previous = row['elapsed_seconds']
    best = min(history, key=lambda r: (r['holdout_mae'], r['epoch']))
    for record in (status, manifest):
        if record['best_epoch'] != best['epoch'] or record['best_holdout_mae'] != best['holdout_mae']:
            raise ValueError('Checkpoint did not follow the fixed holdout selection rule')
    elapsed = seconds(status['elapsed_seconds'])
    if not previous <= elapsed <= {'S':21600, 'L':43200}[width]:
        raise ValueError('Training time exceeds approved cap')
    for kind in ('best', 'last'):
        if sha(root/'work/round2-v1'/label/(kind+'.pth')) != manifest[kind+'_sha256']:
            raise ValueError('Checkpoint hash mismatch')
    report = {'status':'passed', 'width':width, 'complete_epochs':30,
              'best_epoch':best['epoch'], 'best_holdout_mae':best['holdout_mae'],
              'training_elapsed_seconds':elapsed, 'holdout_only_not_official_val':True,
              'frozen_state_unchanged':True, 'checkpoint_hashes_checked':True,
              'files_sha256':{name:sha(folder/name) for name in
                              ('status.json','checkpoint_manifest.json','history.json')},
              'audit_code_sha256':sha(Path(__file__))}
    return report


def audit_training(root, width):
    verify_sources(root)
    report = check_one_training(root, width)
    out = root/'results/round2-v1'/('head-'+width+'-seed42')/'training_audit_s_first_v1.json'
    if out.exists():
        raise FileExistsError('Existing training audit must not be overwritten')
    write(out, report)
    return report


def create_ledger(path, predecessor, source_sha, cfg):
    if predecessor.get('state') != 'running' or predecessor.get('stage') != 'head-S-seed42':
        raise ValueError('Expected the active S controller before L starts')
    if set(predecessor['completed']) != {'cpu-tests','gate','capacity-S','capacity-L'}:
        raise ValueError('Do not import completed training twice')
    spent = seconds(predecessor['spent_seconds'])
    if spent >= cfg['total_M4_seconds']:
        raise TimeoutError('Aggregate M4 budget exhausted')
    # Only preflight is imported. The full adopted S run is settled once below.
    data = {'version':'round2-m4-ledger-v1', 'cap_seconds':cfg['total_M4_seconds'],
            'state':'open', 'active':None, 'automatic_retry':False,
            'scope':'S-first amendment; preflight and each subsequent stage counted once',
            'entries':[{'id':'preflight-before-S', 'outcome':'passed', 'elapsed_seconds':spent,
                        'evidence':{'workflow_budget_before_s_first_v1.json':source_sha}}]}
    with Path(path).open('x') as stream:
        stream.write(json.dumps(data,indent=2)+'\n')
    return BudgetLedger(path)


def require_s_validation(root, ledger):
    """Do not launch L on a process exit alone, or on an unaccounted S result."""
    needed = {'gate-density-head-S-seed42','audit-gate-density-head-S-seed42',
              'val-head-S-seed42','audit-val-head-S-seed42'}
    passed = {e['id'] for e in ledger.data['entries'] if e['outcome']=='passed'}
    if ledger.data['state'] != 'open' or ledger.data['active'] is not None or not needed <= passed:
        raise ValueError('S validation and its audits must all pass before L')
    completed_evidence(root, 'gate-density-head-S-seed42', 'audit', 32)
    completed_evidence(root, 'val-head-S-seed42', 'audit', 1286)


def validate_head(root, ledger, state, state_file, cfg, width):
    python = str(root/'.venv/bin/python')
    def cmd(script, *extra):
        return [python, str(root/'scripts'/script), '--root', str(root), *extra]
    cap = cfg['single_validation_seconds']
    audit_file = root/'results/round2-v1'/('head-'+width+'-seed42')/'training_audit_s_first_v1.json'
    def verified_training():
        if read(audit_file)['status'] != 'passed':
            raise ValueError('Training audit did not pass')
        return {str(audit_file.relative_to(root)):sha(audit_file)}
    run_stage(root, ledger, state, state_file, 'training-audit-'+width+'42', cap,
              lambda limit:cmd('continue_round2_s_first.py','--audit-head',width), verified_training)
    for phase, prefix, n in [('gate','gate-density-',32), ('full','val-',1286)]:
        label = prefix+'head-'+width+'-seed42'
        run_stage(root, ledger, state, state_file, label, cap,
                  lambda limit:cmd('round2_m4_validate.py','--run','head-'+width+'-seed42',
                                    '--phase',phase,'--time-limit-seconds',str(limit)),
                  lambda:completed_evidence(root,label,'density',n))
        run_stage(root, ledger, state, state_file, 'audit-'+label, cap,
                  lambda limit:cmd('audit_round2_m4.py','--label',label),
                  lambda:completed_evidence(root,label,'audit',n))


def continue_after_s(root, ledger, state, state_file, cfg):
    validate_head(root, ledger, state, state_file, cfg, 'S')
    require_s_validation(root, ledger)
    folder = root/'results/round2-v1'
    ready = folder/'s_validation_ready_v1.json'
    if ready.exists():
        raise FileExistsError('S result handoff already exists')
    write(ready, {'status':'passed', 'available_utc':utc(),
                  'evidence':completed_evidence(root,'val-head-S-seed42','audit',1286),
                  'spent_seconds':ledger.spent, 'remaining_seconds':ledger.remaining,
                  'scope':'S alone on full development val; not crossfit width selection'})
    state['s_validation_available'] = True
    write(state_file, state)
    python = str(root/'.venv/bin/python')
    def cmd(script, *extra):
        return [python,str(root/'scripts'/script),'--root',str(root),*extra]
    def l_command(limit):
        context = folder/'L_delegated_budget_s_first_v1.json'
        if context.exists():
            raise FileExistsError('L budget already delegated')
        write(context, {'global_ledger_remaining_seconds':ledger.remaining,
                        'delegated_training_seconds':limit,
                        'child_remaining_total_seconds_means':'allowance delegated to this invocation, not global ledger balance',
                        'original_L_wrapper_unchanged':True, 'started_utc':utc()})
        return cmd('round2_m4_budget.py','--seed','42','--remaining-m4-seconds',str(limit))
    def l_complete():
        status = read(folder/'head-L-seed42/status.json')
        manifest = read(folder/'head-L-seed42/checkpoint_manifest.json')
        if status['status'] != 'complete' or manifest['complete_epochs'] != 30:
            raise ValueError('L did not finish 30 epochs')
        return {name:sha(folder/name) for name in ('head-L-seed42/status.json',
                'head-L-seed42/checkpoint_manifest.json','L_delegated_budget_s_first_v1.json')}
    run_stage(root, ledger, state, state_file, 'head-L-seed42', cfg['single_L_seconds'],
              l_command, l_complete)
    validate_head(root, ledger, state, state_file, cfg, 'L')
    label = 'm4-crossfit-seed42'
    run_stage(root, ledger, state, state_file, label, cfg['single_validation_seconds'],
              lambda limit:cmd('round2_m4_crossfit.py'), lambda:completed_evidence(root,label,'crossfit'))
    run_stage(root, ledger, state, state_file, 'audit-'+label, cfg['single_validation_seconds'],
              lambda limit:cmd('audit_round2_crossfit.py'), lambda:completed_evidence(root,label,'audit'))
    state.update(state='awaiting_seed_sensitivity', stage='R2-seeds43-44-required',
                 spent_seconds=ledger.spent, remaining_seconds=ledger.remaining,
                 all_round2_complete=False, updated_utc=utc())
    write(state_file, state)


def adopt(root, controller_pid):
    cfg, receipts = verify_sources(root)
    folder = root/'results/round2-v1';old_file=folder/'workflow_budget_v1.json'
    state_file=folder/'workflow_s_first_v1.json';ledger_file=folder/'m4_budget_ledger_v1.json'
    hold = read(folder/'controller_hold_for_s_first_v1.json')
    previous = read(old_file)
    if state_file.exists() or ledger_file.exists() or (folder/'head-L-seed42').exists():
        raise FileExistsError('Inspect existing work; do not restart or overwrite')
    if previous['state'] != 'running' or previous['stage'] != 'head-S-seed42':
        raise ValueError('S-first adoption is only valid before L starts')
    if controller_pid != hold['controller_pid'] or previous['stage_pid'] != hold['S_pid']:
        raise ValueError('Controller or S identity differs from the hold receipt')
    controller = process(controller_pid);child=process(hold['S_pid'])
    if not controller or controller['state'] != 'T' or controller['start_ticks'] != hold['controller_start_ticks']:
        raise ValueError('Expected the verified held controller')
    if not script_matches(controller, root/'scripts/continue_round2_budget.py'):
        raise ValueError('Unexpected controller command')
    status = read(folder/'head-S-seed42/status.json')
    if child is None:
        if status['status'] != 'complete':
            raise ValueError('S exited without completion before handoff')
    elif child['start_ticks'] != hold['S_start_ticks'] or not script_matches(child, root/'scripts/round2_m4.py'):
        raise ValueError('S process identity changed')
    if sha(old_file) != hold['workflow_sha256']:
        raise ValueError('Held workflow changed unexpectedly')
    for prefix in ('gate-density-', 'val-'):
        if list(folder.glob(prefix+'head-*')):
            raise FileExistsError('Validation already has outputs')
    saved = folder/'workflow_budget_before_s_first_v1.json'
    if saved.exists():
        raise FileExistsError('Previous controller snapshot already exists')
    write(saved, previous)
    state={'state':'adopting', 'stage':'head-S-seed42','started_utc':utc(),
           'completed':[],'automatic_retry':False,'stage_pid':hold['S_pid'],
           'source_receipts':receipts,'scope':'S full validation before unchanged L training'}
    write(state_file,state)
    stop_identity(controller_pid, hold['controller_start_ticks'])
    with (root/'results/.round2-m4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(old_file,{**previous,'state':'superseded','successor':'workflow_s_first_v1.json','updated_utc':utc()})
        ledger=create_ledger(ledger_file,previous,sha(saved),cfg)
        origin=read(folder/'workflow_before_budget_amendment.json')['stage_started_utc']
        start=datetime.fromisoformat(origin.replace('Z','+00:00')).timestamp()
        ledger.begin('head-S-seed42',cfg['single_S_seconds'],origin)
        state.update(state='running',spent_seconds=ledger.spent,updated_utc=utc());write(state_file,state)
        try:
            while True:
                current=process(hold['S_pid']);status=read(folder/'head-S-seed42/status.json')
                if status['status']=='complete' and current is None:
                    if read(folder/'head-S-seed42/checkpoint_manifest.json')['complete_epochs'] != 30:
                        raise ValueError('S did not complete all epochs')
                    break
                if current is None or current['start_ticks'] != hold['S_start_ticks']:
                    raise RuntimeError('S stopped without completing training')
                if any((folder/'head-S-seed42'/n).exists() for n in ('error.json','failure.json')):
                    raise RuntimeError('S has an unresolved failure')
                if time.time()-start >= cfg['single_S_seconds']:
                    raise TimeoutError('S reached its unchanged 6-hour cap')
                time.sleep(15)
            if not ledger.finish('head-S-seed42',status['elapsed_seconds'],True,
                                 {'head-S-seed42/status.json':sha(folder/'head-S-seed42/status.json')},utc()):
                raise TimeoutError('S exceeded registered runtime')
            state['completed'].append('head-S-seed42');state.pop('stage_pid',None)
            continue_after_s(root,ledger,state,state_file,cfg)
        except BaseException as exc:
            if ledger.data['active'] and ledger.data['active']['id']=='head-S-seed42':
                stop_identity(hold['S_pid'],hold['S_start_ticks'])
                ledger.finish('head-S-seed42',max(0.,time.time()-start),False,{},utc())
            state.update(state='stopped',error_type=type(exc).__name__,spent_seconds=ledger.spent,
                         remaining_seconds=ledger.remaining,updated_utc=utc());write(state_file,state)
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--previous-controller-pid',type=int)
    group.add_argument('--audit-head',choices=['S','L'])
    args=parser.parse_args();root=args.root.resolve()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    if args.audit_head:
        print(json.dumps(audit_training(root,args.audit_head)),flush=True)
    else:
        adopt(root,args.previous_controller_pid)


if __name__=='__main__':
    main()
