"""Adopt the active S run without interruption, then apply the approved L limit."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import write_json,utc
from round2_m4_budget import budget_record


def process(pid):
    try:
        base=Path('/proc')/str(pid)
        raw=(base/'stat').read_text().rsplit(')',1)[1].split()
        if raw[0]=='Z':return None
        return {'start_ticks':int(raw[19]),'args':(base/'cmdline').read_bytes().split(b'\0')}
    except FileNotFoundError:return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--previous-controller-pid',type=int,required=True)
    args=p.parse_args();root=args.root.resolve();amendment,receipt=budget_record(root)
    result=root/'results/round2-v1';old_file=result/'workflow.json';out=result/'workflow_budget_v1.json'
    if out.exists():raise FileExistsError('Inspect the existing budget controller before retrying')
    old=json.loads(old_file.read_text())
    assert old['state']=='running' and old['stage']=='head-S-seed42'
    parent=process(args.previous_controller_pid);child=process(old['stage_pid'])
    assert parent and any(x.endswith(b'/scripts/run_round2_m4_workflow.py') or x==b'scripts/run_round2_m4_workflow.py' for x in parent['args'])
    assert child and str(result/'head-S-seed42').encode() in child['args']
    assert any(x.endswith(b'/scripts/round2_m4.py') for x in child['args'])
    write_json(result/'workflow_before_budget_amendment.json',old)
    state={'state':'adopting','stage':'head-S-seed42','started_utc':utc(),'prior_workflow_started_utc':old['started_utc'],
           'completed':old['completed'],'spent_seconds':old['spent_seconds'],'stage_pid':old['stage_pid'],
           'budget_receipt':receipt,'scope':'unchanged S42 then amended-time-limit L42; stop before validation'}
    write_json(out,state)
    # Only the controller exits. Its training child keeps running, with the
    # original optimizer, RNG, process, and immutable source receipt.
    os.kill(args.previous_controller_pid,signal.SIGTERM)
    lock=(root/'results/.round2-m4.lock').open('a')
    deadline=time.monotonic()+15
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise TimeoutError('Previous controller did not release ownership')
            time.sleep(.2)
    write_json(old_file,{**old,'state':'superseded','successor':'workflow_budget_v1.json','updated_utc':utc()})
    state.update(state='running',updated_utc=utc());write_json(out,state)
    active_started=None;active_accounted=False
    start=datetime.fromisoformat(old['stage_started_utc'].replace('Z','+00:00')).timestamp()
    try:
        sdir=result/'head-S-seed42'
        while True:
            current=process(old['stage_pid'])
            status=json.loads((sdir/'status.json').read_text()) if (sdir/'status.json').exists() else {}
            if status.get('status')=='complete' and (sdir/'checkpoint_manifest.json').exists():
                assert json.loads((sdir/'checkpoint_manifest.json').read_text())['complete_epochs']==30
                if current is None:break
            if current is None or current['start_ticks']!=child['start_ticks']:
                raise RuntimeError('Adopted S training stopped without completing 30 epochs')
            if time.time()-start>amendment['m4_S_single_run_seconds']+120:
                os.kill(old['stage_pid'],signal.SIGTERM)
                raise TimeoutError('Adopted S training exceeded its unchanged runtime cap')
            time.sleep(30)
        state['spent_seconds']+=status['elapsed_seconds'];state['completed'].append('head-S-seed42')
        remaining=amendment['m4_total_seconds']-state['spent_seconds']
        if remaining<=0:raise TimeoutError('Aggregate M4 budget exhausted')
        limit=min(remaining,amendment['m4_L_single_run_seconds'])
        command=[str(root/'.venv/bin/python'),str(root/'scripts/round2_m4_budget.py'),'--root',str(root),
                 '--seed','42','--remaining-m4-seconds',str(remaining)]
        state.update(stage='head-L-seed42',stage_started_utc=utc(),effective_limit_seconds=limit)
        write_json(out,state);started=time.monotonic();active_started=started
        with (root/'logs/round2-head-L-seed42.log').open('x') as log:
            proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
            state.update(stage_pid=proc.pid,updated_utc=utc());write_json(out,state)
            try:code=proc.wait(timeout=limit+120)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=120)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise TimeoutError('Amended L budget exhausted')
        state['spent_seconds']+=time.monotonic()-started;active_accounted=True
        if code:raise RuntimeError('L training exited unsuccessfully')
        assert json.loads((result/'head-L-seed42/status.json').read_text())['status']=='complete'
        assert json.loads((result/'head-L-seed42/checkpoint_manifest.json').read_text())['complete_epochs']==30
        state['completed'].append('head-L-seed42');state.pop('stage_pid',None)
        state.update(state='awaiting_validation',stage='R2-2-required',updated_utc=utc());write_json(out,state)
    except BaseException as exc:
        if active_started is not None and not active_accounted:
            state['spent_seconds']+=time.monotonic()-active_started
        elif active_started is None:
            state['spent_seconds']=old['spent_seconds']+max(0.,time.time()-start)
        state.update(state='stopped',error_type=type(exc).__name__,automatic_retry=False,updated_utc=utc())
        write_json(out,state);raise


if __name__=='__main__':main()
