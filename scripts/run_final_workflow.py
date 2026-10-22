"""Bounded final evaluation stages; stage output and failed work are never reused."""
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_common import sources,read,write,sha,utc
from planb.round3_accounting import StageLedger


def main(root,stage):
    start=time.monotonic();locks=[]
    for name in ('.round2-m4.lock','.m2-m3.lock','.round3.lock'):
        f=(root/'results'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    cfg,method,receipt=sources(root)
    backup=read(root/'provenance/final_backup_verified_v1.json')
    if backup['status']!='passed' or not backup['off_instance_all_objects_verified']:raise ValueError('Off-instance backup required')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU busy')
    out=root/'results/final-v1';out.mkdir(exist_ok=True)
    state_path=out/'workflow_'+stage+'.json'
    if state_path.exists():raise FileExistsError('Existing stage; no implicit restart')
    ledger_path=out/'budget_ledger_v1.json'
    if stage=='fsc147-test':
        old=read(root/'results/round3-m2-v1/budget_ledger_v1.json')
        if old['active'] is not None:raise RuntimeError('Old work unsettled')
        ledger=StageLedger.create(ledger_path,cfg['budgets_seconds'],sum(cfg['budgets_seconds'].values()))
        write(out/'historical_budget_reference.json',{'sha256':sha(root/'results/round3-m2-v1/budget_ledger_v1.json'),
              'round3_spent_seconds':sum(e['seconds'] for e in old['entries']),
              'included_again_in_final_stage_ledger':False})
    else:
        if read(out/'workflow_fsc147-test.json')['state']!='awaiting_carpk':raise ValueError('FSC stage not complete')
        ledger=StageLedger(ledger_path)
    state={'state':'running','stage':stage,'started_utc':utc(),'controller_pid':os.getpid(),
           'controller_start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
           'completed':[],'automatic_retry':False,'code_commit':receipt['code_commit']}
    child=None
    def status(value,**extra):
        state.update(state=value,updated_utc=utc(),**extra);write(state_path,state)
    def interrupt(signum,frame):raise KeyboardInterrupt('Interrupted')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    py=str(root/'.venv/bin/python')
    def execute(name,budget,command,cap,cpu=False):
        nonlocal child
        limit=ledger.begin(name,budget,cap,utc());tick=time.monotonic();ok=False
        status('running',step=name,child_pid=None,limit_seconds=limit)
        env=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_HOME=str(root/'cache/torch'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        try:
            with (root/'logs'/('final-'+name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('running',child_pid=child.pid)
                if child.wait(timeout=max(1,limit-10)):raise RuntimeError('Stage failed; retained log')
                ok=True
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            child=None;accepted=ledger.finish(name,time.monotonic()-tick,ok,utc());status('running' if accepted else 'failed',child_pid=None)
        if not accepted:raise TimeoutError('Stage failure or budget exceeded')
        state['completed'].append(name);status('running')
    runner=[py,'scripts/final_eval.py','--root',str(root)]
    audit=[py,'scripts/audit_final.py','--root',str(root)]
    try:
        ledger.begin(stage+'-controller','preflight',120,utc())
        if not ledger.finish(stage+'-controller',time.monotonic()-start,True,utc()):raise TimeoutError('Preflight budget')
        if stage=='fsc147-test':
            execute('cpu-tests','preflight',[py,'-m','unittest','discover','-s','tests','-p','test_final*.py','-v'],180,True)
            log=root/'logs/final-cpu-tests.log';content=log.read_text();match=re.search(r'Ran (\d+) tests',content)
            if not match or not content.rstrip().endswith('OK'):raise ValueError('Missing unit test proof')
            write(out/'cpu_tests.json',{'status':'passed','n':int(match[1]),'cuda_visible_devices':'','log_sha256':sha(log)})
            execute('gate','preflight',runner+['--dataset','gate'],1200)
            execute('gate-audit','preflight',audit+['--dataset','gate'],300,True)
            if read(out/'gate/audit.json')['status']!='passed':raise ValueError('Gate audit failed')
        execute(stage,'test_and_carpk',runner+['--dataset',stage],6900)
        execute(stage+'-audit','test_and_carpk',audit+['--dataset',stage],900,True)
        if read(out/stage/'audit.json')['status']!='passed':raise ValueError('Final audit failed')
        status('awaiting_carpk' if stage=='fsc147-test' else 'awaiting_demo',child_pid=None)
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,child_pid=None);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--stage',choices=['fsc147-test','carpk'],required=True);a=p.parse_args();main(a.root.resolve(),a.stage)
