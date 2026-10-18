"""One locked, bounded R3-0/R3-1 run, ending before TTA or any test."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_common import sources,read,write_json,utc
from planb.round3_accounting import StageLedger


def main(root):
    started=time.monotonic();locks=[]
    for name in ('.round2-m4.lock','.m2-m3.lock','.round3.lock'):
        f=(root/'results'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    cfg,splits,receipt=sources(root)
    if read(root/'results/m4-v2-v1/workflow_v1.json')['state']!='negative_safety_stop':
        raise ValueError('Previous M4 workflow has not closed')
    if read(root/'results/m4-v2-v1/m4_budget_ledger_v2.json')['active'] is not None:
        raise ValueError('Unsettled previous budget')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('Another GPU process is active')
    out=root/'results/round3-v1'
    if out.exists():raise FileExistsError('Round 3 output exists; no implicit restart')
    out.mkdir();ledger=StageLedger.create(out/'budget_ledger_v1.json',cfg['stage_seconds'],cfg['total_seconds'])
    status_path=out/'m3_workflow_v1.json';child=None;current='preflight'
    state={'state':'running','started_utc':utc(),'controller_pid':os.getpid(),
           'controller_start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
           'registration_commit':receipt['registration_commit'],'code_commit':receipt['code_commit'],
           'completed':[],'automatic_retry':False}
    def status(value,**extra):
        state.update(state=value,stage=current,updated_utc=utc(),**extra);write_json(status_path,state)
    def interrupt(signum,frame):raise KeyboardInterrupt('User or controller shutdown')
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    python=str(root/'.venv/bin/python')
    def execute(identifier,stage,command,cap,cpu=False):
        nonlocal current,child
        current=identifier;tick=time.monotonic();limit=ledger.begin(identifier,stage,cap,utc());ok=False
        status('running',child_pid=None,stage_limit_seconds=limit)
        env=dict(os.environ,PYTHONUNBUFFERED='1',TORCH_HOME=str(root/'cache/torch'),OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        if cpu:env['CUDA_VISIBLE_DEVICES']=''
        try:
            with (root/'logs'/('round3-'+identifier+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                status('running',child_pid=child.pid)
                code=child.wait(timeout=max(1,limit-10))
                if code:raise RuntimeError('Child stage failed; see retained stage log')
                ok=True
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
            child=None
            accepted=ledger.finish(identifier,time.monotonic()-tick,ok,utc())
            status('running' if accepted else 'failed',child_pid=None)
        if not accepted:raise TimeoutError('Stage budget or execution failure')
        state['completed'].append(identifier);status('running')
    try:
        ledger.begin('controller-preflight','R3-0',60,utc());ledger.finish('controller-preflight',time.monotonic()-started,True,utc())
        execute('unit-tests','R3-0',[python,'-m','unittest','discover','-s','tests','-p','test_round3*.py','-v'],120,True)
        runner=[python,'scripts/round3_m3_eval.py','--root',str(root)]
        execute('gate','R3-0',runner+['--phase','gate'],1200)
        gate=read(out/'gate/gate.json')
        if gate['status']!='passed' or gate['n']!=32:raise ValueError('Gate failed')
        auditor=[python,'scripts/audit_round3_m3.py','--root',str(root)]
        for phase in ('smoke','full'):
            for variant in cfg['m3']['runs']:
                label=phase+'-'+variant
                execute(label,'R3-1',runner+['--phase',phase,'--variant',variant],5400 if phase=='full' else 900)
                summary=read(out/label/'summary.json')
                if summary['status']!='complete' or summary['n']!=(1286 if phase=='full' else 32):
                    raise ValueError('Incomplete inference')
                execute('audit-'+label,'R3-1',auditor+['--label',label],900,True)
                if read(out/label/'audit.json')['status']!='passed':raise ValueError('Audit failed')
        execute('crossfit','R3-1',auditor+['--crossfit'],900,True)
        execute('crossfit-audit','R3-1',[python,'scripts/audit_round3_crossfit.py','--root',str(root)],900,True)
        status('awaiting_publication_and_shutdown',child_pid=None,
               scope='M3 only; TTA, calibration, test and new training are not started')
    except BaseException as exc:
        status('failed',child_pid=None,error_type=type(exc).__name__,shutdown_required=True)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
