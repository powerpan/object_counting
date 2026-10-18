"""Bounded R3-2 workflow on the existing instance; no automatic power action."""
import time
PROCESS_STARTED = time.monotonic()
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_tta_common import sources,read,sha,write_json,utc
from planb.round3_followup import inherit_ledger


def main(root):
    locks=[]
    for name in ('.round2-m4.lock','.m2-m3.lock','.round3.lock'):
        f=(root/'results'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    cfg,splits,receipt=sources(root);previous=root/'results/round3-v1'
    if read(previous/'m3_workflow_v1.json')['state']!='awaiting_publication_and_shutdown':
        raise ValueError('M3 stage has not completed')
    manifest=read(previous/'m3_complete_files_v1.json')
    for name,digest in manifest['files_sha256'].items():
        if sha(previous/name)!=digest:raise ValueError('M3 audited evidence changed')
    if read(previous/'m3-crossfit/audit.json')['status']!='passed':raise ValueError('Missing M3 audit')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Another GPU process is active')
    out=root/'results/round3-tta-v1'
    if out.exists():raise FileExistsError('TTA output exists; no implicit restart')
    out.mkdir()
    ledger=inherit_ledger(previous/'budget_ledger_v1.json',out/'budget_ledger_v1.json',
                          manifest['files_sha256']['budget_ledger_v1.json'],cfg['stage_seconds'],cfg['total_seconds'])
    current='tta-controller-preflight';child=None
    state={'state':'running','started_utc':utc(),'controller_pid':os.getpid(),
           'controller_start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
           'registration_commit':receipt['registration_commit'],'code_commit':receipt['code_commit'],
           'completed':[],'automatic_retry':False,'instance_policy':'keep_running_per_latest_instruction'}
    def status(value,**extra):
        state.update(state=value,stage=current,updated_utc=utc(),**extra);write_json(out/'workflow_v1.json',state)
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
                if child.wait(timeout=max(1,limit-10)):
                    raise RuntimeError('Child stage failed; see retained log')
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
        if not accepted:raise TimeoutError('TTA stage failed or exceeded budget')
        state['completed'].append(identifier);status('running')
    try:
        ledger.begin(current,'R3-0',120,utc())
        if not ledger.finish(current,time.monotonic()-PROCESS_STARTED,True,utc()):
            raise TimeoutError('TTA preflight budget exhausted')
        execute('tta-unit-tests','R3-0',[python,'-m','unittest','discover','-s','tests','-p','test_round3*.py','-v'],120,True)
        log=root/'logs/round3-tta-unit-tests.log';text=log.read_text();match=re.search(r'Ran (\d+) tests',text)
        if not match or not text.rstrip().endswith('OK'):raise ValueError('Missing CPU test proof')
        write_json(out/'cpu_tests_v1.json',{'status':'passed','n':int(match[1]),'cuda_visible_devices':'','log_sha256':sha(log)})
        runner=[python,'scripts/round3_tta_eval.py','--root',str(root)]
        auditor=[python,'scripts/audit_round3_tta.py','--root',str(root)]
        execute('tta-gate','R3-0',runner+['--phase','gate','--variant','tta-repeat'],1200)
        execute('tta-audit-gate','R3-0',auditor+['--label','gate'],300,True)
        if read(out/'gate/audit.json')['status']!='passed':raise ValueError('TTA gate failed')
        for phase in ('smoke','full'):
            for variant in ('T2','T4','tta-repeat'):
                label=phase+'-'+variant
                execute('tta-'+label,'R3-2',runner+['--phase',phase,'--variant',variant],5400 if phase=='full' else 900)
                summary=read(out/label/'summary.json')
                if summary['status']!='complete' or summary['n']!=(1286 if phase=='full' else 32):
                    raise ValueError('Incomplete TTA inference')
                execute('tta-audit-'+label,'R3-2',auditor+['--label',label],900,True)
                if read(out/label/'audit.json')['status']!='passed':raise ValueError('TTA audit failed')
        execute('tta-crossfit','R3-2',auditor+['--crossfit'],900,True)
        execute('tta-crossfit-audit','R3-2',auditor+['--audit-crossfit'],900,True)
        status('awaiting_publication',child_pid=None,scope='TTA only; CPU calibration, combination and test not started')
    except BaseException as exc:
        status('failed',child_pid=None,error_type=type(exc).__name__)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
