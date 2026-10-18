"""Bounded optional M2-v2 stage; preserve the existing instance and evidence."""
import time
PROCESS_STARTED=time.monotonic()
import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_m2_common import sources,read,sha,write_json,utc
from planb.round3_followup import inherit_ledger


def main(root):
    locks=[]
    for name in ('.round2-m4.lock','.m2-m3.lock','.round3.lock'):
        f=(root/'results'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    cfg,splits,m2,receipt=sources(root);previous=root/'results/round3-tta-v1'
    if read(previous/'workflow_v1.json')['state']!='awaiting_publication':raise ValueError('TTA not complete')
    manifest=read(previous/'complete_files_v1.json')
    for n,h in manifest['files_sha256'].items():
        if sha(previous/n)!=h:raise ValueError('TTA completed evidence changed')
    for n,h in receipt['prerequisites_sha256'].items():
        if sha(root/n)!=h:raise ValueError('Local completed-stage evidence differs')
    if read(root/'results/round3-combination-v1/assembly-v1/audit.json')['status']!='passed':raise ValueError('Combination not audited')
    if read(root/'results/round3-combination-v1/assembly-v1/execution.json')['state']!='awaiting_publication':raise ValueError('Combination not complete')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('Other GPU work active')
    out=root/'results/round3-m2-v1'
    if out.exists():raise FileExistsError('M2 outputs exist; no restart')
    out.mkdir();ledger=inherit_ledger(previous/'budget_ledger_v1.json',out/'budget_ledger_v1.json',
        manifest['files_sha256']['budget_ledger_v1.json'],cfg['stage_seconds'],cfg['total_seconds'])
    current='m2-controller-preflight';child=None
    state={'state':'running','started_utc':utc(),'controller_pid':os.getpid(),
           'controller_start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),
           'registration_commit':receipt['registration_commit'],'code_commit':receipt['code_commit'],
           'completed':[],'automatic_retry':False,'instance_policy':'keep_running_per_latest_instruction'}
    def status(value,**extra):
        state.update(state=value,stage=current,updated_utc=utc(),**extra);write_json(out/'workflow_v1.json',state)
    def interrupt(signum,frame):raise KeyboardInterrupt('Explicit interruption')
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
                if child.wait(timeout=max(1,limit-10)):raise RuntimeError('M2 child failed; retained log')
                ok=True
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            child=None;accepted=ledger.finish(identifier,time.monotonic()-tick,ok,utc())
            status('running' if accepted else 'failed',child_pid=None)
        if not accepted:raise TimeoutError('M2 stage failed or exceeded budget')
        state['completed'].append(identifier);status('running')
    try:
        ledger.begin(current,'R3-0',120,utc())
        if not ledger.finish(current,time.monotonic()-PROCESS_STARTED,True,utc()):raise TimeoutError('M2 preflight budget exceeded')
        execute('m2-unit-tests','R3-0',[python,'-m','unittest','discover','-s','tests','-p','test_round3*.py','-v'],120,True)
        log=root/'logs/round3-m2-unit-tests.log';text=log.read_text();match=re.search(r'Ran (\d+) tests',text)
        if not match or not text.rstrip().endswith('OK'):raise ValueError('Missing M2 CPU proof')
        write_json(out/'cpu_tests_v1.json',{'status':'passed','n':int(match[1]),'cuda_visible_devices':'','log_sha256':sha(log)})
        runner=[python,'scripts/round3_m2_eval.py','--root',str(root)]
        auditor=[python,'scripts/audit_round3_m2.py','--root',str(root)]
        execute('m2-gate','R3-0',runner+['--phase','gate','--candidate',m2['candidate_order'][0]],1200)
        gate=read(out/'gate/gate.json')
        if gate['status']!='passed' or gate['n']!=32 or not all(r['bitwise_equal'] for r in gate['rows']):raise ValueError('M2 gate failed')
        for phase in ('smoke','full'):
            for repeat in (False,True):
                for candidate in m2['candidate_order']:
                    variant=candidate+('-repeat' if repeat else '');label=phase+'-'+variant
                    cmd=runner+['--phase',phase,'--candidate',candidate]+(['--repeat'] if repeat else [])
                    execute('m2-'+label,'R3-5',cmd,5400 if phase=='full' else 900)
                    summary=read(out/label/'summary.json')
                    if summary['status']!='complete' or summary['n']!=(1286 if phase=='full' else 32):raise ValueError('Incomplete M2 inference')
                    execute('m2-audit-'+label,'R3-5',auditor+['--label',label],900,True)
                    if read(out/label/'audit.json')['status']!='passed':raise ValueError('M2 audit failed')
                    if not repeat and read(out/(label+'-no-fallback')/'audit.json')['status']!='passed':raise ValueError('No-fallback diagnostic audit failed')
        execute('m2-crossfit','R3-5',auditor+['--crossfit'],900,True)
        execute('m2-crossfit-audit','R3-5',auditor+['--audit-crossfit'],900,True)
        status('awaiting_publication',child_pid=None,scope='M2-v2 only; final combination and freeze not completed')
    except BaseException as exc:
        status('failed',child_pid=None,error_type=type(exc).__name__);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
