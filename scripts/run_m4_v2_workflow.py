"""Sequential M4-v2 seed42 bootstrap; shared lock and cumulative prior budget."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.budget import BudgetLedger
from planb.round2.common import sha,write_json,utc
from m4_v2_common import settings,receipt,read
from run_round2_validation_workflow import run_stage,interrupted


def import_budget(root,path,c):
    predecessor=root/c['predecessor_ledger']
    if sha(predecessor)!=c['predecessor_ledger_sha256']:raise ValueError('Prior budget changed')
    old=read(predecessor)
    if old['state']!='stopped' or old['active'] is not None:raise ValueError('Unsettled prior stage')
    spent=math.fsum(e['elapsed_seconds'] for e in old['entries'])
    if not math.isclose(spent,c['predecessor_spent_seconds'],abs_tol=1e-8,rel_tol=0):raise ValueError('Prior budget total mismatch')
    data={'version':'round2-m4-ledger-v1','cap_seconds':108000,'state':'open','active':None,'automatic_retry':False,
          'scope':'Prior S/L/preflight and validation imported once; all v2 work accounted sequentially',
          'entries':[{'id':'round2-through-L-cancellation','outcome':'imported','elapsed_seconds':spent,
                      'evidence':{c['predecessor_ledger']:sha(predecessor)}}]}
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')
    return BudgetLedger(path)


def workflow(root):
    base,old,c,splits=settings(root);sources=receipt(root,'m4-v2-workflow')
    folder=root/'results/m4-v2-v1';folder.mkdir(exist_ok=True);state_path=folder/'workflow_v1.json';ledger_path=folder/'m4_budget_ledger_v2.json'
    if state_path.exists() or ledger_path.exists():raise FileExistsError('Workflow already exists; no automatic restart')
    with (root/'results/.round2-m4.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('Another GPU process is active')
        ledger=import_budget(root,ledger_path,c)
        state={'state':'starting','stage':None,'started_utc':utc(),'completed':[],'automatic_retry':False,
               'registration_commit':sources['registration_commit'],'code_commit':sources['code_commit']}
        write_json(state_path,state);python=str(root/'.venv/bin/python')
        def command(script,limit,*args):return [python,str(root/'scripts'/script),'--root',str(root),'--time-limit-seconds',str(limit),*args]
        def evidence(stage,filename='summary.json'):
            path=folder/stage/filename;r=read(path)
            if r['status'] not in ('passed','complete','negative_safety_stop'):raise ValueError('Stage did not finish normally')
            return {str(path.relative_to(root)):sha(path)}
        def run(stage,script,cap,*args):
            run_stage(root,ledger,state,state_path,'m4v2-'+stage,cap,
                      lambda limit:command(script,limit,*args),lambda:evidence(stage))
        def audit(stage):
            run_stage(root,ledger,state,state_path,'m4v2-audit-'+stage,5400,
                lambda limit:command('audit_m4_v2_pretraining.py',limit,'--stage',stage),lambda:evidence(stage,'audit.json'))
        try:
            run('initial-gate','m4_v2_preflight.py',1800,'--phase','initial-gate');audit('initial-gate')
            run('s-dense-check','m4_s_dense_check.py',1800);audit('s-dense-check')
            run('s-fusion-record','m4_s_fusion_record.py',1800);audit('s-fusion-record')
            run('capacity','m4_v2_preflight.py',1800,'--phase','capacity')
            run('baseline-holdout','m4_v2_train.py',5400,'--phase','baseline-holdout');audit('baseline-holdout')
            run('train-seed42','m4_v2_train.py',25200,'--phase','train','--seed','42');audit('train-seed42')
            status=read(folder/'train-seed42/status.json')['status']
            state.update(state='negative_safety_stop' if status=='negative_safety_stop' else 'awaiting_validation',
                         stage='seed42-training-reviewed',updated_utc=utc(),spent_seconds=ledger.spent,
                         remaining_seconds=ledger.remaining,all_work_complete=False)
            write_json(state_path,state)
        except BaseException as e:
            state.update(state='stopped',error_type=type(e).__name__,updated_utc=utc(),spent_seconds=ledger.spent,remaining_seconds=ledger.remaining)
            write_json(state_path,state);raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    workflow(a.root.resolve())


if __name__=='__main__':main()
