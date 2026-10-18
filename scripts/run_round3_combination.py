"""Combine audited existing outputs after the immutable fit-fold routing plan."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_round3_combination_plan import inputs
from round3_calibration_io import read,sha,write,rows_write,utc
from audit_round3_m3 import report
from planb.round3_combination import ready,assemble

ASSEMBLY_FILES=('planb/round3_combination.py','scripts/run_round3_combination.py',
                'scripts/audit_round3_combination.py','tests/test_round3_combination.py',
                'docs/ROUND3_COMBINATION_ASSEMBLY.md')


def load_inputs(root,asset_root,receipt_path):
    cfg,split,receipt,names,labels,base,variants,cal,evidence=inputs(root,asset_root,receipt_path)
    if any(n not in receipt['files_sha256'] for n in ASSEMBLY_FILES):raise ValueError('Assembly source receipt missing')
    out=root/'results/round3-combination-v1';a=read(out/'plan_audit_v1.json');p=read(out/'plan_protocol_v1.json')
    if a['status']!='passed' or read(out/'plan_execution_v1.json')['state']!='awaiting_combination_implementation':
        raise ValueError('Plan has not been audited')
    for n,h in a['files_sha256'].items():
        if sha(out/n)!=h:raise ValueError('Plan evidence changed')
    if len(p['files_sha256'])!=104 or any(receipt['files_sha256'].get(n)!=h for n,h in p['files_sha256'].items()):
        raise ValueError('Plan source changed')
    for k,v in evidence.items():
        if k!='source_receipt_sha256' and p.get(k)!=v:raise ValueError('Plan input changed')
    plan=read(out/'plan_v1.json');ready(plan)
    routes={**plan['routes']['A'],**plan['routes']['B']}
    if set(routes)!=set(names):raise ValueError('Incomplete fold routing')
    needed=set(routes.values());source={'B0':base} if 'B0' in needed else {};source_sha={}
    paths={'B0':root/cfg['baseline_result']/'predictions.jsonl'}
    paths.update({v:root/'results/round3-v1'/('full-'+v)/'predictions.jsonl' for v in needed if v.startswith(('m3-','b0-'))})
    paths.update({v:root/'results/round3-tta-v1'/('full-'+v)/'predictions.jsonl' for v in needed if v in ('T2','T4')})
    for v in needed:
        if v not in paths:raise ValueError('Required inference not implemented')
        source_sha[v]=sha(paths[v]);source.setdefault(v,{})
        if v=='B0':continue
        with paths[v].open() as f:
            for line in f:
                row=json.loads(line)
                if routes.get(row['image'])==v:source[v][row['image']]=row
    annotations=read(asset_root/'annotations/annotation_FSC147_384.json')
    proof={**evidence,'plan_files_sha256':{n:sha(out/n) for n in ('plan_v1.json','plan_protocol_v1.json','plan_audit_v1.json','plan_execution_v1.json')},
           'selected_source_predictions_sha256':source_sha}
    return cfg,receipt,names,labels,base,source,annotations,plan,routes,proof


def main(root,asset_root,receipt_path):
    started=time.monotonic()
    cfg,receipt,names,labels,base,source,annotations,plan,routes,proof=load_inputs(root,asset_root,receipt_path)
    out=root/'results/round3-combination-v1/assembly-v1'
    if out.exists():raise FileExistsError('Combination assembly exists; no restart')
    out.mkdir();work=root/'work/round3-combination-v1';work.mkdir(parents=True,exist_ok=False)
    def stop(signum,frame):raise TimeoutError('Local combination CPU phase exceeded 600 seconds')
    old=signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(600-(time.monotonic()-started))))
    try:
        write(out/'protocol.json',{**receipt,**proof,'started_utc':utc(),'device':'CPU','gpu_seconds':0,
              'scope':'audited existing-output combination only; no calibration refit or test',
              'cpu_timeout_seconds':600,'instance_policy':'keep_running'})
        rows=[assemble(source[routes[n]][n],routes[n],annotations[n],proof['selected_source_predictions_sha256'][routes[n]]) for n in names]
        rows_write(out/'predictions.jsonl',rows)
        r=report(rows,base,labels,cfg['bootstrap_replicates'])
        summary={'status':'complete','n':len(rows),'scope':'development_only_no_test','result':r,
                 'routing_counts':{v:sum(x==v for x in routes.values()) for v in sorted(set(routes.values()))},
                 'all_val_future_decision':plan['decisions']['future-only'],
                 'final_freeze':'pending_optional_M2_and_independent_human_labels',
                 'timing':'source-component timing only; no new end-to-end combination inference',
                 'density':'reconstructed from selected audited source; source density arrays remain preserved',
                 'independent_audit':'pending','finished_utc':utc()}
        write(out/'summary.json',summary)
        with (work/'audit.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/audit_round3_combination.py','--root',str(root),
                            '--asset-root',str(asset_root),'--receipt',str(receipt_path)],cwd=root,
                            stdout=log,stderr=subprocess.STDOUT,check=True,
                            timeout=max(1,590-(time.monotonic()-started)))
        write(out/'execution.json',{'state':'awaiting_publication','completed_utc':utc(),
              'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,'automatic_retry':False,
              'instance_policy':'keep_running'})
    except BaseException as exc:
        write(out/'error.json',{'state':'failed','error_type':type(exc).__name__,'utc':utc(),
              'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,'automatic_retry':False})
        raise
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.asset_root.resolve(),a.receipt.resolve())
