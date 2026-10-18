"""Single-owner R2-0 and seed-42 head training, stopping before model selection."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import write_json, utc


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve()
    lock=(root/'results/.round2-m4.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=root/'results/round2-v1';result.mkdir(exist_ok=True)
    state_file=result/'workflow.json'
    if state_file.exists():raise FileExistsError('Inspect existing workflow before restarting')
    cfg=json.loads((root/'configs/round2_v1.json').read_text())
    py=str(root/'.venv/bin/python');entry=str(root/'scripts/round2_m4.py')
    state={'state':'running','stage':'preflight','started_utc':utc(),'spent_seconds':0.,'completed':[],
           'scope':'R2-0 and two seed42 heads only; later validation/selection still required'}
    stages=[('cpu-tests',[py,'-m','unittest','discover','-s','tests','-v'],300,None),
            ('gate',[py,entry,'--root',str(root),'--phase','gate','--output',str(result/'gate')],900,('gate','gate.json','passed'))]
    for width in ('S','L'):
        stages.append(('capacity-'+width,[py,entry,'--root',str(root),'--phase','capacity','--width',width,
                      '--output',str(result/('capacity-'+width))],900,('capacity-'+width,'capacity.json','passed')))
    for width in ('S','L'):
        name='head-'+width+'-seed42'
        stages.append((name,[py,entry,'--root',str(root),'--phase','train','--width',width,'--seed','42',
                      '--output',str(result/name)],cfg['stop']['m4_single_run_seconds'],(name,'status.json','complete')))
    for name,command,limit,expected in stages:
        remaining=cfg['stop']['m4_total_seconds']-state['spent_seconds']
        if remaining<=0:
            state.update(state='budget_exceeded',updated_utc=utc());write_json(state_file,state);return
        limit=min(limit,remaining)
        if name.startswith('head-'):command+=['--time-limit-seconds',str(limit)]
        state.update(stage=name,stage_started_utc=utc(),updated_utc=utc());write_json(state_file,state)
        started=time.monotonic()
        try:
            with (root/'logs'/('round2-'+name+'.log')).open('x') as log:
                proc=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
                state['stage_pid']=proc.pid;write_json(state_file,state)
                try:
                    code=proc.wait(timeout=limit+(120 if name.startswith('head-') else 0))
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
                    raise TimeoutError('Registered stage budget reached')
            if code:raise RuntimeError('Stage exited unsuccessfully')
            if expected:
                folder,file,want=expected
                actual=json.loads((result/folder/file).read_text())['status']
                if actual!=want:raise RuntimeError('Stage did not satisfy completion condition: '+actual)
        except Exception as exc:
            state['spent_seconds']+=time.monotonic()-started
            state.update(state='stopped',error_type=type(exc).__name__,automatic_retry=False,updated_utc=utc())
            write_json(state_file,state);raise
        state['spent_seconds']+=time.monotonic()-started
        state['completed'].append(name)
        state.pop('stage_pid',None);write_json(state_file,state)
    state.update(state='awaiting_validation',stage='R2-2-required',updated_utc=utc())
    write_json(state_file,state)


if __name__=='__main__':main()
