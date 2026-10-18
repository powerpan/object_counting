"""R3-3 CPU calibration, after audited TTA completion and a pushed source receipt."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_calibration_io import load,prepare,report,write,rows_write,sha,utc
from planb.round3_calibration import crossfit


def main(root,asset_root,receipt_path):
    started=time.monotonic();cfg,split,receipt,baseline,annotations,labels,evidence=load(root,asset_root,receipt_path)
    out=root/'results/round3-calibration-v1'
    if out.exists():raise FileExistsError('Calibration outputs exist; no automatic restart')
    out.mkdir();work=root/'work/round3-calibration-v1';work.mkdir(parents=True,exist_ok=False)
    def stop(signum,frame):raise TimeoutError('Local CPU phase exceeded 600 seconds')
    old=signal.signal(signal.SIGALRM,stop);signal.alarm(max(1,int(600-(time.monotonic()-started))))
    try:
        write(out/'protocol.json',{**receipt,**evidence,'started_utc':utc(),'stage':'R3-3',
              'device':'CPU','gpu_seconds':0,'cpu_timeout_seconds':600,'split':'val','n':1286,
              'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
              'config_sha256':sha(root/'configs/round3_v1.json'),
              'normalization':'positive_mass_cells_of_fit_fold_only',
              'scope':'B0-based calibration; no combination or test'})
        records=prepare(baseline,annotations);names=[r['image'] for r in baseline];base={r['image']:r for r in baseline}
        rows_write(out/'input_cells.jsonl',[records[n] for n in names])
        result=crossfit(names,{'A':split['val_A'],'B':split['val_B']},records,labels,base,cfg['m4_v3'])
        for kind,rows in result['rows'].items():rows_write(out/(kind+'.jsonl'),rows)
        models={k:v for k,v in result.items() if k!='rows'};write(out/'models.json',models)
        beta_a=np.asarray(models['outer_models']['A']['regional']['coefficients'])[1:]
        beta_b=np.asarray(models['outer_models']['B']['regional']['coefficients'])[1:]
        summary={'status':'complete','scope':'development_only_no_test','n':1286,
                 'roles':{k:report(v,base,labels,cfg['bootstrap_replicates']) for k,v in result['rows'].items()},
                 'outer_choices':{k:{z:v[z] for z in ('selected','admitted','metrics')} for k,v in models['outer_decisions'].items()},
                 'all_val_final_choice':{z:models['final_decision'][z] for z in ('selected','admitted','metrics')},
                 'regional_slope_signs_agree':(np.sign(beta_a)==np.sign(beta_b)).tolist(),
                 'localization':'B0 boxes unchanged; calibration changes density mass only',
                 'runtime':'saved-output CPU analysis; no end-to-end inference timing claim',
                 'independent_audit':'pending','finished_utc':utc()}
        write(out/'summary.json',summary)
        elapsed=time.monotonic()-started
        with (work/'audit.log').open('x') as log:
            subprocess.run([sys.executable,'scripts/audit_round3_calibration.py','--root',str(root),
                            '--asset-root',str(asset_root),'--receipt',str(receipt_path)],cwd=root,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(1,590-elapsed))
        write(out/'execution.json',{'state':'awaiting_publication','completed_utc':utc(),
              'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,
              'automatic_retry':False,'instance_policy':'keep_running'})
    except BaseException as exc:
        write(out/'error.json',{'state':'failed','error_type':type(exc).__name__,'utc':utc(),
              'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,'automatic_retry':False})
        raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.asset_root.resolve(),a.receipt.resolve())
