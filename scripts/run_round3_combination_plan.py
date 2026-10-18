"""Plan R3-4 from immutable audited evidence before deciding missing GPU work."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_calibration_io import load,read,sha,write,utc
from planb.round3_combination_plan import make_plan

PLAN_FILES=('planb/round3_combination_plan.py','scripts/run_round3_combination_plan.py',
            'scripts/audit_round3_combination_plan.py','tests/test_round3_combination_plan.py',
            'docs/ROUND3_COMBINATION_EXECUTION.md')


def inputs(root,asset_root,receipt_path):
    cfg,split,receipt,baseline,annotations,labels,evidence=load(root,asset_root,receipt_path)
    if any(n not in receipt['files_sha256'] for n in PLAN_FILES):raise ValueError('Incomplete combination plan receipt')
    cal=root/'results/round3-calibration-v1';audit=read(cal/'audit.json');execution=read(cal/'execution.json')
    if audit['status']!='passed' or execution['state']!='awaiting_publication':raise ValueError('Calibration is not audited')
    for n,h in audit['files_sha256'].items():
        if sha(cal/n)!=h:raise ValueError('Calibration evidence changed')
    old=read(cal/'protocol.json')['files_sha256']
    if len(old)!=99 or any(receipt['files_sha256'].get(n)!=h for n,h in old.items()):raise ValueError('Previous sources changed')
    m3=root/'results/round3-v1';manifest=read(m3/'m3_complete_files_v1.json')
    if manifest['status']!='passed' or manifest['scope']!='M3 stage only; no TTA or test':raise ValueError('M3 completion missing')
    for n,h in manifest['files_sha256'].items():
        if sha(m3/n)!=h:raise ValueError('M3 evidence changed')
    base={r['image']:r for r in baseline};names=list(base);variants={};hashes={}
    locations={v:m3/('full-'+v) for v in cfg['m3']['candidate_order']+cfg['m3']['mask_candidate_order']}
    locations.update({v:root/'results/round3-tta-v1'/('full-'+v) for v in cfg['tta']['candidate_order']})
    import json
    for variant,directory in locations.items():
        a=read(directory/'audit.json')
        if a['status']!='passed' or a['n']!=1286:raise ValueError('Variant audit missing')
        for n,h in a['files_sha256'].items():
            if sha(directory/n)!=h:raise ValueError('Variant source changed')
        data={}
        with (directory/'predictions.jsonl').open() as f:
            for line in f:
                r=json.loads(line);n=r['image']
                if n in data or n not in base or r['truth']!=base[n]['truth']:raise ValueError('Variant identity differs')
                data[n]={k:r[k] for k in ('image','truth','prediction')}
                if variant.startswith('m3-v2-g'):
                    if r['detail']['base_count']!=base[n]['prediction']:raise ValueError('Trigger baseline changed')
                    data[n]['triggered']=r['detail']['triggered']
        if list(data)!=names:raise ValueError('Variant order differs')
        variants[variant]=data;hashes[variant]=sha(directory/'predictions.jsonl')
    evidence={**evidence,'calibration_audit_sha256':sha(cal/'audit.json'),
              'calibration_models_sha256':sha(cal/'models.json'),'calibration_execution_sha256':sha(cal/'execution.json'),
              'm3_completion_sha256':sha(m3/'m3_complete_files_v1.json'),'variant_predictions_sha256':hashes}
    return cfg,split,receipt,names,labels,base,variants,read(cal/'models.json'),evidence


def main(root,asset_root,receipt_path):
    started=time.monotonic()
    cfg,split,receipt,names,labels,base,variants,cal,evidence=inputs(root,asset_root,receipt_path)
    out=root/'results/round3-combination-v1'
    if out.exists():raise FileExistsError('Combination plan exists; no automatic restart')
    out.mkdir()
    try:
        write(out/'plan_protocol_v1.json',{**receipt,**evidence,'started_utc':utc(),'device':'CPU',
              'gpu_seconds':0,'stage':'R3-4 planning only','instance_policy':'keep_running'})
        plan=make_plan(names,{'A':split['val_A'],'B':split['val_B']},labels,base,variants,cal,cfg)
        write(out/'plan_v1.json',plan)
        from audit_round3_combination_plan import audit_plan
        audit_plan(plan,names,{'A':split['val_A'],'B':split['val_B']},labels,base,variants,cal,cfg)
        write(out/'plan_audit_v1.json',{'status':'passed','n':len(names),'verified_utc':utc(),
              'module_choices_and_admission_independently_recomputed':True,
              'routing_and_missing_variants_independently_recomputed':True,
              'files_sha256':{n:sha(out/n) for n in ('plan_protocol_v1.json','plan_v1.json')}})
        write(out/'plan_execution_v1.json',{'state':'awaiting_combination_implementation',
              'completed_utc':utc(),'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,
              'automatic_retry':False,'instance_policy':'keep_running'})
    except BaseException as exc:
        write(out/'plan_error_v1.json',{'state':'failed','error_type':type(exc).__name__,'utc':utc(),
              'cpu_wall_seconds':time.monotonic()-started,'gpu_seconds':0,'automatic_retry':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--asset-root',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.asset_root.resolve(),a.receipt.resolve())
