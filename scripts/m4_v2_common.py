"""Immutable receipts, assets and shared runtime for M4-v2 entries."""
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import setup,sha,write_json,utc,verify_inputs,execution_record,check_memory
from planb.m4v2 import expand_holdout

REGISTRATION='6ea52ea'


def read(path):return json.loads(Path(path).read_text())


def settings(root):
    base,original=setup(root)
    c=read(root/'configs/m4_v2.json')
    if sha(root/c['base_config'])!=c['base_config_sha256'] or sha(root/c['split_file'])!=c['split_sha256']:
        raise ValueError('M4-v2 registered configuration differs')
    splits=read(root/c['split_file'])
    labels_path=root/'data/fsc147/annotations/ImageClasses_FSC147.txt'
    if sha(labels_path)!=original['source_sha256']['ImageClasses_FSC147.txt']:raise ValueError('Class labels changed')
    labels=dict(x.strip().split(maxsplit=1) for x in labels_path.read_text().splitlines() if x.strip())
    annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    counts={n:len(annotations[n]['points']) for n in original['train_fit']+original['train_holdout']}
    if counts!=splits['counts']:raise ValueError('Registered counts differ from official annotations')
    fit,hold,added=expand_holdout(original,counts,labels)
    if fit!=splits['train_fit'] or hold!=splits['train_holdout'] or added!=splits['added_holdout_classes']:
        raise ValueError('Registered holdout differs from fixed ranking rule')
    if c['test_enabled'] or c['batch_size']!=8 or c['total_m4_seconds']!=108000:
        raise ValueError('Unexpected scope')
    return base,original,c,splits


def receipt(root,stage):
    original=execution_record(root,stage)
    p=root/'provenance/m4_v2_receipt_v1.json';r=read(p)
    if not r['remote_main_verified'] or not r['registration_commit'].startswith(REGISTRATION):
        raise ValueError('Missing M4-v2 preregistration')
    for n,h in r['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('M4-v2 source receipt mismatch: '+n)
    return {'stage':stage,'started_utc':utc(),'registration_commit':r['registration_commit'],
            'code_commit':r['code_commit'],'files_sha256':r['files_sha256'],
            'receipt_sha256':sha(p),'original_training_source':original,
            'config_sha256':sha(root/'configs/m4_v2.json'),'precision':'FP32','tf32':False}


def begin(root,stage,limit,cap):
    if not 0<limit<=cap:raise ValueError('Runtime may only tighten registered cap')
    out=root/'results/m4-v2-v1'/stage
    if out.exists():raise FileExistsError('Existing experiment evidence must not be overwritten')
    record=receipt(root,stage);record['time_limit_seconds']=limit
    out.mkdir(parents=True);write_json(out/'protocol.json',record)
    return out,time.monotonic()


def deadline(start,limit):
    if time.monotonic()-start>=limit:raise TimeoutError('Registered stage runtime exceeded')


def fail(out,start,exc):
    write_json(out/'error.json',{'status':'failed','error_type':type(exc).__name__,
               'elapsed_seconds':time.monotonic()-start,'updated_utc':utc(),'automatic_retry':False})


def model_and_adapter(root):
    from run_m2_m3 import load_model
    from planb.round2.adapter import Round2Adapter
    model=load_model(root);return model,Round2Adapter(model)


def count_metrics(rows,key='prediction'):
    import math
    if not rows: return {'n':0,'mae':None,'rmse':None}
    error=[r[key]-r['truth'] for r in rows]
    return {'n':len(rows),'mae':math.fsum(abs(e) for e in error)/len(rows),
            'rmse':math.sqrt(math.fsum(e*e for e in error)/len(rows))}
