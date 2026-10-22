"""Immutable source and asset checks for finalization only."""
import hashlib
import json
from pathlib import Path
import sys
import os
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)


def sources(root):
    from round3_m2_common import sources as prior_sources
    prior_sources(root)
    receipt=read(root/'provenance/final_implementation_receipt_v1.json')
    if not receipt['remote_main_verified']:raise ValueError('Final entry not pushed')
    for n,h in receipt['files_sha256'].items():
        if sha(root/n)!=h:raise ValueError('Final source hash mismatch: '+n)
    cfg=read(root/'configs/final_v1.json')
    if sha(root/cfg['method_config'])!=cfg['method_config_sha256']:raise ValueError('Method config changed')
    if sha(root/cfg['test_manifest'])!=cfg['test_manifest_sha256']:raise ValueError('Test manifest changed')
    return cfg,read(root/cfg['method_config']),receipt


def asset_checks(root,cfg,dataset):
    required={'weights/GECO2FSCD.pth':cfg['geco_sha256'],
              'cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt':cfg['sam2_sha256']}
    manifest=read(root/cfg['test_manifest'])
    for n,h in manifest['annotation_sha256'].items():required['data/fsc147/annotations/'+n]=h
    if dataset=='fsc147-test':
        for row in manifest['files']:required[row['path']]=row['sha256']
    elif dataset=='gate':
        val=read(root/'manifests/fsc147_val.json')
        for row in val['files']:required[row['path']]=row['sha256']
    elif dataset=='carpk':
        carpk=read(root/'manifests/carpk_test_v1.json')
        if carpk['split']!='test' or not carpk['license_confirmed']:raise ValueError('CARPK not authorized')
        for row in carpk['files']:
            for key in ('image','annotation'):required[row[key+'_path']]=row[key+'_sha256']
    else:raise ValueError('Unknown dataset')
    for n,h in required.items():
        if sha(root/n)!=h:raise ValueError('Asset hash mismatch: '+n)
    return {'verified_files':len(required),'geco_sha256':cfg['geco_sha256'],'sam2_sha256':cfg['sam2_sha256']}
