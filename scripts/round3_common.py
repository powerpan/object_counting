"""Read-only source and input checks for the registered third round."""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.common import sha,write_json,utc,seed_everything,PIN

REGISTRATION='ab4f53a'
RECEIPTS=['round2_code_receipt.json','round2_budget_receipt.json','round2_analysis_receipt.json',
          'round2_validation_workflow_receipt.json','round2_s_first_receipt.json','m4_v2_receipt_v1.json',
          'round3_m3_receipt_v1.json']


def read(path):return json.loads(Path(path).read_text())


def sources(root):
    receipts={}
    for name in RECEIPTS:
        r=read(root/'provenance'/name)
        for file,h in r['files_sha256'].items():
            if sha(root/file)!=h:raise ValueError('Source receipt mismatch: '+file)
        receipts[name]=sha(root/'provenance'/name)
    r=read(root/'provenance/round3_m3_receipt_v1.json')
    if not r['registration_commit'].startswith(REGISTRATION) or not r['remote_main_verified']:
        raise ValueError('Round 3 has not been preregistered and pushed')
    cfg=read(root/'configs/round3_v1.json');splits=read(root/cfg['split_file'])
    if sha(root/cfg['split_file'])!=cfg['split_sha256'] or cfg['final_test']['enabled']:
        raise ValueError('Unexpected split or test scope')
    return cfg,splits,{'registration_commit':r['registration_commit'],'code_commit':r['code_commit'],
                      'files_sha256':r['files_sha256'],'receipts_sha256':receipts}


def assets(root,cfg,splits):
    bdir=root/cfg['baseline_result'];bp=read(bdir/'protocol.json');cp=read(bdir/'checkpoint_load.json')
    manifest=read(root/'manifests/fsc147_val.json')
    if manifest['split']!='val' or manifest['n']!=1286:raise ValueError('Wrong input manifest')
    for entry in manifest['files']:
        if sha(root/entry['path'])!=entry['sha256']:raise ValueError('Validation asset hash mismatch')
    required={'weights/GECO2FSCD.pth':cp['fsc_sha256'],
              'cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt':cp['sam2_sha256']}
    for name,h in splits['source_sha256'].items():required['data/fsc147/annotations/'+name]=h
    for file,h in required.items():
        if sha(root/file)!=h:raise ValueError('Frozen asset mismatch')
    official=read(root/'data/fsc147/annotations/Train_Test_Val_FSC_147.json')
    names=official['val']
    if names!=bp['ids'] or len(names)!=1286 or len(set(names))!=1286:
        raise ValueError('Validation order differs')
    if set(names)!=set(splits['val_A']+splits['val_B']) or set(splits['val_A'])&set(splits['val_B']):
        raise ValueError('Class folds do not partition validation')
    if set(names)&set(official['test']+official['train']):raise ValueError('Split overlap')
    baseline={r['image']:r for r in map(json.loads,(bdir/'predictions.jsonl').read_text().splitlines())}
    return names,baseline,{'val_files_verified':len(manifest['files']),
                          'manifest_sha256':sha(root/'manifests/fsc147_val.json'),
                          'checkpoint_sha256':cp['fsc_sha256'],'sam2_sha256':cp['sam2_sha256'],
                          'baseline_predictions_sha256':sha(bdir/'predictions.jsonl')}


def runtime(root):
    up=root/'external/GECO2'
    if subprocess.check_output(['git','-C',str(up),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('Unexpected upstream')
    subprocess.run(['git','-C',str(up),'diff','--quiet'],check=True)
    subprocess.run(['git','-C',str(up),'diff','--cached','--quiet'],check=True)
    os.environ['TORCH_HOME']=str(root/'cache/torch');sys.path.insert(0,str(up))
    seed_everything(42)
