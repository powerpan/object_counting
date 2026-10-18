"""Shared, fail-closed bookkeeping for the second round."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch


PIN = '67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b'
REGISTRATION = 'ce0fc1a'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        h.update(key.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def setup(root):
    root = root.resolve(); upstream = root/'external/GECO2'
    if subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('Unexpected upstream revision')
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet'], check=True)
    os.environ['TORCH_HOME'] = str(root/'cache/torch')
    sys.path.insert(0, str(upstream)); sys.path.insert(0, str(root/'scripts'))
    cfg = json.loads((root/'configs/round2_v1.json').read_text())
    if sha(root/cfg['split_file']) != cfg['split_sha256']:
        raise ValueError('Fixed split file changed')
    if sha(root/cfg['train_manifest']) != cfg['train_manifest_sha256']:
        raise ValueError('Train manifest changed')
    seed_everything(cfg['seed'])
    return cfg, json.loads((root/cfg['split_file']).read_text())


def execution_record(root, stage):
    receipt = json.loads((root/'provenance/round2_code_receipt.json').read_text())
    if not receipt['registration_commit'].startswith(REGISTRATION):
        raise ValueError('Missing preregistration')
    for name, value in receipt['files_sha256'].items():
        if sha(root/name) != value:
            raise ValueError('Code differs from committed receipt: '+name)
    return {'stage': stage, 'started_utc': utc(), 'registration_commit': receipt['registration_commit'],
            'registration_commit_time': receipt['registration_commit_time'],
            'code_commit': receipt['code_commit'], 'files_sha256': receipt['files_sha256'],
            'config_sha256': sha(root/'configs/round2_v1.json'), 'precision':'FP32', 'tf32':False,
            'torch':torch.__version__, 'cuda':torch.version.cuda}


def check_memory(limit=40000):
    allocated = torch.cuda.max_memory_allocated()/2**20
    if allocated > limit:
        raise RuntimeError('Preregistered allocated-memory limit exceeded')
    return allocated


def verify_inputs(root, cfg):
    base = json.loads((root/cfg['baseline_result']/'protocol.json').read_text())
    required = {'data/fsc147/annotations/annotation_FSC147_384.json': base['annotation_sha256'],
                'data/fsc147/annotations/Train_Test_Val_FSC_147.json': base['split_sha256'],
                'weights/GECO2FSCD.pth': base['checkpoint_sha256']}
    sam = json.loads((root/cfg['baseline_result']/'checkpoint_load.json').read_text())
    required['cache/torch/hub/checkpoints/sam2_hiera_base_plus.pt'] = sam['sam2_sha256']
    download = json.loads((root/'provenance/training_download.json').read_text())
    if download['status'] != 'complete' or download['manifest_sha256'] != cfg['train_manifest_sha256']:
        raise ValueError('Training download has not passed verification')
    manifest = json.loads((root/cfg['train_manifest']).read_text())
    required.update({x['path']: x['sha256'] for x in manifest['files']})
    for name, value in required.items():
        if sha(root/name) != value:
            raise ValueError('Input hash mismatch: '+name)
    return {'status':'passed', 'verified_files':len(required),
            'train_images':3659, 'train_manifest_sha256':cfg['train_manifest_sha256'],
            'fsc_checkpoint_sha256':base['checkpoint_sha256'], 'sam2_sha256':sam['sam2_sha256']}
