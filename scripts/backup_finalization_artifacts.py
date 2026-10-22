"""Lossless content-addressed backup; never deserialize model checkpoints."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main(root, out):
    if out.exists(): raise FileExistsError('Backup output exists; verify it, do not overwrite')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu: raise RuntimeError('GPU is busy')
    ledger=json.loads((root/'results/round3-m2-v1/budget_ledger_v1.json').read_text())
    if ledger.get('active') is not None: raise RuntimeError('Unsettled prior work')
    state=json.loads((root/'results/round3-m2-v1/workflow_v1.json').read_text())
    if state['state']!='awaiting_publication': raise RuntimeError('Prior experiment incomplete')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try: cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except (OSError,ProcessLookupError): continue
        scripts=[Path(os.fsdecode(x)).name for x in cmd if x.endswith(b'.py')]
        if any(n.startswith(('run_round','continue_round','run_m4','round2_m4_train','train_m4')) for n in scripts):
            raise RuntimeError('Experiment controller or training process is still alive')
    out.mkdir(parents=True)
    candidates=[]
    for directory in ['results','logs','provenance','manifests']:
        candidates += [p for p in (root/directory).rglob('*') if p.is_file() and not p.is_symlink()]
    candidates += [p for p in (root/'work').rglob('*') if p.is_file() and not p.is_symlink()
                   and p.suffix in ('.npz','.npy','.pt','.pth','.ckpt','.log','.json','.jsonl','.txt','.exit')
                   and out not in p.parents]
    candidates += [p for p in root.iterdir() if p.is_file() and p.suffix in ('.log','.exit')]
    # Original density/checkpoint digests are checked where the prior evidence provides them.
    expected={}; expected_checkpoint=set()
    for p in (root/'results').rglob('predictions.jsonl'):
        for line in p.read_text().splitlines():
            row=json.loads(line)
            if row.get('density_file_sha256'):
                path=Path('work')/p.parent.relative_to(root/'results')/(Path(row['image']).stem+'.npz')
                if (root/path).exists(): expected[str(path)]=row['density_file_sha256']
    # The cancelled L and safety-stopped v2 saved digests in their closure
    # records, rather than the normal completed-training manifest.
    records = list((root/'results').rglob('*checkpoint*manifest*.json'))
    records += [root/'results/m4-v2-v1/train-seed42/summary.json',
                root/'results/round2-v1/L_user_cancellation_v1.json']
    for p in records:
        def collect(obj):
            if isinstance(obj,dict):
                for k,v in obj.items():
                    if 'sha256' in k and isinstance(v,str) and len(v)==64: expected_checkpoint.add(v)
                    else: collect(v)
            elif isinstance(obj,list):
                for v in obj: collect(v)
        collect(json.loads(p.read_text()))
    files=[]; unique={}; checkpoints=0; verified_density=0
    for p in sorted(set(candidates)):
        name=str(p.relative_to(root));h=sha(p)
        if name in expected:
            if h!=expected[name]: raise ValueError('Prior density SHA mismatch: '+name)
            verified_density+=1
        if p.suffix in ('.pth','.pt','.ckpt'):
            if h not in expected_checkpoint: raise ValueError('Checkpoint missing prior SHA evidence: '+name)
            checkpoints+=1
        files.append({'path':name,'sha256':h,'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777})
        unique.setdefault(h,p)
    manifest={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'all saved work density arrays and trained checkpoints; all results/logs/provenance/manifests',
              'official_data_and_weights_excluded':True,'checkpoint_content_loaded':False,
              'files':files,'file_count':len(files),'unique_objects':len(unique),
              'logical_bytes':sum(f['bytes'] for f in files),
              'stored_object_bytes':sum(p.stat().st_size for p in unique.values()),
              'prior_file_sha_verified_density_arrays':verified_density,'prior_checkpoint_sha_verified':checkpoints}
    content=(json.dumps(manifest,indent=2)+'\n').encode()
    (out/'manifest.json').write_bytes(content)
    with tarfile.open(out/'artifacts.tar.gz','w:gz',compresslevel=1) as tar:
        info=tarfile.TarInfo('manifest.json');info.size=len(content);tar.addfile(info,io.BytesIO(content))
        for h,p in unique.items():
            info=tarfile.TarInfo('objects/'+h);info.size=p.stat().st_size;info.mode=0o600
            with p.open('rb') as f: tar.addfile(info,f)
    receipt={k:v for k,v in manifest.items() if k!='files'}
    receipt.update(archive_sha256=sha(out/'artifacts.tar.gz'),archive_bytes=(out/'artifacts.tar.gz').stat().st_size,
                   manifest_sha256=sha(out/'manifest.json'),state='awaiting_off_instance_copy_and_verification')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.output.resolve())
