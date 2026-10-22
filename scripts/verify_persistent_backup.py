"""Copy an already sealed backup to mounted private storage and verify every object."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from final_common import read,write,sha,utc


def main(root,source,storage):
    if not storage.is_mount() or storage.stat().st_dev==source.stat().st_dev:
        raise ValueError('Target must be separately mounted storage')
    meta=read(source/'receipt.json');archive=source/'artifacts.tar.gz';start=time.monotonic()
    used=sum(p.stat().st_size for p in storage.rglob('*') if p.is_file())
    if used+meta['archive_bytes']+10_000_000>20_000_000_000:
        raise ValueError('Would exceed authorized free storage allowance')
    if sha(archive)!=meta['archive_sha256']:raise ValueError('Source archive changed')
    dest=storage/'counting-finalization-20261022';dest.mkdir(exist_ok=False)
    target=dest/'artifacts.tar.gz';shutil.copyfile(archive,target)
    if sha(target)!=meta['archive_sha256']:raise ValueError('Copied archive SHA mismatch')
    with tarfile.open(target,'r|gz') as tar:
        member=tar.next()
        if member.name!='manifest.json':raise ValueError('Missing manifest')
        raw=tar.extractfile(member).read()
        if hashlib.sha256(raw).hexdigest()!=meta['manifest_sha256']:raise ValueError('Manifest changed')
        manifest=json.loads(raw);(dest/'manifest.json').write_bytes(raw)
        expected={r['sha256']:r['bytes'] for r in manifest['files']};verified=set()
        for member in tar:
            if member.name=='manifest.json':continue
            key=member.name.removeprefix('objects/')
            if key not in expected or not member.isfile() or member.size!=expected[key] or key in verified:
                raise ValueError('Unexpected stored object')
            h=hashlib.sha256();f=tar.extractfile(member)
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            if h.hexdigest()!=key:raise ValueError('Stored object SHA mismatch')
            verified.add(key)
        if verified!=set(expected):raise ValueError('Missing stored objects')
    receipt={'status':'passed','off_instance_all_objects_verified':True,
             'destination':'owner-private AutoDL regional file storage, separate from instance disks',
             'within_authorized_free_20GB':True,'archive_sha256':meta['archive_sha256'],
             'archive_bytes':meta['archive_bytes'],'manifest_sha256':meta['manifest_sha256'],
             'logical_files':manifest['file_count'],'unique_objects':len(verified),
             'prior_checkpoint_sha_verified':meta['prior_checkpoint_sha_verified'],
             'prior_file_sha_verified_density_arrays':meta['prior_file_sha_verified_density_arrays'],
             'elapsed_seconds':time.monotonic()-start,'finished_utc':utc()}
    write(dest/'verification.json',receipt);write(root/'provenance/final_backup_verified_v1.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True);a=p.parse_args();main(a.root.resolve(),a.source.resolve(),a.storage.resolve())
