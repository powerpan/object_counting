"""Fetch the frozen test image list server-side; require official per-file SHA."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import time
import requests
from final_common import read,write,sha,utc


def main(root):
    cfg=read(root/'configs/final_v1.json');manifest=read(root/cfg['test_manifest'])
    if sha(root/cfg['test_manifest'])!=cfg['test_manifest_sha256'] or manifest['n']!=1190:raise ValueError('Frozen list changed')
    receipt=root/'provenance/final_test_download_v1.json'
    if receipt.exists():raise FileExistsError('Download already completed; verify existing files')
    revision=read(root/'manifests/fsc147_val.json')['mirror_revision'];start=time.monotonic()
    def one(row):
        name=row['image'];p=root/row['path']
        if Path(name).name!=name or row['path']!='data/fsc147/images_384_VarV2/'+name:raise ValueError('Invalid image path')
        if p.exists():
            if sha(p)!=row['sha256']:raise ValueError('Existing test asset differs')
            return
        url='https://hf-mirror.com/datasets/isentropic/FSC147/resolve/'+revision+'/images_384_VarV2/'+name
        for attempt in range(3):
            try:
                resp=requests.get(url,timeout=(15,90));resp.raise_for_status()
                temp=p.with_suffix('.download-part');temp.write_bytes(resp.content)
                if temp.stat().st_size!=row['bytes'] or sha(temp)!=row['sha256']:raise ValueError('Official SHA mismatch')
                temp.replace(p);return
            except requests.RequestException:
                if attempt==2:raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,f in enumerate(as_completed([pool.submit(one,r) for r in manifest['files']]),1):
            f.result()
            if n%100==0 or n==1190:print('verified',n,'/1190',flush=True)
    write(receipt,{'status':'complete','n':1190,'manifest_sha256':sha(root/cfg['test_manifest']),
                   'mirror_revision':revision,'all_match_original_archive':True,'inference_performed':False,
                   'finished_utc':utc(),'elapsed_seconds':time.monotonic()-start})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
