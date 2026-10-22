"""Transport-only mirror; accept only exact author image bytes, never mirror labels."""
import argparse
import hashlib
from pathlib import Path
import time
import zipfile
import requests
from final_common import read,write,sha,utc

REV='0845566dba8b9f5abf0a6c1ac00dc348c18a8189'
URL='https://hf-mirror.com/datasets/ShantyCam/carpk/resolve/'+REV+'/carpk.zip?download=true'
ARCHIVE_SHA='b76234239f6a44c64adfef08b2ceddaecceb8ba670e966f952f9f1efd9eef08f'


def main(root):
    start=time.monotonic();manifest=read(root/'manifests/carpk_test_v1.json')
    receipt=root/'provenance/final_carpk_download_v1.json'
    if receipt.exists():raise FileExistsError('Existing completed download')
    archive=root/'work/assets/carpk-mirror.zip';archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():
        part=archive.with_suffix('.partial')
        with requests.get(URL,stream=True,timeout=(20,45)) as response:
            response.raise_for_status()
            with part.open('xb') as f:
                for chunk in response.iter_content(4*1024*1024):
                    if time.monotonic()-start>600:raise TimeoutError('Bounded asset download exceeded')
                    f.write(chunk)
        if sha(part)!=ARCHIVE_SHA:raise ValueError('Mirror transport archive hash mismatch')
        part.rename(archive)
    elif sha(archive)!=ARCHIVE_SHA:raise ValueError('Existing mirror changed')
    with zipfile.ZipFile(archive) as z:
        names={}
        for info in z.infolist():
            if not info.is_dir():names.setdefault(Path(info.filename).name,[]).append(info)
        for row in manifest['files']:
            matches=names.get(row['image'],[])
            if len(matches)!=1:raise ValueError('Missing or ambiguous official image '+row['image'])
            raw=z.read(matches[0])
            if hashlib.sha256(raw).hexdigest()!=row['image_sha256']:raise ValueError('Not identical to official image '+row['image'])
            dest=root/row['image_path'];dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if sha(dest)!=row['image_sha256']:raise ValueError('Existing image differs')
            else:dest.write_bytes(raw)
            if sha(root/row['annotation_path'])!=row['annotation_sha256']:raise ValueError('Official annotation differs')
    write(receipt,{'status':'complete','n':len(manifest['files']),'all_images_match_author_archive':True,
        'annotation_source':'author archive only; mirror annotations unused','transport_revision':REV,
        'transport_archive_sha256':ARCHIVE_SHA,'manifest_sha256':sha(root/'manifests/carpk_test_v1.json'),
        'inference_performed':False,'elapsed_seconds':time.monotonic()-start,'finished_utc':utc()})
    print('Verified official CARPK test images:',len(manifest['files']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
