"""Download the archive linked by the CARPK author. No password stored here."""
import argparse
from pathlib import Path
import time
from final_common import write,sha,utc


def main(root):
    import gdown
    target=root/'data/carpk-official/datasets.zip'
    if target.exists():raise FileExistsError('Archive already exists; inspect and verify it')
    target.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    # Public download URL from https://lafi.github.io/LPN/; not a credential.
    url='https://drive.google.com/uc?id=0BwSzgS8Mm48Ud2h2dW40Wko3a1E&export=download&resourcekey=0-34K_uP-vYM7EWq0Q2iIVaw'
    partial=target.with_suffix('.zip.partial')
    if partial.exists():raise FileExistsError('Partial download exists; no automatic retry')
    result=gdown.download(url,output=str(partial),quiet=True)
    if not result:raise RuntimeError('Official archive download failed')
    partial.replace(target)
    write(root/'provenance/carpk_download_v1.json',{'status':'downloaded_pending_extraction',
          'author_page':'https://lafi.github.io/LPN/','archive_sha256':sha(target),'bytes':target.stat().st_size,
          'license_accepted_by_owner':True,'password_stored':False,'finished_utc':utc(),
          'elapsed_seconds':time.monotonic()-started})
    print('Official archive downloaded; extraction and split checks remain.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
