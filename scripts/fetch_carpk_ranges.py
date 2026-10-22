"""Download only official test images from a fixed ZIP via verified HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from pathlib import Path
import struct
import time
import zlib
import requests
from final_common import read,write,sha,utc

URL='https://hf-mirror.com/datasets/ShantyCam/carpk/resolve/0845566dba8b9f5abf0a6c1ac00dc348c18a8189/carpk.zip?download=true'
SIZE=2081580450


def directory(raw,offset):
    entries=[];i=0
    while raw[i:i+4]==b'PK\x01\x02':
        v=struct.unpack('<4s6H3L5H2L',raw[i:i+46]);nl,el,cl=v[10:13]
        name=raw[i+46:i+46+nl].decode('utf-8')
        entry={'name':name,'offset':v[16],'compressed':v[8],'size':v[9],'method':v[4],'flags':v[3]}
        extra=raw[i+46+nl:i+46+nl+el];j=0;zip64=None
        while j+4<=len(extra):
            tag,length=struct.unpack('<HH',extra[j:j+4]);j+=4
            if tag==1:zip64=extra[j:j+length]
            j+=length
        q=0
        for key in ['size','compressed','offset']:
            if entry[key]==0xffffffff:
                if zip64 is None or q+8>len(zip64):raise ValueError('Missing ZIP64 size/offset')
                entry[key]=struct.unpack('<Q',zip64[q:q+8])[0];q+=8
        entries.append(entry)
        i+=46+nl+el+cl
    if i!=len(raw):raise ValueError('Unexpected central directory')
    entries.sort(key=lambda v:v['offset'])
    for j,e in enumerate(entries):e['end']=(entries[j+1]['offset'] if j+1<len(entries) else offset)-1
    return entries


def member(raw,entry):
    h=struct.unpack('<4s5H3L2H',raw[:30])
    if h[0]!=b'PK\x03\x04' or h[2]&1 or entry['flags']&1:raise ValueError('Unexpected/encrypted transport entry')
    start=30+h[9]+h[10]
    if raw[30:30+h[9]].decode('utf-8')!=entry['name']:raise ValueError('Entry name mismatch')
    data=raw[start:start+entry['compressed']]
    if entry['method']==8:data=zlib.decompress(data,-15)
    elif entry['method']!=0:raise ValueError('Unsupported transport compression')
    if len(data)!=entry['size']:raise ValueError('Entry length mismatch')
    return data


def main(root):
    start=time.monotonic();manifest=read(root/'manifests/carpk_test_v1.json')
    receipt=root/'provenance/final_carpk_download_v1.json'
    if receipt.exists():raise FileExistsError('Completed download exists')
    prefix=root/'work/assets/carpk-mirror.partial'
    def get(a,b):
        if not (0<=a<=b<SIZE) or b-a>10_000_000:raise ValueError('Unbounded range')
        if prefix.exists() and prefix.stat().st_size>b:
            with prefix.open('rb') as f:f.seek(a);return f.read(b-a+1)
        for attempt in range(2):
            if time.monotonic()-start>900:raise TimeoutError('Asset transfer limit')
            try:
                with requests.get(URL+f'&range={a}-{b}',headers={'Range':f'bytes={a}-{b}'},stream=True,timeout=(15,30)) as r:
                    if r.status_code!=206 or r.headers.get('content-range')!=f'bytes {a}-{b}/{SIZE}':
                        raise ValueError('Server did not honor exact range')
                    data=r.raw.read(b-a+1)
                    if len(data)!=b-a+1:raise IOError('Truncated range')
                    return data
            except (requests.RequestException,IOError):
                if attempt:raise
    tail=get(SIZE-65536,SIZE-1);i=tail.rfind(b'PK\x05\x06')
    if i<0:raise ValueError('Missing ZIP directory')
    eocd=struct.unpack('<4s4H2LH',tail[i:i+22]);length,offset=eocd[5:7]
    central=get(offset,offset+length-1);entries=directory(central,offset);lookup={}
    for e in entries:lookup.setdefault(Path(e['name']).name,[]).append(e)
    def one(row):
        dest=root/row['image_path']
        if dest.exists():
            if sha(dest)!=row['image_sha256']:raise ValueError('Existing image mismatch')
            return
        matches=lookup.get(row['image'],[])
        if len(matches)!=1:raise ValueError('Missing or ambiguous image')
        entry=matches[0]
        if entry['size']>10_000_000:raise ValueError('Unexpected image size')
        data=member(get(entry['offset'],entry['end']),entry)
        if hashlib.sha256(data).hexdigest()!=row['image_sha256']:raise ValueError('Not identical to author image')
        if sha(root/row['annotation_path'])!=row['annotation_sha256']:raise ValueError('Official label changed')
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for n,f in enumerate(as_completed([pool.submit(one,r) for r in manifest['files']]),1):
            f.result()
            if n%25==0 or n==len(manifest['files']):print('verified',n,'/',len(manifest['files']),flush=True)
    write(receipt,{'status':'complete','n':len(manifest['files']),'all_images_match_author_archive':True,
        'transport':'fixed mirror ZIP byte ranges, eight workers, no mirror labels',
        'transport_revision':'0845566dba8b9f5abf0a6c1ac00dc348c18a8189',
        'central_directory_sha256':hashlib.sha256(central).hexdigest(),'whole_transport_archive_downloaded':False,
        'manifest_sha256':sha(root/'manifests/carpk_test_v1.json'),'inference_performed':False,
        'elapsed_seconds':time.monotonic()-start,'finished_utc':utc()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root.resolve())
