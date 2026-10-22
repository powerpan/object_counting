"""Package audited numerical evidence without source images, weights, or raw logs."""
import argparse
import copy
from pathlib import Path
import shutil
import tarfile
from final_common import read,write,sha,utc
import json


def main(root,dataset):
    source=root/'results/final-v1'/dataset;audit=read(source/'audit.json')
    if audit['status']!='passed' or audit['predictions_sha256']!=sha(source/'predictions.jsonl') or audit['summary_sha256']!=sha(source/'summary.json'):
        raise ValueError('Missing or changed audited results')
    dest=root/'results/final-public-v1'/dataset;dest.mkdir(parents=True,exist_ok=False)
    for n in ['summary.json','audit.json','protocol.json']:
        shutil.copyfile(source/n,dest/n)
    rows=[json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()]
    fields=['prediction','density_mass','game1','game2','localization','inference_seconds',
            'peak_allocated_mib','peak_reserved_mib','density_file_sha256','density_array_sha256']
    with (dest/'counts.jsonl').open('x') as f:
        for r in rows:
            small={k:r[k] for k in ['image','class','width','height','truth','exemplars_xyxy','size_ratio','size_group']}
            small['detail']={k:r['detail'][k] for k in ['route','triggered','original_view_matches_b0','median_exemplar_side_model_pixels']}
            for m in ['b0','final']:small[m]={k:r[m][k] for k in fields}
            f.write(json.dumps(small,allow_nan=False,separators=(',',':'))+'\n')
    if dataset=='fsc147-test':
        selected=read(source/'qualitative_selection.json');write(dest/'qualitative_selection.json',selected)
        ids={n for group in selected.values() for n in group}
        write(dest/'qualitative_rows.json',[r for r in rows if r['image'] in ids])
    write(dest/'raw_evidence.json',{'raw_predictions_sha256':sha(source/'predictions.jsonl'),
         'raw_predictions_retained_in_private_backup':True,'compact_counts_do_not_include_all_boxes':True})
    manifest={p.name:sha(p) for p in sorted(dest.iterdir()) if p.is_file()}
    write(dest/'files.json',{'dataset':dataset,'finished_utc':utc(),'files_sha256':manifest})
    archive=root/'work/final-v1'/('export-'+dataset+'.tar.xz')
    with tarfile.open(archive,'x:xz') as tar:
        for p in sorted(dest.iterdir()):tar.add(p,arcname=p.name,recursive=False)
    print(json.dumps({'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(manifest)+1}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=['gate','fsc147-test','carpk'],required=True)
    a=p.parse_args();main(a.root.resolve(),a.dataset)
