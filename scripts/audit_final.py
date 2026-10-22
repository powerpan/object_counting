"""Rebuild frozen outputs and scores from saved predictions; no model loaded."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_common import read,write,sha,sources,utc
from planb.final_metrics import size_group,report,qualitative_selection
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.spatial import localization,cell_mass
from audit_round3_tta import rebuild_view


def compare(a,b,path='root'):
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Keys differ: '+path)
        for k in a:compare(a[k],b[k],path+'.'+k)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('Length differs: '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'.'+str(i))
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not np.isfinite(a) or not np.isclose(a,b,rtol=1e-12,atol=1e-8):raise ValueError('Numeric mismatch: '+path)
    elif a!=b:raise ValueError('Value mismatch: '+path)


def main(root,dataset):
    cfg,method,receipt=sources(root);out=root/'results/final-v1'/dataset
    if (out/'audit.json').exists():raise FileExistsError('Existing audit')
    started=time.monotonic();protocol=read(out/'protocol.json');summary=read(out/'summary.json')
    if summary['status']!='complete' or protocol['config_sha256']!=sha(root/'configs/final_v1.json'):raise ValueError('Incomplete or changed configuration')
    rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
    if [r['image'] for r in rows]!=protocol['ids'] or len(rows)!=len(set(protocol['ids'])):raise ValueError('Wrong sample list')
    if dataset=='carpk':
        man=read(root/'manifests/carpk_test_v1.json');records={r['image']:r for r in man['files']}
        if protocol['ids']!=man['ids']:raise ValueError('CARPK split mismatch')
    else:
        annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
        classes=dict(line.split('\t',1) for line in (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if line)
        if dataset=='fsc147-test' and protocol['ids']!=read(root/cfg['test_manifest'])['ids']:raise ValueError('Test split mismatch')
    max_pixel_diff=0.
    for row in rows:
        n=row['image'];h,w=row['height'],row['width']
        if dataset=='carpk':
            rec=records[n];prompts=np.asarray(rec['exemplars'],dtype=np.float32);points=points_for_game(rec['points'],h,w);category=rec['class']
        else:
            ann=annotations[n];prompts=np.asarray(ann['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(3,4)
            points=points_for_game(ann['points'],h,w);category=classes[n]
        compare(row['exemplars_xyxy'],prompts.tolist());compare(row['truth'],len(points));compare(row['class'],category)
        ratio,group=size_group(prompts,w,h,cfg['size_groups']['cutoffs']);compare(row['size_ratio'],ratio);compare(row['size_group'],group)
        detail=row['detail'];expected=detail['median_exemplar_side_model_pixels']<=32 or row['b0']['prediction']>=200
        compare(detail['triggered'],expected);compare(detail['route'],'M3-v2-g3' if expected else 'T2')
        if not detail['original_view_matches_b0']:raise ValueError('Missing paired B0 gate')
        if dataset=='gate' and not detail['native_adapter_gate_passed']:raise ValueError('Missing native gate')
        for label in ['b0','final']:
            result=row[label];boxes=np.asarray(result['boxes_xyxy'],dtype=np.float64).reshape(-1,4)
            if label=='b0' or expected:
                rebuilt=boxes_to_density(boxes,h,w);prediction=float(len(boxes))
            else:
                views=detail['method_detail']['views']
                if [(v['scale_factor'],v['flipped']) for v in views]!=[(1.,False),(1.,True)]:raise ValueError('Unregistered TTA views')
                independent=[rebuild_view(v,h,w) for v in views]
                rebuilt=(independent[0][0]+independent[1][0])/2
                prediction=(len(independent[0][1])+len(independent[1][1]))/2
            path=root/'work/final-v1'/dataset/(Path(n).stem+'-'+label+'.npz')
            if sha(path)!=result['density_file_sha256']:raise ValueError('Stored array file changed')
            density=np.load(path,allow_pickle=False)['density']
            if hashlib.sha256(density.tobytes()).hexdigest()!=result['density_array_sha256']:raise ValueError('Stored array changed')
            if not np.allclose(density,rebuilt,rtol=1e-12,atol=1e-8):raise ValueError('Density reconstruction mismatch')
            max_pixel_diff=max(max_pixel_diff,float(np.max(np.abs(density-rebuilt))))
            compare(result['prediction'],prediction);compare(result['density_mass'],float(rebuilt.sum()))
            compare(result['mass8'],cell_mass(rebuilt).tolist())
            for level in (1,2):compare(result['game'+str(level)],game(rebuilt,points,level))
            compare(result['localization'],localization(boxes,points,prompts))
    rebuilt_report=report(rows,cfg,dataset)
    for key,value in rebuilt_report.items():compare(summary[key],value,key)
    if dataset=='fsc147-test':compare(read(out/'qualitative_selection.json'),qualitative_selection(rows,cfg))
    write(out/'audit.json',{'status':'passed','n':len(rows),'checked_truth_prompts_routes_density_metrics':True,
          'count_game_localization_reconstructed':True,'bootstrap_recomputed_with_registered_algorithm':True,
          'density_max_pixel_difference':max_pixel_diff,'predictions_sha256':sha(out/'predictions.jsonl'),
          'summary_sha256':sha(out/'summary.json'),'elapsed_seconds':time.monotonic()-started,'finished_utc':utc()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',required=True,choices=['gate','fsc147-test','carpk'])
    a=p.parse_args();main(a.root.resolve(),a.dataset)
