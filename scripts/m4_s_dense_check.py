"""Fixed S checkpoint on all 43 dense training images; descriptive diagnosis only."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.data import Inputs
from planb.round2.head import DensityHead
from planb.round2.spatial import restore_density
from planb.round2.common import state_sha,sha,write_json,verify_inputs,check_memory
from planb.m4v2 import frozen_forward,diagnostic_label
from m4_v2_common import settings,begin,deadline,fail,read,model_and_adapter,count_metrics


def summary(rows):
    def stats(z):
        if not z:return {'n':0}
        return {**count_metrics(z),'median_S_over_truth':float(np.median([r['prediction']/r['truth'] for r in z])),
                'median_B0_over_truth':float(np.median([r['detector_prediction']/r['truth'] for r in z])),
                'maximum_S_prediction':max(r['prediction'] for r in z),
                'maximum_S_over_truth':max(r['prediction']/r['truth'] for r in z),
                'overcount_n':sum(r['prediction']>r['truth'] for r in z)}
    groups={part:{str(lo)+'-'+str(hi):stats([r for r in rows if r['original_partition']==part and lo<=r['truth']<hi])
                  for lo,hi in [(300,500),(500,1000),(1000,1000000)]} for part in ('fit','holdout')}
    chosen=[r['prediction']/r['truth'] for r in rows if r['original_partition']=='fit' and r['truth']>=500]
    if len(chosen)!=15:raise ValueError('Unexpected fixed diagnostic sample')
    ratio=float(np.median(chosen))
    return {'status':'complete','n':len(rows),'groups':groups,'decision_n':len(chosen),'decision_median_ratio':ratio,
            'descriptive_label':diagnostic_label(ratio),'causal_conclusion':False,'v2_settings_may_change':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--time-limit-seconds',type=float,required=True)
    a=p.parse_args();root=a.root.resolve();base,old,c,splits=settings(root);out,start=begin(root,'s-dense-check',a.time_limit_seconds,1800)
    try:
        assert read(root/'results/m4-v2-v1/initial-gate/summary.json')['status']=='passed'
        verify_inputs(root,base)
        manifest=read(root/'results/round2-v1/head-S-seed42/checkpoint_manifest.json')
        assert manifest['complete_epochs']==30 and manifest['best_epoch']==29
        path=root/'work/round2-v1/head-S-seed42/best.pth';assert sha(path)==manifest['best_sha256']
        protocol=read(out/'protocol.json');protocol.update(checkpoint_sha256=sha(path),ids=splits['s_dense_check'])
        write_json(out/'protocol.json',protocol)
        model,adapter=model_and_adapter(root);frozen=state_sha(model);assert frozen==manifest['frozen_model_sha256']
        head=DensityHead(base['m4']['widths']['S']).cuda().eval()
        saved=torch.load(path,map_location='cpu',weights_only=True);head.load_state_dict(saved['head']);del saved
        inputs=Inputs(root,splits['s_dense_check']);rows=[];maps=root/'work/m4-v2-v1/s-dense-check/maps';maps.mkdir(parents=True)
        with torch.no_grad(),(out/'predictions.jsonl').open('x') as stream:
            for name in splits['s_dense_check']:
                deadline(start,a.time_limit_seconds);sample=inputs.prepare(name,with_target=False)
                features,output,boxes,scores=frozen_forward(adapter,sample)
                q=head(features[:,:257],sample['mask'][None].cuda())
                d=restore_density(q,sample['height'],sample['width'],sample['valid_height'],sample['valid_width'])
                path=maps/(Path(name).stem+'.npz');np.savez_compressed(path,density=d)
                row={'image':name,'truth':len(inputs.annotations[name]['points']),'prediction':float(d.sum()),
                     'detector_prediction':len(boxes),'boxes_xyxy':boxes.tolist(),'width':sample['width'],'height':sample['height'],
                     'original_partition':'fit' if name in splits['s_dense_fit'] else 'holdout','density_map_sha256':sha(path)}
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                del features,output,q;check_memory()
        assert state_sha(model)==frozen;deadline(start,a.time_limit_seconds)
        write_json(out/'summary.json',{**summary(rows),'elapsed_seconds':time.monotonic()-start,
                   'frozen_state_sha256':frozen,'peak_allocated_mib':check_memory()})
    except BaseException as e:fail(out,start,e);raise


if __name__=='__main__':main()
