"""One paired B0/frozen-final evaluation. No test selection or restart."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_common import read,write,sha,utc,sources,asset_checks
from round3_common import runtime
from run_m2_m3 import load_model,smoke_ids
from planb.final_inference import FinalEngine
from planb.final_metrics import size_group,report,qualitative_selection
from planb.density import points_for_game,game
from planb.round2.spatial import localization,cell_mass
from planb.round2.common import state_sha


def inputs(root,cfg,dataset):
    if dataset=='carpk':
        manifest=read(root/'manifests/carpk_test_v1.json')
        records={r['image']:r for r in manifest['files']}
        return manifest['ids'],records,None
    manifest=read(root/cfg['test_manifest'])
    ann=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
    classes=dict(line.split('\t',1) for line in
                 (root/'data/fsc147/annotations/ImageClasses_FSC147.txt').read_text().splitlines() if line)
    baseline=None
    if dataset=='gate':
        baseline={r['image']:r for r in map(json.loads,(root/'results/b0-val-b1-boundary-r1/predictions.jsonl').read_text().splitlines())}
        names=smoke_ids(list(baseline),baseline)
    else:names=manifest['ids']
    records={n:{'image_path':'data/fsc147/images_384_VarV2/'+n,'class':classes[n],
                'exemplars':np.asarray(ann[n]['box_examples_coordinates'],dtype=np.float32)[:3,[0,2],:].reshape(3,4).tolist(),
                'points':ann[n]['points']} for n in names}
    return names,records,baseline


def main(root,dataset):
    cfg,method,receipt=sources(root)
    out=root/'results/final-v1'/dataset
    if out.exists():raise FileExistsError('Existing evaluation; no implicit retry')
    out.mkdir(parents=True);start=time.monotonic()
    try:
        runtime(root);verified=asset_checks(root,cfg,dataset)
        names,records,baseline=inputs(root,cfg,dataset)
        write(out/'protocol.json',{'dataset':dataset,'ids':names,'n':len(names),'started_utc':utc(),
              'config_sha256':sha(root/'configs/final_v1.json'),'implementation':receipt,
              'assets':verified,'ground_truth_used_by_policy':False,'precision':'FP32','tf32':False,
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda})
        model=load_model(root);frozen=state_sha(model);engine=FinalEngine(model,method)
        maps=root/'work/final-v1'/dataset;maps.mkdir(parents=True,exist_ok=False);rows=[]
        with (out/'predictions.jsonl').open('x') as stream:
            for name in names:
                rec=records[name]
                with Image.open(root/rec['image_path']) as im:image=T.ToTensor()(im.convert('RGB'))
                h,w=image.shape[-2:];prompts=torch.tensor(rec['exemplars'],dtype=torch.float32)
                b,f,bd,fd,detail=engine.run_pair(image,prompts,gate=dataset=='gate')
                if baseline is not None:
                    if b['prediction']!=baseline[name]['prediction'] or not np.array_equal(
                            np.asarray(b['boxes_xyxy'],dtype=np.float32).reshape(-1,4),
                            np.asarray(baseline[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)):
                        raise ValueError('Clone B0 differs from saved validation B0')
                # Scoring starts only after inference. No truth is supplied to FinalEngine.
                points=points_for_game(rec['points'],h,w)
                ratio,group=size_group(prompts.numpy(),w,h,cfg['size_groups']['cutoffs'])
                row={'image':name,'class':rec['class'],'width':w,'height':h,'truth':len(points),
                     'exemplars_xyxy':prompts.tolist(),'size_ratio':ratio,'size_group':group,'detail':detail}
                for label,result,density in [('b0',b,bd),('final',f,fd)]:
                    if not np.isfinite(density).all() or (density<0).any() or abs(density.sum()-result['prediction'])>1e-6:
                        raise ValueError('Invalid density')
                    if result['peak_allocated_mib']>cfg['max_memory_allocated_mib']:raise MemoryError('Allocated memory limit')
                    path=maps/(Path(name).stem+'-'+label+'.npz');np.savez_compressed(path,density=density)
                    result.update(density_mass=float(density.sum()),game1=game(density,points,1),game2=game(density,points,2),
                        localization=localization(result['boxes_xyxy'],points,prompts.numpy()),mass8=cell_mass(density).tolist(),
                        density_file_sha256=sha(path),density_array_sha256=hashlib.sha256(density.tobytes()).hexdigest())
                    row[label]=result
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
                write(out/'progress.json',{'done':len(rows),'total':len(names),'elapsed_seconds':time.monotonic()-start,'updated_utc':utc()})
                if len(rows)%25==0 or len(rows)==len(names):print('completed',len(rows),'/',len(names),flush=True)
        if state_sha(model)!=frozen:raise ValueError('Model changed')
        write(out/'summary.json',{'status':'complete','finished_utc':utc(),'elapsed_seconds':time.monotonic()-start,
              'frozen_model_sha256':frozen,**report(rows,cfg,dataset)})
        if dataset=='fsc147-test':write(out/'qualitative_selection.json',qualitative_selection(rows,cfg))
    except BaseException as exc:
        write(out/'error.json',{'status':'failed','error_type':type(exc).__name__,'automatic_retry':False,
                              'elapsed_seconds':time.monotonic()-start,'updated_utc':utc()})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dataset',choices=['gate','fsc147-test','carpk'],required=True);a=p.parse_args();main(a.root.resolve(),a.dataset)
