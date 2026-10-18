"""Gate, smoke, or complete val. Ground truth is used only after inference."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from round3_common import sources,assets,runtime,write_json,sha,utc,read
from run_m2_m3 import load_model,smoke_ids
from planb.round3_m3 import M3Engine,require_equal
from planb.geco_adapter import postprocess
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.common import state_sha
from planb.round2.spatial import cell_mass,point_cells,grid_features,localization


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['gate','smoke','full'],required=True)
    p.add_argument('--variant',default='m3-v2-g2');a=p.parse_args();root=a.root.resolve()
    cfg,splits,receipt=sources(root)
    if a.variant not in cfg['m3']['runs']:raise ValueError('Unknown variant')
    label='gate' if a.phase=='gate' else a.phase+'-'+a.variant
    out=root/'results/round3-v1'/label
    if out.exists():raise FileExistsError('Existing evidence')
    out.mkdir(parents=True);start=time.monotonic()
    try:
        runtime(root);names,baseline,verified=assets(root,cfg,splits)
        if a.phase!='full':names=smoke_ids(names,baseline)
        record={**receipt,**verified,'phase':a.phase,'variant':a.variant,'split':'val','ids':names,
                'started_utc':utc(),'config_sha256':sha(root/'configs/round3_v1.json'),
                'precision':'FP32','tf32':False,'torch':torch.__version__,'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(),'ground_truth_used_by_policy':False}
        write_json(out/'protocol.json',record)
        model=load_model(root);frozen=state_sha(model);engine=M3Engine(model,cfg)
        annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
        maps=root/'work/round3-v1'/label
        if a.phase!='gate':maps.mkdir(parents=True,exist_ok=False)
        torch.cuda.reset_peak_memory_stats();rows=[];gate=[]
        with torch.inference_mode(),(out/'predictions.jsonl').open('x') as stream:
            for name in names:
                with Image.open(root/'data/fsc147/images_384_VarV2'/name) as im:image=T.ToTensor()(im.convert('RGB'))
                h,w=image.shape[-2:];anno=annotations[name]
                prompts=torch.tensor(anno['box_examples_coordinates'],dtype=torch.float32)[:3,[0,2],:].reshape(-1,4)
                if prompts.shape!=(3,4) or not ((prompts[:,2:]-prompts[:,:2])>0).all():raise ValueError('Invalid exemplars')
                torch.cuda.synchronize();tick=time.monotonic()
                if a.phase=='gate':
                    x,prompts_gpu,scale=engine.prepare(image,prompts)
                    original=model(x,prompts_gpu)
                    adapted,f,app=engine.adapter.forward(x,prompts_gpu)
                    require_equal(original[0][0],adapted)
                    boxes,_=postprocess(adapted,int(w*scale),int(h*scale))
                    observed=boxes.cpu().numpy()*(1024/scale)
                    expected=np.asarray(baseline[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)
                    if not np.array_equal(observed,expected):raise ValueError('Disabled features changed B0 boxes')
                    gate.append({'image':name,'bitwise_equal':True,'count':len(observed)})
                    del original,adapted,f,app,x,prompts_gpu,boxes
                    engine.check_memory()
                    write_json(out/'progress.json',{'done':len(gate),'total':len(names),'updated_utc':utc()})
                    continue
                boxes,scores,first,detail=engine.run(image,prompts,a.variant)
                torch.cuda.synchronize();inference=time.monotonic()-tick
                expected=np.asarray(baseline[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)
                if not np.array_equal(first,expected):raise ValueError('Initial pass differs from B0')
                points=points_for_game(anno['points'],h,w)
                density=boxes_to_density(boxes,h,w)
                if abs(float(density.sum())-len(boxes))>1e-8:raise ValueError('Mass conservation failed')
                mass=cell_mass(density);_,truth_cells=point_cells(points,h,w)
                path=maps/(Path(name).stem+'.npz');np.savez_compressed(path,density=density)
                row={'image':name,'truth':len(points),'prediction':len(boxes),'width':w,'height':h,
                     'density_mass':float(density.sum()),'game1':game(density,points,1),'game2':game(density,points,2),
                     'inference_seconds_per_image':inference,'boxes_xyxy':boxes.tolist(),'objectness':scores.tolist(),
                     'exemplars_xyxy':prompts.tolist(),'mass8':mass.tolist(),'truth8':truth_cells.tolist(),
                     'grid_features':grid_features(boxes,scores,prompts.numpy(),density).tolist(),
                     'localization':localization(boxes,points,prompts.numpy()),
                     'density_array_sha256':__import__('hashlib').sha256(density.tobytes()).hexdigest(),
                     'density_file_sha256':sha(path),'detail':detail}
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
                write_json(out/'progress.json',{'done':len(rows),'total':len(names),'elapsed_seconds':time.monotonic()-start,
                           'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'updated_utc':utc()})
                if len(rows)%25==0 or len(rows)==len(names):print(json.dumps({'done':len(rows),'total':len(names),'variant':a.variant}),flush=True)
        if state_sha(model)!=frozen:raise ValueError('Frozen model parameters changed')
        summary={'status':'passed' if a.phase=='gate' else 'complete','n':len(names),'phase':a.phase,'variant':a.variant,
                 'elapsed_seconds':time.monotonic()-start,'finished_utc':utc(),'frozen_sha256':frozen,
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                 'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        if a.phase=='gate':summary['rows']=gate
        write_json(out/('gate.json' if a.phase=='gate' else 'summary.json'),summary)
    except Exception as exc:
        write_json(out/'error.json',{'status':'failed','error_type':type(exc).__name__,
                   'elapsed_seconds':time.monotonic()-start,'automatic_retry':False,'updated_utc':utc()})
        raise


if __name__=='__main__':main()
