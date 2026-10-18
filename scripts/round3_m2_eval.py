"""Frozen M2-v2 gate and val inference; point labels are read only after policy."""
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
from round3_m2_common import sources,assets,runtime,read,sha,write_json,utc
from run_m2_m3 import load_model,smoke_ids
from planb.round3_m2 import M2Engine
from planb.round3_m3 import require_equal
from planb.geco_adapter import postprocess
from planb.density import boxes_to_density,points_for_game,game
from planb.round2.common import state_sha
from planb.round2.spatial import cell_mass,point_cells,localization


def main(root,phase,candidate,repeat):
    cfg,splits,m2,receipt=sources(root)
    if candidate not in m2['candidates']:raise ValueError('Unregistered M2 candidate')
    variant=candidate+('-repeat' if repeat else '')
    label='gate' if phase=='gate' else phase+'-'+variant
    out=root/'results/round3-m2-v1'/label
    if out.exists():raise FileExistsError('M2 evidence exists; no restart')
    out.mkdir(parents=True);started=time.monotonic()
    try:
        runtime(root);names,baseline,verified=assets(root,cfg,splits)
        if phase!='full':names=smoke_ids(names,baseline)
        write_json(out/'protocol.json',{**receipt,**verified,'phase':phase,'candidate':candidate,'repeat':repeat,
              'variant':variant,'ids':names,'split':'val','started_utc':utc(),'precision':'FP32','tf32':False,
              'config_sha256':sha(root/'configs/round3_v1.json'),'m2_config_sha256':sha(root/'configs/round3_m2_v1.json'),
              'ground_truth_used_by_policy':False,'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(),'no_fallback_reuses_saved_second_pass':not repeat})
        model=load_model(root);frozen=state_sha(model);engine=M2Engine(model,cfg,m2)
        annotations=read(root/'data/fsc147/annotations/annotation_FSC147_384.json')
        maps=root/'work/round3-m2-v1'/label
        if phase!='gate':maps.mkdir(parents=True,exist_ok=False)
        torch.cuda.reset_peak_memory_stats();gate=[];done=0
        with torch.inference_mode(),(out/'predictions.jsonl').open('x') as stream:
            for name in names:
                with Image.open(root/'data/fsc147/images_384_VarV2'/name) as im:image=T.ToTensor()(im.convert('RGB'))
                h,w=image.shape[-2:];anno=annotations[name]
                exemplars=torch.tensor(anno['box_examples_coordinates'],dtype=torch.float32)[:3,[0,2],:].reshape(-1,4)
                torch.cuda.synchronize();tick=time.monotonic()
                if phase=='gate':
                    x,prompts,scale=engine.prepare(image,exemplars)
                    native=model(x,prompts);adapted,f,a=engine.adapter.forward(x,prompts)
                    require_equal(native[0][0],adapted)
                    boxes,_=postprocess(adapted,int(w*scale),int(h*scale))
                    expected=np.asarray(baseline[name]['boxes_xyxy'],np.float32).reshape(-1,4)
                    if not np.array_equal(boxes.cpu().numpy()*(1024/scale),expected):raise ValueError('M2 gate differs from B0')
                    gate.append({'image':name,'bitwise_equal':True,'count':len(boxes)})
                    del native,adapted,f,a,x,prompts,boxes
                    engine.check_memory()
                else:
                    boxes,scores,first,first_scores,detail=engine.run(image,exemplars,candidate,repeat)
                    torch.cuda.synchronize();inference=time.monotonic()-tick
                    if not np.array_equal(first,np.asarray(baseline[name]['boxes_xyxy'],np.float32).reshape(-1,4)):
                        raise ValueError('M2 first pass differs from B0')
                    points=points_for_game(anno['points'],h,w)
                    density=boxes_to_density(boxes,h,w)
                    path=maps/(Path(name).stem+'.npz');np.savez_compressed(path,density=density)
                    if not repeat:
                        raw=boxes_to_density(detail['second_boxes_xyxy'],h,w) if detail['fallback'] else density
                        raw_path=maps/(Path(name).stem+'-no-fallback.npz') if detail['fallback'] else path
                        if detail['fallback']:np.savez_compressed(raw_path,density=raw)
                        detail['no_fallback_density_file_sha256']=sha(raw_path)
                        detail['no_fallback_density_array_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
                    for c in detail['pseudo_exemplars']:
                        x0,y0,x1,y1=c['box_xyxy']
                        c['annotated_points_inside']=int(((points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)).sum())
                    _,truth8=point_cells(points,h,w)
                    row={'image':name,'truth':len(points),'prediction':len(boxes),'width':w,'height':h,
                         'density_mass':float(density.sum()),'game1':game(density,points,1),'game2':game(density,points,2),
                         'inference_seconds_per_image':inference,'boxes_xyxy':boxes.tolist(),'objectness':scores.tolist(),
                         'exemplars_xyxy':exemplars.tolist(),'mass8':cell_mass(density).tolist(),'truth8':truth8.tolist(),
                         'localization':localization(boxes,points,exemplars.numpy()),
                         'density_file_sha256':sha(path),'density_array_sha256':hashlib.sha256(density.tobytes()).hexdigest(),
                         'detail':detail}
                    if abs(density.sum()-len(boxes))>1e-6:raise ValueError('M2 density mass differs')
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                done+=1
                write_json(out/'progress.json',{'done':done,'total':len(names),'elapsed_seconds':time.monotonic()-started,
                           'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'updated_utc':utc()})
                if done%25==0 or done==len(names):print(json.dumps({'done':done,'total':len(names),'variant':variant}),flush=True)
        if state_sha(model)!=frozen:raise ValueError('Frozen M2 model changed')
        summary={'status':'passed' if phase=='gate' else 'complete','n':len(names),'phase':phase,'variant':variant,
                 'frozen_sha256':frozen,'elapsed_seconds':time.monotonic()-started,'finished_utc':utc(),
                 'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                 'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        if phase=='gate':summary['rows']=gate
        write_json(out/('gate.json' if phase=='gate' else 'summary.json'),summary)
    except BaseException as exc:
        write_json(out/'error.json',{'status':'failed','error_type':type(exc).__name__,'utc':utc(),
                   'elapsed_seconds':time.monotonic()-started,'automatic_retry':False})
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['gate','smoke','full'],required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--repeat',action='store_true');a=p.parse_args();main(a.root.resolve(),a.phase,a.candidate,a.repeat)
