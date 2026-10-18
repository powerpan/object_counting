"""Baseline-preserving gate and bounded capacity probes; never reads L."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.m4v2 import ResidualHead,frozen_forward,residual_loss
from planb.round2.data import Inputs
from planb.round2.common import state_sha,seed_everything,write_json,check_memory,verify_inputs
from m4_v2_common import settings,begin,deadline,fail,model_and_adapter,read


def batch(adapter,inputs,names,indices,seed,epoch,original_indices,augment=True):
    samples=[inputs.prepare(names[i],seed=seed,epoch=epoch,index=original_indices[names[i]],augment=augment) for i in indices]
    features=torch.empty(len(indices),258,512,512,device='cuda')
    with torch.no_grad():
        for j,s in enumerate(samples):
            single,output,boxes,scores=frozen_forward(adapter,s)
            if not torch.isfinite(single).all():raise FloatingPointError('Nonfinite frozen features')
            features[j:j+1].copy_(single);del single,output
    return (features,torch.stack([s['target'] for s in samples]).cuda(),
            torch.stack([s['mask'] for s in samples]).cuda(),
            torch.tensor([s['count'] for s in samples],dtype=torch.float32,device='cuda'))


def train_step(head,optimizer,b):
    features,target,mask,counts=b;optimizer.zero_grad(set_to_none=True)
    raw=head(features,mask);loss,detail=residual_loss(raw,target,mask,counts);loss.backward()
    for p in head.parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():raise FloatingPointError('Invalid gradient')
    torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step();check_memory()
    return float(loss.detach()),detail


def gate(root,base,original,c,splits,out,start,limit):
    inputs_ok=verify_inputs(root,base);model,adapter=model_and_adapter(root);frozen=state_sha(model)
    names=[r['image'] for r in read(root/'results/round2-v1/gate/gate.json')['rows']]
    baseline={r['image']:r for r in map(json.loads,(root/base['baseline_result']/'predictions.jsonl').read_text().splitlines())}
    inputs=Inputs(root,names);head=ResidualHead().cuda().eval();rows=[];torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for name in names:
            deadline(start,limit);s=inputs.prepare(name,with_target=False)
            standard=model(s['image'].cuda(),s['boxes'].cuda())[0][0]
            features,output,boxes,scores=frozen_forward(adapter,s)
            for k in ('pred_boxes','box_v','scores'):
                if not torch.equal(standard[k],output[k]):raise AssertionError('Original GeCo2 output changed: '+k)
            if not np.array_equal(boxes,np.asarray(baseline[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4)):
                raise AssertionError('B0 final boxes changed')
            mask=s['mask'][None].cuda();raw=head(features,mask)
            if not torch.equal(raw,features[:,257:258]*mask):raise AssertionError('Initial residual is nonzero')
            if not torch.isfinite(raw).all() or (raw<0).any():raise FloatingPointError('Invalid initial density')
            predicted=float(raw.double().sum());difference=abs(predicted-len(boxes))
            if difference>c['start_count_tolerance']:raise AssertionError('Initial count differs from B0')
            rows.append({'image':name,'prediction':predicted,'B0_count':len(boxes),'absolute_difference':difference,'bitwise_equal':True})
            del standard,features,output,raw,mask;check_memory()
    if state_sha(model)!=frozen:raise AssertionError('Frozen parameters changed')
    deadline(start,limit)
    write_json(out/'summary.json',{'status':'passed','n':32,'rows':rows,'maximum_count_difference':max(r['absolute_difference'] for r in rows),
               'parameters':sum(p.numel() for p in head.parameters()),'frozen_state_sha256':frozen,'inputs':inputs_ok,
               'elapsed_seconds':time.monotonic()-start,'peak_allocated_mib':check_memory()})


def capacity(root,base,original,c,splits,out,start,limit):
    if read(root/'results/m4-v2-v1/initial-gate/summary.json')['status']!='passed':raise ValueError('Initial gate required')
    verify_inputs(root,base);model,adapter=model_and_adapter(root);frozen=state_sha(model)
    names=splits['train_fit'];inputs=Inputs(root,names[:8]);old_indices={n:i for i,n in enumerate(original['train_fit'])};records=[]
    for size in [4,8]:
        seed_everything(42);torch.cuda.empty_cache();head=ResidualHead().cuda()
        optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats();fixed=torch.cuda.memory_allocated()/2**20;beg=time.monotonic();losses=[]
        for _ in range(c['capacity_steps']):
            deadline(start,limit);b=batch(adapter,inputs,names,list(range(size)),42,0,old_indices,False)
            loss,_=train_step(head,optimizer,b);losses.append(loss);del b
        torch.cuda.synchronize();peak=check_memory();projected=fixed+2*(peak-fixed)*c['capacity_projection_margin']
        records.append({'batch_size':size,'steps':4,'losses':losses,'seconds':time.monotonic()-beg,
                        'peak_allocated_mib':peak,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                        'projected_batch8_mib':projected if size==4 else None})
        write_json(out/'progress.json',{'records':records})
        del optimizer,head
        if size==4 and projected>c['capacity_batch4_projection_limit_mib']:
            raise RuntimeError('Batch8 projection unsafe; registered batch is not changed automatically')
    if state_sha(model)!=frozen or any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen model changed')
    deadline(start,limit);write_json(out/'summary.json',{'status':'passed','selected_batch':8,'records':records,
        'frozen_state_sha256':frozen,'elapsed_seconds':time.monotonic()-start})


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['initial-gate','capacity'],required=True)
    p.add_argument('--time-limit-seconds',type=float,required=True);a=p.parse_args();root=a.root.resolve()
    base,original,c,splits=settings(root);out,start=begin(root,a.phase,a.time_limit_seconds,1800)
    try:(gate if a.phase=='initial-gate' else capacity)(root,base,original,c,splits,out,start,a.time_limit_seconds)
    except BaseException as e:fail(out,start,e);raise


if __name__=='__main__':main()
