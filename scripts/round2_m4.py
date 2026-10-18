"""Prerecorded M4 gate, bounded capacity measurement, and frozen-head training.

No test split, automatic retry, unregistered width, or upstream mutation.
"""
import argparse
import json
import math
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planb.round2.adapter import Round2Adapter
from planb.round2.common import setup, execution_record, write_json, state_sha, check_memory, seed_everything, sha, utc, verify_inputs
from planb.round2.data import Inputs
from planb.round2.head import DensityHead, density_loss
from planb.geco_adapter import postprocess
from planb.density import points_for_game
from run_m2_m3 import load_model


INTERRUPTED = False


def interrupted(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True


def make_batch(adapter, inputs, names, indices, *, seed, epoch, augment):
    prepared = [inputs.prepare(names[i], seed=seed, epoch=epoch, index=int(i), augment=augment) for i in indices]
    with torch.no_grad():
        features = torch.empty(len(prepared), 257, 512, 512, device='cuda')
        for j, sample in enumerate(prepared):
            single = adapter.density_features(sample['image'].cuda(), sample['boxes'].cuda())
            if not torch.isfinite(single).all():
                raise FloatingPointError('Nonfinite frozen features')
            features[j:j+1].copy_(single)
            del single
    target = torch.stack([s['target'] for s in prepared]).cuda()
    mask = torch.stack([s['mask'] for s in prepared]).cuda()
    counts = torch.tensor([s['count'] for s in prepared], dtype=torch.float32, device='cuda')
    return features, target, mask, counts


def step(head, optimizer, batch, training=True):
    x, target, mask, counts = batch
    if training:
        optimizer.zero_grad(set_to_none=True)
    prediction = head(x, mask)
    loss, details = density_loss(prediction, target, mask, counts)
    if training:
        loss.backward()
        for parameter in head.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Missing or nonfinite density-head gradient')
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    check_memory()
    return float(loss.detach()), details


def build_head(cfg, width):
    c = cfg['m4']
    return DensityHead(c['widths'][width], c['groupnorm_groups'], c['final_scaled_density_init']).cuda()


def optimizer_for(head, cfg):
    c = cfg['m4']
    return torch.optim.AdamW(head.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])


def gate(root, cfg, splits, out):
    record = execution_record(root, 'R2-0-gate')
    write_json(out/'protocol.json', record)
    write_json(out/'inputs.json', verify_inputs(root, cfg))
    train_names = splits['train_fit'] + splits['train_holdout']
    inputs = Inputs(root, train_names + splits['val_A'] + splits['val_B'])
    # Audit all training point coordinates without running a model on them.
    maximum = 0
    for name in train_names:
        with Image.open(root/'data/fsc147/images_384_VarV2'/name) as im:
            w, h = im.size
        points = points_for_game(inputs.annotations[name]['points'], h, w)
        maximum = max(maximum, len(points))
    write_json(out/'training_annotation_audit.json', {'status':'passed','n':len(train_names),'maximum_count':maximum})
    model = load_model(root); adapter = Round2Adapter(model)
    before = state_sha(model)
    bdir = root/cfg['baseline_result']
    baseline = {r['image']:r for r in map(json.loads, (bdir/'predictions.jsonl').read_text().splitlines())}
    old_gate = json.loads((root/'results/m2-m3-v1/gate/gate.json').read_text())
    names = [r['image'] for r in old_gate['rows']]
    from utils.data import resize_and_pad
    rows = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for name in names:
            sample = inputs.prepare(name,with_target=False)
            with Image.open(root/'data/fsc147/images_384_VarV2'/name) as im:
                from torchvision.transforms.functional import to_tensor
                image = to_tensor(im.convert('RGB'))
            boxes = torch.tensor(inputs.annotations[name]['box_examples_coordinates'], dtype=torch.float32)[:3,[0,2],:].reshape(-1,4)
            x, prompts, scale = resize_and_pad(image, boxes)
            assert torch.equal(inputs.normalize(x).unsqueeze(0), sample['image'])
            assert torch.equal(prompts.unsqueeze(0), sample['boxes']) and scale == sample['scale']
            x, prompts = sample['image'].cuda(), sample['boxes'].cuda()
            original = model(x, prompts)
            result, features, appearance = adapter.forward(x, prompts)
            for key in ('pred_boxes','box_v','scores'):
                assert torch.equal(original[0][0][key], result[key]), key
            final, _ = postprocess(result, int(sample['width']*scale), int(sample['height']*scale))
            final = final.cpu().numpy()*(1024/scale)
            assert np.array_equal(final, np.asarray(baseline[name]['boxes_xyxy'],dtype=np.float32).reshape(-1,4))
            rows.append({'image':name,'bitwise_equal':True,'count':len(final)})
            del original, result, features, appearance, x, prompts, final
            check_memory()
            print(json.dumps({'gate_done':len(rows),'total':len(names)}),flush=True)
    assert state_sha(model) == before
    write_json(out/'gate.json', {'status':'passed','n':len(rows),'rows':rows,'dense_shape':[1,256,512,512],
               'input_shape':[1,257,512,512], 'frozen_state_sha256':before,
               'peak_allocated_mib':check_memory(), 'finished_utc':utc()})


def capacity(root, cfg, splits, out, width):
    write_json(out/'protocol.json', execution_record(root, 'R2-0-capacity-'+width))
    assert json.loads((root/'results/round2-v1/gate/gate.json').read_text())['status']=='passed'
    model = load_model(root); adapter = Round2Adapter(model); before = state_sha(model)
    names = splits['train_fit']; inputs = Inputs(root, names[:8])
    records = []; chosen = 4
    for batch_size in cfg['m4']['batch_candidates']:
        seed_everything(cfg['seed']); torch.cuda.empty_cache()
        head = build_head(cfg, width); optimizer = optimizer_for(head,cfg)
        parameters = sum(p.numel() for p in head.parameters())
        assert parameters == {'S':2142401,'L':7381377}[width]
        torch.cuda.reset_peak_memory_stats()
        fixed = torch.cuda.memory_allocated()/2**20; started = time.perf_counter(); losses=[]
        for index in range(cfg['m4']['capacity_probe_steps']):
            batch = make_batch(adapter, inputs, names, list(range(batch_size)),seed=42,epoch=0,augment=False)
            loss, details = step(head,optimizer,batch); losses.append(loss)
            del batch
        torch.cuda.synchronize()
        peak = check_memory(); seconds = time.perf_counter()-started
        if any(p.grad is not None for p in model.parameters()):
            raise AssertionError('Frozen model received gradients')
        projected = fixed + 2*(peak-fixed)*cfg['m4']['capacity_projection_margin']
        records.append({'batch_size':batch_size,'finite_loss_and_gradients':True,'losses':losses,
                        'parameters':parameters,'fixed_mib':fixed,'peak_allocated_mib':peak,
                        'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                        'seconds':seconds,'seconds_per_image':seconds/(batch_size*len(losses)),
                        'projected_batch8_mib':projected if batch_size==4 else None,
                        'image_ids':names[:batch_size]})
        chosen = batch_size
        write_json(out/'capacity.json', {'status':'running','width':width,'records':records})
        del optimizer, head
        if batch_size==4 and projected > cfg['m4']['batch8_projected_peak_limit_mib']:
            break
    assert state_sha(model)==before
    write_json(out/'capacity.json', {'status':'passed','width':width,'selected_batch':chosen,'records':records,
               'frozen_state_unchanged':True,'frozen_state_sha256':before,'finished_utc':utc()})


def learning_rate(step_index, steps_per_epoch, cfg):
    c=cfg['m4']; warm=steps_per_epoch*c['warmup_epochs']; total=steps_per_epoch*c['epochs']
    if step_index < warm:
        return c['lr']*(step_index+1)/warm
    progress=(step_index-warm)/max(total-warm-1,1)
    return c['min_lr']+.5*(c['lr']-c['min_lr'])*(1+math.cos(math.pi*progress))


def save_checkpoint(path, head, optimizer, state):
    rng=np.random.get_state()
    value={**state,'head':head.state_dict(),'optimizer':optimizer.state_dict(),
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
           'python_rng':random.getstate(),'numpy_rng':{'name':rng[0],'keys':rng[1].tolist(),
              'position':rng[2],'has_gauss':rng[3],'cached_gaussian':rng[4]}}
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


def train(root,cfg,splits,out,width,seed,resume=False):
    started=time.perf_counter(); previous_seconds=0.
    work=root/'work/round2-v1'/out.name;work.mkdir(parents=True,exist_ok=True)
    capacity_record=json.loads((root/'results/round2-v1'/('capacity-'+width)/'capacity.json').read_text())
    assert capacity_record['status']=='passed'
    batch_size=capacity_record['selected_batch']
    if not resume:
        write_json(out/'protocol.json',{**execution_record(root,'R2-1-'+out.name),'width':width,'seed':seed,
                   'batch_size':batch_size,'accumulation_steps':1,'train_n':len(splits['train_fit']),
                   'holdout_n':len(splits['train_holdout']),'effective_time_limit_seconds':cfg['stop']['m4_single_run_seconds'],
                   'capacity_sha256':sha(root/'results/round2-v1'/('capacity-'+width)/'capacity.json')})
    verify_inputs(root,cfg)
    seed_everything(seed)
    model=load_model(root);adapter=Round2Adapter(model);frozen=state_sha(model)
    head=build_head(cfg,width);optimizer=optimizer_for(head,cfg)
    train_names=splits['train_fit'];holdout=splits['train_holdout']
    inputs=Inputs(root,train_names+holdout)
    steps_per_epoch=math.ceil(len(train_names)/batch_size)
    state={'epoch':0,'next_batch':0,'global_step':0,'best_holdout_mae':float('inf'),'best_epoch':None,
           'epoch_loss_sum':0.,'epoch_seen':0,'history':[],'seed':seed,'width':width,
           'frozen_state_sha256':frozen,'config_sha256':sha(root/'configs/round2_v1.json')}
    if resume:
        previous=json.loads((out/'status.json').read_text())
        if previous['status']!='interrupted':
            raise ValueError('Only explicitly interrupted runs can resume')
        original_protocol=json.loads((out/'protocol.json').read_text())
        for file,value in original_protocol['files_sha256'].items():
            if sha(root/file)!=value:raise ValueError('Resume code differs from original run')
        saved=torch.load(work/'last.pth',map_location='cuda',weights_only=True)
        assert saved['frozen_state_sha256']==frozen and saved['config_sha256']==state['config_sha256']
        head.load_state_dict(saved['head']);optimizer.load_state_dict(saved['optimizer'])
        state={k:saved[k] for k in state};previous_seconds=saved['elapsed_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        random.setstate(saved['python_rng'])
        nr=saved['numpy_rng'];np.random.set_state((nr['name'],np.array(nr['keys'],dtype=np.uint32),nr['position'],nr['has_gauss'],nr['cached_gaussian']))
    torch.cuda.reset_peak_memory_stats()
    def elapsed(): return previous_seconds+time.perf_counter()-started
    def snapshot(status):
        state['elapsed_seconds']=elapsed()
        save_checkpoint(work/'last.pth',head,optimizer,state)
        write_json(out/'status.json',{'status':status,'epoch':state['epoch'],'next_batch':state['next_batch'],
            'global_step':state['global_step'],'best_epoch':state['best_epoch'],'best_holdout_mae':state['best_holdout_mae'] if math.isfinite(state['best_holdout_mae']) else None,
            'elapsed_seconds':elapsed(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'updated_utc':utc()})
    try:
        for epoch in range(state['epoch'],cfg['m4']['epochs']):
            order=np.random.default_rng(seed+epoch).permutation(len(train_names))
            head.train()
            for batch_index in range(state['next_batch'],steps_per_epoch):
                if elapsed() >= cfg['stop']['m4_single_run_seconds']:
                    snapshot('budget_exceeded');raise TimeoutError('M4 single-run budget exceeded')
                if INTERRUPTED:
                    snapshot('interrupted');return
                indices=order[batch_index*batch_size:(batch_index+1)*batch_size].tolist()
                batch=make_batch(adapter,inputs,train_names,indices,seed=seed,epoch=epoch,augment=True)
                rate=learning_rate(state['global_step'],steps_per_epoch,cfg)
                for group in optimizer.param_groups:group['lr']=rate
                loss,details=step(head,optimizer,batch)
                state['epoch_loss_sum']+=loss*len(indices);state['epoch_seen']+=len(indices)
                state['global_step']+=1;state['next_batch']=batch_index+1
                del batch
                if state['global_step']%50==0:
                    progress={'epoch':epoch+1,'batch':batch_index+1,'batches':steps_per_epoch,'loss':loss,
                              'lr':rate,'elapsed_seconds':elapsed(),'peak_mib':check_memory()}
                    write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
                if state['global_step']%100==0:snapshot('running')
            head.eval();absolute_error=0.;holdout_loss=0.
            with torch.no_grad():
                for offset in range(0,len(holdout),batch_size):
                    if elapsed()>=cfg['stop']['m4_single_run_seconds']:
                        snapshot('budget_exceeded');raise TimeoutError('M4 single-run budget exceeded')
                    if INTERRUPTED:
                        snapshot('interrupted');return
                    indices=list(range(offset,min(offset+batch_size,len(holdout))))
                    batch=make_batch(adapter,inputs,holdout,indices,seed=seed,epoch=epoch,augment=False)
                    loss,detail=step(head,optimizer,batch,training=False)
                    absolute_error+=detail['count_mae']*len(indices);holdout_loss+=loss*len(indices)
                    del batch
            after=state_sha(model)
            if after!=frozen or any(p.grad is not None for p in model.parameters()):
                raise AssertionError('Frozen GeCo2 state changed')
            if elapsed()>=cfg['stop']['m4_single_run_seconds']:
                snapshot('budget_exceeded');raise TimeoutError('M4 single-run budget exceeded')
            row={'epoch':epoch+1,'train_n':state['epoch_seen'],'train_loss':state['epoch_loss_sum']/state['epoch_seen'],
                 'holdout_n':len(holdout),'holdout_mae':absolute_error/len(holdout),
                 'holdout_loss':holdout_loss/len(holdout),'frozen_state_sha256':after,
                 'elapsed_seconds':elapsed()}
            assert state['epoch_seen']==len(train_names)
            state['history'].append(row);state['epoch']=epoch+1;state['next_batch']=0
            state['epoch_seen']=0;state['epoch_loss_sum']=0.
            if row['holdout_mae']<state['best_holdout_mae']:
                state['best_holdout_mae']=row['holdout_mae'];state['best_epoch']=epoch+1
                state['elapsed_seconds']=elapsed();save_checkpoint(work/'best.pth',head,optimizer,state)
            snapshot('running');write_json(out/'history.json',state['history'])
            print(json.dumps(row),flush=True)
        snapshot('complete')
        write_json(out/'checkpoint_manifest.json',{'best_sha256':sha(work/'best.pth'),
            'last_sha256':sha(work/'last.pth'),'best_epoch':state['best_epoch'],
            'best_holdout_mae':state['best_holdout_mae'],'complete_epochs':state['epoch'],
            'weights_published':False,'frozen_model_sha256':frozen})
    except BaseException as exc:
        if not isinstance(exc,TimeoutError):
            write_json(out/'failure.json',{'status':'failed','error_type':type(exc).__name__,
                'epoch':state['epoch'],'next_batch':state['next_batch'],'elapsed_seconds':elapsed(),
                'updated_utc':utc(),'automatic_retry':False})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['gate','capacity','train'],required=True)
    p.add_argument('--width',choices=['S','L'],default='S');p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--time-limit-seconds',type=float,default=None)
    args=p.parse_args();root=args.root.resolve();out=args.output
    if args.resume and args.phase!='train':p.error('Only training has explicit resume')
    if out.exists() and any(out.iterdir()) and not args.resume:raise FileExistsError('Existing evidence must not be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    cfg,splits=setup(root)
    if args.time_limit_seconds is not None:
        if not 0 < args.time_limit_seconds <= cfg['stop']['m4_single_run_seconds']:
            p.error('An effective time limit may only tighten the registered cap')
        cfg['stop']['m4_single_run_seconds']=args.time_limit_seconds
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        if args.phase=='gate':gate(root,cfg,splits,out)
        elif args.phase=='capacity':capacity(root,cfg,splits,out,args.width)
        else:train(root,cfg,splits,out,args.width,args.seed,args.resume)
    except BaseException as exc:
        write_json(out/'error.json',{'status':'failed','error_type':type(exc).__name__,'phase':args.phase,'updated_utc':utc(),'automatic_retry':False})
        raise


if __name__=='__main__':main()
