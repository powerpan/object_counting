"""M4-v2 training and same-holdout detector reference; explicit finite stops."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from planb.round2.data import Inputs
from planb.round2.common import state_sha,sha,write_json,check_memory,seed_everything,verify_inputs,utc
from planb.m4v2 import ResidualHead,frozen_forward,epoch_indices,residual_loss,safety_stop
from m4_v2_common import settings,begin,deadline,fail,read,model_and_adapter,count_metrics
from m4_v2_preflight import batch,train_step
from round2_m4 import learning_rate,save_checkpoint

STOP=False
def interrupt(signum,frame):
    global STOP
    STOP=True


def holdout_reference(root,base,old,c,splits,out,start,limit):
    verify_inputs(root,base);model,adapter=model_and_adapter(root);frozen=state_sha(model)
    inputs=Inputs(root,splits['train_holdout']);rows=[]
    with torch.no_grad(),(out/'predictions.jsonl').open('x') as stream:
        for name in splits['train_holdout']:
            deadline(start,limit)
            if STOP:raise InterruptedError('Reference interrupted')
            sample=inputs.prepare(name,with_target=False);features,output,boxes,scores=frozen_forward(adapter,sample)
            row={'image':name,'truth':len(inputs.annotations[name]['points']),'prediction':len(boxes),'boxes_xyxy':boxes.tolist()}
            rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush();del features,output;check_memory()
    assert state_sha(model)==frozen;deadline(start,limit)
    write_json(out/'summary.json',{'status':'complete',**count_metrics(rows),'frozen_state_sha256':frozen,
        'source_predictions_sha256':sha(out/'predictions.jsonl'),'elapsed_seconds':time.monotonic()-start})


def train(root,base,old,c,splits,out,start,limit,seed):
    if seed!=42:
        approval=read(root/'results/m4-v2-v1/val-seed42/audit.json')
        if approval['status']!='passed' or not approval['candidate_gate']:raise ValueError('Extra seed not eligible')
    assert read(root/'results/m4-v2-v1/capacity/summary.json')['selected_batch']==8
    reference_dir=root/'results/m4-v2-v1/baseline-holdout';reference=read(reference_dir/'summary.json')
    assert reference['status']=='complete' and reference['n']==396
    assert read(reference_dir/'audit.json')['status']=='passed'
    verify_inputs(root,base);seed_everything(seed);model,adapter=model_and_adapter(root);frozen=state_sha(model)
    assert frozen==reference['frozen_state_sha256']
    head=ResidualHead().cuda();optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    fit=splits['train_fit'];hold=splits['train_holdout'];inputs=Inputs(root,fit+hold)
    old_index={n:i for i,n in enumerate(old['train_fit']+old['train_holdout'])}
    work=root/'work/m4-v2-v1'/out.name
    if work.exists():raise FileExistsError('Existing weights must not be overwritten')
    work.mkdir(parents=True);torch.cuda.reset_peak_memory_stats()
    protocol=read(out/'protocol.json');protocol.update(seed=seed,fit_n=len(fit),holdout_n=len(hold),batch_size=8,
        baseline_predictions_sha256=sha(reference_dir/'predictions.jsonl'),baseline_rmse=reference['rmse'])
    write_json(out/'protocol.json',protocol)
    state={'epoch':0,'next_batch':0,'global_step':0,'history':[],'best_epoch':None,'best_holdout_rmse':float('inf'),
           'seed':seed,'frozen_state_sha256':frozen,'config_sha256':sha(root/'configs/m4_v2.json'),
           'epoch_seen':0,'epoch_loss_sum':0.}
    def snapshot(status):
        state['elapsed_seconds']=time.monotonic()-start
        save_checkpoint(work/'last.pth',head,optimizer,state)
        write_json(out/'status.json',{'status':status,**{k:state[k] for k in ('epoch','next_batch','global_step','best_epoch','elapsed_seconds')},
            'best_holdout_rmse':state['best_holdout_rmse'] if math.isfinite(state['best_holdout_rmse']) else None,
            'peak_allocated_mib':check_memory(),'updated_utc':utc(),'automatic_retry':False})
    def check_stop():
        if STOP:snapshot('interrupted');raise InterruptedError('Training cancelled')
        if time.monotonic()-start>=limit:snapshot('budget_exceeded');raise TimeoutError('Training time budget reached')
    try:
        for epoch in range(30):
            order=epoch_indices(fit,splits['counts'],seed,epoch);steps=math.ceil(len(order)/8)
            assert len(order)==3359 and steps==420
            head.train();total=0.;seen=0;state['epoch_loss_sum']=0.;state['epoch_seen']=0
            for batch_no,offset in enumerate(range(0,len(order),8)):
                check_stop();b=batch(adapter,inputs,fit,order[offset:offset+8],seed,epoch,old_index,True)
                rate=learning_rate(state['global_step'],steps,base)
                for g in optimizer.param_groups:g['lr']=rate
                loss,details=train_step(head,optimizer,b);n=len(order[offset:offset+8]);total+=loss*n;seen+=n
                state.update(global_step=state['global_step']+1,next_batch=batch_no+1,epoch_loss_sum=total,epoch_seen=seen);del b
                if state['global_step']%50==0:
                    write_json(out/'progress.json',{'epoch':epoch+1,'batch':batch_no+1,'batches':steps,'loss':loss,'loss_terms':details,
                        'lr':rate,'elapsed_seconds':time.monotonic()-start,'peak_allocated_mib':check_memory()})
                if state['global_step']%100==0:snapshot('running')
            head.eval();rows=[];hold_loss=0.
            prediction_file=out/('holdout-epoch%02d.jsonl'%(epoch+1))
            with torch.no_grad(),prediction_file.open('x') as stream:
                for offset in range(0,len(hold),8):
                    check_stop();indices=list(range(offset,min(offset+8,len(hold))))
                    b=batch(adapter,inputs,hold,indices,seed,epoch,old_index,False);features,target,mask,counts=b
                    raw=head(features,mask);loss,detail=residual_loss(raw,target,mask,counts)
                    hold_loss+=float(loss)*len(indices)
                    predicted=raw.clamp_min(0).double().flatten(1).sum(1).cpu().tolist()
                    raw_count=raw.double().flatten(1).sum(1).cpu().tolist()
                    negative=(-raw).clamp_min(0).double().flatten(1).sum(1).cpu().tolist()
                    for j,i in enumerate(indices):
                        row={'image':hold[i],'truth':splits['counts'][hold[i]],'prediction':predicted[j],
                             'raw_prediction':raw_count[j],'negative_mass':negative[j]}
                        rows.append(row);stream.write(json.dumps(row)+'\n')
                    del raw,b,features,target,mask,counts;check_memory()
            after=state_sha(model)
            if after!=frozen or any(p.grad is not None for p in model.parameters()):raise AssertionError('Frozen model changed')
            check_stop();metrics=count_metrics(rows)
            row={'epoch':epoch+1,'train_n':seen,'train_loss':total/seen,'holdout_n':len(rows),
                 'holdout_mae':metrics['mae'],'holdout_rmse':metrics['rmse'],'holdout_loss':hold_loss/len(hold),
                 'baseline_holdout_rmse':reference['rmse'],'holdout_negative_mass_mean':sum(r['negative_mass'] for r in rows)/len(rows),
                 'predictions_sha256':sha(prediction_file),'frozen_state_sha256':frozen,'elapsed_seconds':time.monotonic()-start}
            state['history'].append(row);state['epoch']=epoch+1;state['next_batch']=0
            if metrics['rmse']<state['best_holdout_rmse']:
                state.update(best_holdout_rmse=metrics['rmse'],best_epoch=epoch+1,elapsed_seconds=time.monotonic()-start)
                save_checkpoint(work/'best.pth',head,optimizer,state)
            write_json(out/'history.json',state['history']);snapshot('running');print(json.dumps(row),flush=True)
            if safety_stop(epoch+1,metrics['rmse'],reference['rmse']):
                snapshot('negative_safety_stop')
                write_json(out/'summary.json',{'status':'negative_safety_stop','complete_epochs':epoch+1,
                    'epoch10_rmse':metrics['rmse'],'baseline_rmse':reference['rmse'],'candidate_eligible':False,
                    'best_sha256':sha(work/'best.pth'),'last_sha256':sha(work/'last.pth'),'elapsed_seconds':time.monotonic()-start})
                return
        snapshot('complete')
        write_json(out/'checkpoint_manifest.json',{'complete_epochs':30,'best_epoch':state['best_epoch'],
            'best_holdout_rmse':state['best_holdout_rmse'],'best_sha256':sha(work/'best.pth'),'last_sha256':sha(work/'last.pth'),
            'frozen_model_sha256':frozen,'config_sha256':state['config_sha256']})
        write_json(out/'summary.json',{'status':'complete','epochs':30,'elapsed_seconds':time.monotonic()-start})
    except BaseException:
        # Periodic and explicit interruption snapshots remain; never retry here.
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--phase',choices=['baseline-holdout','train'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42);p.add_argument('--time-limit-seconds',type=float,required=True)
    a=p.parse_args();root=a.root.resolve();base,old,c,splits=settings(root)
    stage=a.phase if a.phase=='baseline-holdout' else 'train-seed'+str(a.seed)
    out,start=begin(root,stage,a.time_limit_seconds,5400 if a.phase=='baseline-holdout' else 25200)
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        if a.phase=='baseline-holdout':holdout_reference(root,base,old,c,splits,out,start,a.time_limit_seconds)
        else:train(root,base,old,c,splits,out,start,a.time_limit_seconds,a.seed)
    except BaseException as e:fail(out,start,e);raise


if __name__=='__main__':main()
