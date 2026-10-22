"""Loopback-only Gradio demo of the frozen route; one pair per button press."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_common import sources,read,write,sha,utc
from round3_common import runtime
from run_m2_m3 import load_model
from planb.final_inference import FinalEngine
from planb.demo_ui import selection,boxes_image,density_image
from planb.round3_accounting import StageLedger


def main(root,port):
    import gradio as gr
    cfg,method,receipt=sources(root);locks=[];start=time.monotonic()
    for name in ('.round2-m4.lock','.m2-m3.lock','.round3.lock'):
        f=(root/'results'/name).open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    for ds in ['fsc147-test','carpk']:
        if read(root/'results/final-v1'/ds/'audit.json')['status']!='passed':raise ValueError('Complete evaluations first')
    out=root/'results/final-v1/demo';out.mkdir(exist_ok=False)
    maps=root/'work/final-v1/demo';maps.mkdir(parents=True,exist_ok=False)
    ledger=StageLedger(root/'results/final-v1/budget_ledger_v1.json')
    limit=ledger.begin('gradio-demo','demo',3600,utc());calls=[];failed=False
    app=None
    def stop(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(int(limit)-5)
    try:
        runtime(root);model=load_model(root);engine=FinalEngine(model,method)
        write(out/'status.json',{'state':'running','started_utc':utc(),'pid':os.getpid(),'limit_seconds':limit,'gpu':torch.cuda.get_device_name()})
        def load_example(key):
            im=np.array(Image.open(root/'data/fsc147/images_384_VarV2'/cfg['demo'][key]).convert('RGB'))
            return im,im,[],None,'請點選三個目標的左上角、右下角，共六次。',None,None,''
        def upload(im):
            if im is None:return None,[],None,'請上傳圖片。',None,None,''
            return im,[],None,'請點選三個目標的左上角、右下角，共六次。',None,None,''
        def click(im,points,evt:gr.SelectData):
            if im is None:raise gr.Error('請先上傳圖片。')
            if len(points)>=6:raise gr.Error('已選三框；如需重選，請清除示例。')
            new=points+[list(evt.index)];boxes=selection(new,im.shape[1],im.shape[0])
            canvas=boxes_image(im,[],prompts=boxes)
            if len(new)%2:
                pic=Image.fromarray(canvas);from PIL import ImageDraw
                x,y=new[-1];ImageDraw.Draw(pic).ellipse((x-4,y-4,x+4,y+4),fill=(0,220,255));canvas=np.array(pic)
            return canvas,new,f'已選 {len(boxes)}／3 個示例框（{len(new)}／6 次點選）。'
        def clear(im):return im,[],None,'已清除，請重新點選六個角點。',None,None,''
        def display(cache,mode):
            if cache is None:return None,None,''
            key='b0' if mode=='B0' else 'final';r=cache[key]
            return density_image(cache['image'],cache[key+'_density']),boxes_image(cache['image'],r['boxes_xyxy'],prompts=cache['prompts']),(
                f"### 估計數量：{r['prediction']:.2f}\n\n推論 {r['inference_seconds']:.3f} 秒 · GPU {torch.cuda.get_device_name()}\n\n"
                f"方法：{'原圖基準' if key=='b0' else cache['detail']['route']}。密度圖亮度只用於顯示，計數取密度總和。")
        def predict(im,points,mode):
            nonlocal failed
            if im is None or len(points)!=6:raise gr.Error('請先完成三個示例框。')
            prompts=selection(points,im.shape[1],im.shape[0])
            try:
                b,f,bd,fd,detail=engine.run_pair(T.ToTensor()(Image.fromarray(im)),torch.tensor(prompts,dtype=torch.float32))
                record={'call':len(calls)+1,'image_rgb_sha256':hashlib.sha256(im.tobytes()).hexdigest(),
                        'exemplars_xyxy':prompts,'b0':b,'final':f,'detail':detail,'finished_utc':utc(),'truth_supplied':False}
                calls.append(record);write(out/'calls.json',calls)
                np.savez_compressed(maps/f'call-{len(calls)}.npz',image=im,b0=bd,final=fd)
                cache={'image':im,'prompts':prompts,'b0':b,'final':f,'b0_density':bd,'final_density':fd,'detail':detail}
                return (cache,*display(cache,mode))
            except BaseException:
                failed=True;write(out/'failure.json',{'automatic_retry':False,'call':len(calls)+1,'time':utc()});os.kill(os.getpid(),signal.SIGTERM);raise
        with gr.Blocks(title='類別無關物件計數',analytics_enabled=False) as app:
            gr.Markdown('# 類別無關物件計數\n上傳一張 RGB 圖片，框選三個目標示例，即可估計同類物件數量。')
            original=gr.State();points=gr.State([]);cache=gr.State()
            with gr.Row():
                ordinary=gr.Button('載入一般圖');dense=gr.Button('載入密集圖');reset=gr.Button('清除示例')
            with gr.Row():
                canvas=gr.Image(label='點選每個示例的兩個角點',type='numpy',sources=['upload'],format='png',height=430)
                with gr.Column():
                    message=gr.Markdown('請上傳圖片。');mode=gr.Radio(['B0','最終系統'],value='最終系統',label='顯示方法')
                    run=gr.Button('開始計數',variant='primary');metrics=gr.Markdown()
            with gr.Row():
                heat=gr.Image(label='密度疊圖',interactive=False,format='png',height=380)
                detections=gr.Image(label='檢測框',interactive=False,format='png',height=380)
            ordinary.click(lambda:load_example('ordinary_val'),outputs=[original,canvas,points,cache,message,heat,detections,metrics])
            dense.click(lambda:load_example('dense_val'),outputs=[original,canvas,points,cache,message,heat,detections,metrics])
            canvas.upload(upload,inputs=canvas,outputs=[original,points,cache,message,heat,detections,metrics])
            canvas.select(click,inputs=[original,points],outputs=[canvas,points,message])
            reset.click(clear,inputs=original,outputs=[canvas,points,cache,message,heat,detections,metrics])
            run.click(predict,inputs=[original,points,mode],outputs=[cache,heat,detections,metrics])
            mode.change(display,inputs=[cache,mode],outputs=[heat,detections,metrics])
        app.queue(default_concurrency_limit=1).launch(server_name='127.0.0.1',server_port=port,share=False,prevent_thread_lock=True,show_error=False)
        while True:time.sleep(1)
    except KeyboardInterrupt:pass
    except BaseException:
        failed=True;raise
    finally:
        signal.alarm(0)
        if app is not None:app.close()
        elapsed=time.monotonic()-start
        accepted=ledger.finish('gradio-demo',elapsed,bool(calls) and not failed,utc())
        write(out/'status.json',{'state':'complete' if accepted else 'failed','calls':len(calls),'elapsed_seconds':elapsed,
              'finished_utc':utc(),'gpu':torch.cuda.get_device_name(),'preset_examples_source':'val only',
              'input_content_hashes_saved':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=7860)
    a=p.parse_args();main(a.root.resolve(),a.port)
