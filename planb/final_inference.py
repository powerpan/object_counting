"""Frozen final route. Public inference interface accepts pixels and prompts only."""
import time
import numpy as np
import torch
from planb.round3_m3 import M3Engine, require_equal
from planb.round3_tta import TTAEngine
from planb.geco_adapter import postprocess
from planb.density import boxes_to_density


def trigger(median_side, count):
    return bool(median_side <= 32.0 or count >= 200)


class FinalEngine:
    def __init__(self, model, config):
        self.model=model;self.m3=M3Engine(model,config);self.tta=TTAEngine(model,config)

    @torch.inference_mode()
    def run_pair(self,image,exemplars,gate=False):
        if image.ndim!=3 or image.shape[0]!=3 or exemplars.shape!=(3,4):
            raise ValueError('Expected RGB and exactly three exemplar boxes')
        h,w=image.shape[-2:]
        if not torch.isfinite(image).all() or not torch.isfinite(exemplars).all():
            raise ValueError('Nonfinite input')
        e=exemplars.numpy()
        if not ((e[:,2:]>e[:,:2]).all() and (e>=0).all() and (e[:,[0,2]]<=w).all() and (e[:,[1,3]]<=h).all()):
            raise ValueError('Invalid exemplar extent')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        x,p,scale=self.m3.prepare(image,exemplars)
        native=self.model(x,p)[0][0]
        if not all(torch.isfinite(native[k]).all() for k in ('pred_boxes','box_v','scores')):
            raise ValueError('Nonfinite native output')
        d,v=postprocess(native,int(w*scale),int(h*scale))
        boxes=d.cpu().numpy()*(1024/scale);scores=v.cpu().numpy()
        median=float(torch.sqrt((p[0,:,2:]-p[0,:,:2]).prod(1)).median())
        density=boxes_to_density(boxes,h,w)
        torch.cuda.synchronize();base_seconds=time.monotonic()-tick
        b0={'prediction':float(len(boxes)),'boxes_xyxy':boxes.tolist(),'objectness':scores.tolist(),
            'inference_seconds':base_seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        if gate:
            raw,f,a=self.m3.adapter.forward(x,p);require_equal(native,raw)
            del raw,f,a
        del native,d,v,x,p
        routed=trigger(median,len(boxes))
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.monotonic()
        if routed:
            fb,fs,first,detail=self.m3.run(image,exemplars,'m3-v2-g3')
            if not detail['triggered'] or not detail['tiling_applied']: raise ValueError('Final M3 routing differs')
            final_density=boxes_to_density(fb,h,w);count=float(len(fb));route='M3-v2-g3'
        else:
            final_density,count,fb,fs,detail=self.tta.run(image,exemplars,'T2')
            first=fb;route='T2'
        torch.cuda.synchronize();final_seconds=time.monotonic()-tick
        if not np.array_equal(first,boxes): raise ValueError('Final original view differs from native B0')
        if abs(float(final_density.sum())-count)>1e-6: raise ValueError('Final map count mismatch')
        self.m3.check_memory()
        final={'prediction':count,'boxes_xyxy':fb.tolist(),'objectness':fs.tolist(),
               'inference_seconds':final_seconds,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
               'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        return b0,final,density,final_density,{'route':route,'triggered':routed,
            'median_exemplar_side_model_pixels':median,'base_scale':scale,
            'native_adapter_gate_passed':gate,'original_view_matches_b0':True,'method_detail':detail}
