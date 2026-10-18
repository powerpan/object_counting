"""Compare the compiled deformable-attention op with its PyTorch reference."""
import argparse
import json
import sys
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.upstream.resolve()))
from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch

torch.manual_seed(42)
shapes = torch.tensor([[6, 4], [3, 2]], dtype=torch.long, device='cuda')
starts = torch.tensor([0, 24], dtype=torch.long, device='cuda')
v = torch.rand(1, 30, 2, 4, device='cuda') * .01
loc = torch.rand(1, 2, 2, 2, 2, 2, device='cuda')
attn = torch.rand(1, 2, 2, 2, 2, device='cuda') + 1e-5
attn = attn / attn.sum((-1, -2), keepdim=True)
reference = ms_deform_attn_core_pytorch(v, shapes, loc, attn)
compiled = MSDeformAttnFunction.apply(v, shapes, starts, loc, attn, 2)
torch.testing.assert_close(compiled, reference, rtol=1e-4, atol=1e-6)
dv, dl, da = [t.double().requires_grad_() for t in (v, loc, attn)]
passed = torch.autograd.gradcheck(MSDeformAttnFunction.apply,
    (dv, shapes, starts, dl, da, 2), eps=1e-6, atol=1e-5, rtol=1e-3)
assert passed
report = {'status':'passed', 'test':'compiled_deformable_attention',
          'forward_max_abs_error': float((compiled-reference).abs().max()),
          'double_precision_gradcheck':bool(passed), 'torch':torch.__version__,
          'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0)}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
