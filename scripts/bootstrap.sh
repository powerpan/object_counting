#!/usr/bin/env bash
set -euo pipefail

ROOT="${PLANB_ROOT:-/root/autodl-tmp/planb}"
PYTHON="${PLANB_PYTHON:-/root/miniconda3/bin/python}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="8.9"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export TORCH_HOME="$ROOT/cache/torch"
export HF_HOME="$ROOT/cache/huggingface"
PIN=67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b

mkdir -p "$ROOT" "$ROOT/logs" "$ROOT/external" "$ROOT/weights" "$ROOT/data" "$ROOT/results"
test -x "$CUDA_HOME/bin/nvcc"
"$PYTHON" -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade 'pip<26' wheel 'setuptools<81' ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 hydra-core==1.3.2 scikit-image pycocotools einops scipy pillow tqdm gdown huggingface-hub==0.34.3 opencv-python-headless==4.11.0.86 matplotlib

if [[ ! -d "$ROOT/external/GECO2/.git" ]]; then
  git clone https://github.com/jerpelhan/GECO2.git "$ROOT/external/GECO2"
fi
git -C "$ROOT/external/GECO2" checkout --detach "$PIN"
test "$(git -C "$ROOT/external/GECO2" rev-parse HEAD)" = "$PIN"
cd "$ROOT/external/GECO2"
python -m pip install --no-build-isolation ./Deformable-DETR/models/ops
if [[ ! -e models/ops ]]; then
  cp -R Deformable-DETR/models/ops models/ops
fi
python -m pip check
python -m pip freeze > "$ROOT/logs/environment.lock.txt"
python - <<'PY'
import json, torch, torchvision
from models.ops.modules.ms_deform_attn import MSDeformAttn
from models.counter_infer import build_model
assert torch.cuda.is_available()
x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
(x @ x).square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print(json.dumps({'stage':'environment_import_and_cuda_smoke', 'torch':torch.__version__,
    'torchvision':torchvision.__version__, 'cuda':torch.version.cuda,
    'gpu':torch.cuda.get_device_name(0), 'total_memory_mib':torch.cuda.get_device_properties(0).total_memory/2**20,
    'status':'passed'}, indent=2))
PY
printf 'BOOTSTRAP_COMPLETE\n'
