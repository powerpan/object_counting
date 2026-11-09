#!/usr/bin/env bash
# Build the isolated inference environment on a scheduled CPU node.
set -euo pipefail
: "${PLANB_ROOT:?Set PLANB_ROOT to the new, isolated project directory}"
: "${CUDA_HOME:?Load the CUDA 11.8 module first}"
test -n "${SLURM_JOB_ID:-}"
test "${SLURM_JOB_PARTITION:-}" = amd
test -x "$CUDA_HOME/bin/nvcc"
cd "$PLANB_ROOT"
test ! -e .venv
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export PIP_CACHE_DIR="$PLANB_ROOT/cache/pip" PIP_RETRIES=1 PIP_DEFAULT_TIMEOUT=30
export TORCH_HOME="$PLANB_ROOT/cache/torch"
export PIP_INDEX_URL=https://pypi.org/simple
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p logs provenance results external
python scripts/hpc4_port.py --root "$PLANB_ROOT" --phase sources
python -m venv .venv
source .venv/bin/activate
python -m pip install 'pip<26' 'setuptools<81' wheel ninja
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 hydra-core==1.3.2 scikit-image pycocotools einops scipy pillow tqdm gdown huggingface-hub==0.34.3 opencv-python-headless==4.11.0.86 matplotlib
test ! -e external/GECO2
git clone --no-checkout https://github.com/jerpelhan/GECO2.git external/GECO2
git -C external/GECO2 checkout --detach 67d27e8a93f14226f94d1f2c5c6dab5b5dd7162b
python -m pip install --no-build-isolation ./external/GECO2/Deformable-DETR/models/ops
cp -R external/GECO2/Deformable-DETR/models/ops external/GECO2/models/ops
python -m pip check
python -m pip freeze > logs/hpc4-environment.lock.txt
python scripts/hpc4_port.py --root "$PLANB_ROOT" --phase assets
printf 'HPC4_CPU_BOOTSTRAP_COMPLETE\n'
