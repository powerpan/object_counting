#!/bin/bash
#SBATCH --job-name=planb-half-blend
#SBATCH --partition=amd
#SBATCH --qos=amd_qos
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --no-requeue
set -euo pipefail
: "${PLANB_ROOT:?Set the deployment root}"
module load miniconda3/24.3.0-quc3pyu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PLANB_ROOT"
"$PLANB_ROOT/.venv/bin/python" -m unittest discover -s tests -p 'test_half_blend.py' -v
exec "$PLANB_ROOT/.venv/bin/python" scripts/run_half_blend_replay.py --root "$PLANB_ROOT"
