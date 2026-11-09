#!/bin/bash
#SBATCH --job-name=planb-fscd-assets-v1
#SBATCH --partition=amd
#SBATCH --qos=amd_qos
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --no-requeue

set -euo pipefail
: "${PLANB_ROOT:?Set the existing deployment root at submission}"
# Pass --account at submission; keep the actual school account private.
module load miniconda3/24.3.0-quc3pyu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$PLANB_ROOT/.venv/bin/python" "$PLANB_ROOT/scripts/fscd_lvis_assets.py" --root "$PLANB_ROOT"
