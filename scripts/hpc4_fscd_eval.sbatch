#!/bin/bash
#SBATCH --job-name=planb-fscd-eval
#SBATCH --partition=gpu-l20
#SBATCH --qos=l20_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
: "${PLANB_ROOT:?Set the deployment root}"
: "${PLANB_PHASE:?Set smoke or full}"
module load miniconda3/24.3.0-quc3pyu cuda/11.8.0-nctwsal
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$PLANB_ROOT"
"$PLANB_ROOT/.venv/bin/python" -m unittest discover -s tests -p 'test_fscd_lvis_*.py' -v
exec "$PLANB_ROOT/.venv/bin/python" "$PLANB_ROOT/scripts/run_fscd_lvis_external.py" --root "$PLANB_ROOT" --phase "$PLANB_PHASE"
