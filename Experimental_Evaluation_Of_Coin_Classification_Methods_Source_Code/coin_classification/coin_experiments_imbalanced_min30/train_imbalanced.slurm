#!/bin/bash
#SBATCH --job-name=coin_imbalance
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:standard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/imbalance_%j.log
#SBATCH --error=logs/imbalance_%j.err
set -euo pipefail

# Submit from this campaign directory. Runtime files may live outside the source tree.
cd "${SLURM_SUBMIT_DIR:?Submit from the coin_experiments_imbalanced_min30 directory}"
test -f coin_experiments/run_experiments.py
PROJECT_ROOT="$(cd .. && pwd)"
BASE="${COIN_ENV_ROOT:-$HOME/coin_finki}"
BASE="$(cd "$BASE" && pwd)"
export SINGULARITYENV_TORCH_HOME="$BASE/pretrained_weights"
export SINGULARITYENV_OMP_NUM_THREADS=8 SINGULARITYENV_MKL_NUM_THREADS=8 SINGULARITYENV_OPENBLAS_NUM_THREADS=8 SINGULARITYENV_PYTHONNOUSERSITE=1
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
fi
CONTAINER="$BASE/coin_cuda126.sif"
PYTHON="$BASE/container_venv/bin/python"
echo "Job ID: $SLURM_JOB_ID | Node: $(hostname) | Start: $(date)"
nvidia-smi
singularity exec --nv --cleanenv "$CONTAINER" "$PYTHON" -u coin_experiments/run_experiments.py --output experiments_imbalanced --device cuda --preflight-only --check-device
singularity exec --nv --cleanenv "$CONTAINER" "$PYTHON" -u coin_experiments/run_experiments.py --output experiments_imbalanced --device cuda --experiment all --method all --confirm-audit
singularity exec --nv --cleanenv "$CONTAINER" "$PYTHON" -u coin_experiments/analyze_results.py --output experiments_imbalanced
singularity exec --nv --cleanenv "$CONTAINER" "$PYTHON" -u coin_experiments/compare_baseline.py --baseline "$PROJECT_ROOT/experiments_28" --output experiments_imbalanced
echo "All expanded experiments and anchor comparison completed: $(date)"
