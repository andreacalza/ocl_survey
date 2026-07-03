#!/bin/bash
# SLURM job array: naive baseline + ER with increasing replay buffer sizes,
# on the split_cifar100_til (multi-head, task-incremental) setup with anytime
# (every-50-steps) evaluation.
#
# EDIT BEFORE SUBMITTING:
#   - #SBATCH --partition   : your cluster's GPU partition name
#   - #SBATCH --gres         : your cluster's GPU request syntax (some clusters
#                              use --gpus=1 instead of --gres=gpu:1)
#   - CONDA_ENV_NAME / conda activation path if not using the default conda
#     install location
#   - DATASET_ROOT           : where CIFAR100 is (pre-)downloaded on this
#                              cluster; many clusters block internet access
#                              from compute nodes, so download it once on the
#                              login node first (it will auto-download to this
#                              path on first run otherwise).
#
# Submit with: sbatch scripts/hpc/slurm_buffer_sweep.sbatch
# Each array task = one (strategy, mem_size) configuration, run on its own GPU.

#SBATCH --job-name=ocl_buffer_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION

set -e

CONDA_ENV_NAME=ocl_survey
DATASET_ROOT=$HOME/.avalanche/data/cifar100

# array index -> (strategy, mem_size); index 0 is the naive (no-replay) baseline
STRATEGIES=(naive er er er er er)
MEM_SIZES=(2000 200 500 1000 2000 5000)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}
MEM_SIZE=${MEM_SIZES[$SLURM_ARRAY_TASK_ID]}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$REPO_ROOT/logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export PYTHONPATH="$REPO_ROOT"
export HYDRA_FULL_ERROR=1

cd "$REPO_ROOT/experiments"
echo "Task $SLURM_ARRAY_TASK_ID: strategy=$STRATEGY mem_size=$MEM_SIZE"

python main.py strategy=$STRATEGY experiment=split_cifar100_til evaluation=anytime \
  strategy.device=cuda strategy.mem_size=$MEM_SIZE \
  benchmark.dataset_root=$DATASET_ROOT \
  evaluation.num_gpus=1 evaluation.parallel_evaluation=false \
  experiment.debug=false
