#!/bin/bash
#SBATCH --job-name=lp_probe
#SBATCH --partition=allgroups
#SBATCH --gres=gpu:TESLA-T4-16GB:1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-71
#SBATCH --output=logs/lp_%A_%a.out

# Linear-probe (deep-forgetting) analysis over ALL runs of the 1/3/5-epoch sweep.
# One array task == one run dir. Each task writes <run_dir>/lp_curve.csv, which is
# what scripts/analysis/forgetting_by_epochs.py reads to build the deep table.
#
# IMPORTANT: run this only AFTER training has finished, so every run dir already
# contains its checkpoints/ folder.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ocl_survey

cd ~/tesi/ocl_survey

# Roots to scan (override by exporting ROOTS="dirA dirB ..." before sbatch).
ROOTS="${ROOTS:-results_ep1 results_ep3 results_ep5}"

# Subsample of training features used to FIT each probe. 200/class matches the
# methodology chapter and has negligible effect on the converged accuracy, while
# cutting feature-extraction time substantially. Set to 0 for the full train set.
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-200}"

# Use every STRIDE-th checkpoint. With eval_every=50 there are ~100/300/500
# checkpoints per run (1/3/5 epochs); STRIDE=1 processes them all (heavier but
# gives the full LP curve). Raise it (e.g. 5) if you only need the forgetting
# summary rather than a dense curve.
STRIDE="${STRIDE:-1}"

# Enumerate every run dir exactly once (each has a config.yaml next to logs.json).
mapfile -t RUNS < <(find $ROOTS -name config.yaml -printf '%h\n' 2>/dev/null | sort)

N=${#RUNS[@]}
echo "Found $N run dirs under: $ROOTS"
if [ "$N" -eq 0 ]; then
  echo "ERROR: no run dirs found. Are the ROOTS correct and training finished?"
  exit 1
fi
if [ "$SLURM_ARRAY_TASK_ID" -ge "$N" ]; then
  echo "Array index $SLURM_ARRAY_TASK_ID >= $N runs — nothing to do, exiting."
  exit 0
fi

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "=== [$SLURM_ARRAY_TASK_ID/$N] LP probe on: $RUN ==="
nvidia-smi --query-gpu=name --format=csv,noheader

if [ ! -d "$RUN/checkpoints" ]; then
  echo "WARNING: $RUN has no checkpoints/ — was it trained with experiment.save_checkpoints=true?"
fi

PYTHONPATH=~/tesi/ocl_survey python scripts/analysis/compute_lp.py \
  "$RUN" \
  --stride "$STRIDE" \
  --train-per-class "$TRAIN_PER_CLASS" \
  --device cuda

echo "=== done: $RUN/lp_curve.csv ==="
