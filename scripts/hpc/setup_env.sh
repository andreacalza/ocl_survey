#!/bin/bash
# One-time environment setup on the HPC cluster (conda-based).
# Usage: bash scripts/hpc/setup_env.sh
#
# Verified locally against avalanche-lib==0.6.0 on Python 3.12 (see
# requirements.txt). If your cluster's CUDA version needs a specific torch
# build, install torch/torchvision separately BEFORE `pip install -r
# requirements.txt` using the index-url from https://pytorch.org/get-started
# (check `module avail cuda` or `nvidia-smi` for the CUDA version available).

set -e

ENV_NAME=ocl_survey
PYTHON_VERSION=3.12

conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Install a CUDA-enabled torch/torchvision build BEFORE this line if needed,
# e.g.: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pip install -r "$REPO_ROOT/requirements.txt"

echo "Environment '$ENV_NAME' ready. Activate with: conda activate $ENV_NAME"
echo "Remember to set PYTHONPATH=$REPO_ROOT before running experiments."
