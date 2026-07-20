#!/usr/bin/env bash
#SBATCH --job-name=glm52-nvfp4
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-glm52-nvfp4-%j.log
#
# GLM-5.2-NVFP4 on one RTX PRO 6000 node. Submit from the repo root:
#   sbatch scripts/slurm_glm52_nvfp4.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/GLM-5.2-NVFP4 scripts/slurm_glm52_nvfp4.sbatch
# Requires the RTX environment (.venv-cu130 + cuda13); see README "RTX 6000 environment".
# After it starts, the endpoint is http://<node>:8000/v1 on the allocated node (`squeue --me`).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$GLM52_NVFP4_MODEL}"
echo "serving GLM-5.2-NVFP4 on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL)"
exec bash scripts/vllm_glm52_nvfp4.sh
