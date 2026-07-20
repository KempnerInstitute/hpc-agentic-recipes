#!/usr/bin/env bash
#SBATCH --job-name=glm46
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-glm46-%j.log
#
# GLM-4.6 FP8 on one H200 node. Submit from the repo root:
#   sbatch scripts/slurm_glm46.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/GLM-4.6-FP8 scripts/slurm_glm46.sbatch
# After it starts, the endpoint is http://<node>:8000/v1 on the allocated node (`squeue --me`).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$GLM46_MODEL}"
echo "serving GLM-4.6 on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL)"
exec bash scripts/vllm_glm46.sh
