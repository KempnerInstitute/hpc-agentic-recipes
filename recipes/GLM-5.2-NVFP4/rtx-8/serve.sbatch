#!/usr/bin/env bash
#SBATCH --job-name=glm52-nvfp4
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/GLM-5.2-NVFP4/rtx-8
echo "serving glm-5.2 on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
