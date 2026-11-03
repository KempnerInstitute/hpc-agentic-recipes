#!/usr/bin/env bash
#SBATCH --job-name=mistral-nvfp4
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8/serve.sbatch" >&2; exit 2; }
source "$R/env/env.sh"
[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
echo "serving mistral-large-4 on $(hostname -s):${API_PORT:-8000}"
exec bash "$R/serve.sh"
