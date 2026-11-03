#!/usr/bin/env bash
#SBATCH --job-name=mimo-v2.6-pro
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/MiMo-V2.6-Pro-RL/rtx-8"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/MiMo-V2.6-Pro-RL/rtx-8/serve.sbatch" >&2; exit 2; }
echo "serving mimo-v2.6-pro on $(hostname -s):${API_PORT:-8000}"
exec bash "$R/serve.sh"
