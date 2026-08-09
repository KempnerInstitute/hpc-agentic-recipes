#!/usr/bin/env bash
#SBATCH --job-name=qwen3-235b
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Qwen3-235B-A22B/rtx-8/serve.sbatch
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Qwen3-235B-A22B/rtx-8
echo "serving qwen3-235b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
