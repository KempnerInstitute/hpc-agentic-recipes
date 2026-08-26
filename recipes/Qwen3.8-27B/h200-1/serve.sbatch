#!/usr/bin/env bash
#SBATCH --job-name=qwen38-27b-h200-1
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Qwen3.8-27B/h200-1/serve.sbatch
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Qwen3.8-27B/h200-1
echo "serving qwen3.8-27b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
