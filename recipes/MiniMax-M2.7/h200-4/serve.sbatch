#!/usr/bin/env bash
#SBATCH --job-name=minimax-m2.7
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1440G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/MiniMax-M2.7/h200-4
echo "serving minimax-m2.7 on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
