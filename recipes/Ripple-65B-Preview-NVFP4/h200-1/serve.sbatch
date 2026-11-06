#!/usr/bin/env bash
#SBATCH --job-name=ripple-nvfp4-1gpu
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/Ripple-65B-Preview-NVFP4/h200-1"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/Ripple-65B-Preview-NVFP4/h200-1/serve.sbatch" >&2; exit 2; }
source "$R/env/env.sh"
[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
echo "serving ripple on $(hostname -s):${API_PORT:-8000}"
exec bash "$R/serve.sh"
