#!/usr/bin/env bash
#SBATCH --job-name=muse-glimmer
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/Muse-Glimmer-30B/h200-1"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/Muse-Glimmer-30B/h200-1/serve.sbatch" >&2; exit 2; }
echo "serving muse-glimmer on $(hostname -s):${API_PORT:-8000}"
exec bash "$R/serve.sh"
