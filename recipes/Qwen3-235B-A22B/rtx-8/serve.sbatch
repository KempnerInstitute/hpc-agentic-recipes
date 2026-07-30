#!/usr/bin/env bash
#SBATCH --job-name=qwen3-235b
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Qwen3-235B-A22B/rtx-8/serve.sbatch
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Qwen3-235B-A22B/rtx-8
echo "serving qwen3-235b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
