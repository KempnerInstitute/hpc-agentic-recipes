#!/usr/bin/env bash
#SBATCH --job-name=gemma26-h200-1
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/gemma-4-26B-A4B-it/h200-1/serve.sbatch
#
# 16 CPUs is the per-GPU allowance on kempner_h200, and 180 GB is half of the 360 GB it allows per
# GPU, which is far more than a single-GPU server needs. The request therefore schedules easily and
# leaves the node's other three GPUs to other jobs.
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/gemma-4-26B-A4B-it/h200-1
echo "serving gemma-4-26b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
