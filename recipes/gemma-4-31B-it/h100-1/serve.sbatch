#!/usr/bin/env bash
#SBATCH --job-name=gemma31-h100-1
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/gemma-4-31B-it/h100-1/serve.sbatch
#
# 24 CPUs is the per-GPU allowance on kempner_h100, and 180 GB of host memory is far more than a single-GPU server needs. The request therefore schedules easily and leaves the node's other three GPUs to other jobs.
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/gemma-4-31B-it/h100-1
Q="${QUANT-fp8}"
echo "serving gemma-4-31b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1  (quant ${Q:-bf16})"
exec bash "$R/serve.sh"
