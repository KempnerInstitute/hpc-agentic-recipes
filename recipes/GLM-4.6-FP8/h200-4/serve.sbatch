#!/usr/bin/env bash
#SBATCH --job-name=glm46
#SBATCH --partition=kempner_h200
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/GLM-4.6-FP8/h200-4/serve.sbatch
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/GLM-4.6-FP8/h200-4
echo "serving glm-4.6 on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
