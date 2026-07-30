#!/usr/bin/env bash
#SBATCH --job-name=qwen3-coder
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/serve.sbatch
#
# All eight GPUs are used as TP4 x PP2, not TP8, because this FP8 checkpoint cannot shard eight ways.
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8
echo "serving qwen3-coder-480b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1  (TP${TP:-4} x PP${PP:-2})"
exec bash "$R/serve.sh"
