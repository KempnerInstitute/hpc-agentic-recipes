#!/usr/bin/env bash
#SBATCH --job-name=muse-nvfp4
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang/serve.sbatch" >&2; exit 2; }
source "$R/env/env.sh"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }

NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"
echo "serving muse-spark on $NODE:${API_PORT:-8000}"
echo "log: $MUSE_LOG_DIR/muse-nvfp4-sglang.log"

exec bash "$R/serve.sh"
