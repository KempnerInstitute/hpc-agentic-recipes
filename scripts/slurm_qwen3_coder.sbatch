#!/usr/bin/env bash
#SBATCH --job-name=qwen3-coder
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-qwen3-coder-%j.log
#
# Qwen3-Coder-480B-A35B-Instruct-FP8 on one full RTX PRO 6000 node (8 GPUs, TP4 x PP2).
# Submit from the repo root:
#   sbatch scripts/slurm_qwen3_coder.sbatch
# TP4 is mandatory for this FP8 checkpoint (see the note in scripts/vllm_qwen3.sh), so the 8 GPUs are
# used as TP4 x PP2 rather than TP8. Requires the RTX environment (.venv-cu130 + cuda13).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$CODER_MODEL}"
export SERVED_NAME="${SERVED_NAME:-qwen3-coder-480b}"
export ENV_LIB="${ENV_LIB:-lib_env_cu130.sh}" TP="${TP:-4}" PP="${PP:-2}"
export TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}" REASONING_PARSER=""
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
echo "serving $SERVED_NAME on $(hostname): http://$(hostname):${API_PORT}/v1  (TP$TP x PP$PP)"
exec bash scripts/vllm_qwen3.sh
