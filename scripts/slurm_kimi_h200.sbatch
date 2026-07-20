#!/usr/bin/env bash
#SBATCH --job-name=kimi-h200
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-kimi-h200-%j.log
#
# Kimi-K2.7-Code across 2 H200 nodes (Ray + vLLM, TP4 x PP2). Submit from the repo root:
#   sbatch scripts/slurm_kimi_h200.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/Kimi-K2.7-Code scripts/slurm_kimi_h200.sbatch
# EXTRA_ARGS defaults to --skip-mm-profiling (cross-node multimodal profiling otherwise deadlocks).
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$KIMI_MODEL}"
export ENV_LIB=lib_env.sh RAY=1 TP="${TP:-4}" PP="${PP:-2}" ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
export EXTRA_ARGS="${EXTRA_ARGS:---skip-mm-profiling --mm-processor-cache-gb 0}"
S="$(pwd)/scripts"
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
WORKER="${nodes[1]}"
ipof () { srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)  endpoint http://$HEAD:${API_PORT}/v1"

srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$S/ray_head.sh" "$HEAD_IP" "$RAY_PORT" &
sleep 12
srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$WORKER" --overlap bash "$S/ray_worker.sh" "$HEAD_IP" "$RAY_PORT" "$WORKER_IP" &
sleep 12

srun --export=ALL,API_PORT="$API_PORT" --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$S/vllm_kimi.sh" "$HEAD_IP" "$RAY_PORT"
