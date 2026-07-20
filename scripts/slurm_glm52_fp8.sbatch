#!/usr/bin/env bash
#SBATCH --job-name=glm52-fp8
#SBATCH --partition=kempner_h200
#SBATCH --account=kempner_dev
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-glm52-fp8-%j.log
#
# GLM-5.2 FP8 across 2 H200 nodes (Ray + vLLM, TP4 x PP2). Submit from the repo root:
#   sbatch scripts/slurm_glm52_fp8.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/GLM-5.2-FP8 scripts/slurm_glm52_fp8.sbatch
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$GLM52_MODEL}"
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

srun --export=ALL,API_PORT="$API_PORT" --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$S/vllm_glm.sh" "$HEAD_IP" "$RAY_PORT"
