#!/usr/bin/env bash
#SBATCH --job-name=glm52-fp8
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/GLM-5.2-FP8/h200-4-nodes2
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
WORKER="${nodes[1]}"
ipof () { srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"
echo "serving glm-5.2: http://$HEAD:${API_PORT:-8000}/v1"

srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/ray_head.sh" "$HEAD_IP" "$RAY_PORT" &
sleep 12
srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$WORKER" --overlap bash "$R/ray_worker.sh" "$HEAD_IP" "$RAY_PORT" "$WORKER_IP" &
sleep 12

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/serve.sh" "$HEAD_IP" "$RAY_PORT"
