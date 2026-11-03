#!/usr/bin/env bash
#SBATCH --job-name=qwen38-24t-fp8
#SBATCH --partition=kempner_h200
#SBATCH --nodes=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1440G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
#   sbatch --account=<acct> recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/serve.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
ipof () { srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
echo "head=$HEAD ($HEAD_IP)  workers=${nodes[*]:1}"
echo "serving qwen3.8-2.4t: http://$HEAD:${API_PORT:-8000}/v1"

srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/ray_head.sh" "$HEAD_IP" "$RAY_PORT" &
sleep 12
for w in "${nodes[@]:1}"; do
  srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$w" --overlap \
    bash "$R/ray_worker.sh" "$HEAD_IP" "$RAY_PORT" "$(ipof "$w")" &
  sleep 4
done
sleep 20

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/serve.sh" "$HEAD_IP" "$RAY_PORT"
