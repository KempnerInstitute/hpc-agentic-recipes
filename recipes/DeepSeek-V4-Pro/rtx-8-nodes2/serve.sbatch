#!/usr/bin/env bash
#SBATCH --job-name=dsv4-rtx
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=2
#SBATCH --gpus=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# DeepSeek-V4-Pro across two RTX PRO 6000 nodes: Ray, TP8 inside each node, PP2 between them.
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/DeepSeek-V4-Pro/rtx-8-nodes2/serve.sbatch
#
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/DeepSeek-V4-Pro/rtx-8-nodes2
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
WORKER="${nodes[1]}"
ipof () {
  srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c \
    'ip -br -4 addr show ib0 2>/dev/null | awk "{print \$3}" | cut -d/ -f1 | grep . || hostname -i | awk "{print \$1}"'
}
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"
echo "serving deepseek-v4-pro: http://$HEAD:${API_PORT:-8000}/v1"

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash -c \
  "source $R/env/env.sh; ray stop >/dev/null 2>&1 || true; ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=8 --disable-usage-stats --block" &
sleep 15
srun --export=ALL --nodes=1 --ntasks=1 -w "$WORKER" --overlap bash -c \
  "source $R/env/env.sh; ray stop >/dev/null 2>&1 || true; ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP --num-gpus=8 --block" &
sleep 15

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/serve.sh" "$HEAD_IP" "$RAY_PORT"
