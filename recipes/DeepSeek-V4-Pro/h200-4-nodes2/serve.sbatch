#!/usr/bin/env bash
#SBATCH --job-name=dsv4-h200
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# DeepSeek-V4-Pro across two H200 nodes: Ray, TP4 inside each node, PP2 between them.
# Expected to be problematic: the routed experts are FP4 and Hopper has no FP4 hardware. Prefer
# recipes/DeepSeek-V4-Pro/rtx-8-nodes2 unless you are deliberately recording the H200 outcome.
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/DeepSeek-V4-Pro/h200-4-nodes2/serve.sbatch
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/DeepSeek-V4-Pro/h200-4-nodes2
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
WORKER="${nodes[1]}"
# Ray needs a routable address, not a hostname, for both ranks. These nodes carry InfiniBand, which is
# also what NCCL and Gloo are pinned to in env/env.sh.
ipof () { srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"
echo "serving deepseek-v4-pro: http://$HEAD:${API_PORT:-8000}/v1"

# Each H200 node contributes 4 GPUs to the Ray cluster. --block keeps the srun step alive so Slurm
# does not tear the node down while vLLM is using it.
srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash -c \
  "source $R/env/env.sh; ray stop >/dev/null 2>&1 || true; ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --num-gpus=4 --disable-usage-stats --block" &
sleep 12
srun --export=ALL --nodes=1 --ntasks=1 -w "$WORKER" --overlap bash -c \
  "source $R/env/env.sh; ray stop >/dev/null 2>&1 || true; ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP --num-gpus=4 --block" &
sleep 12

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/serve.sh" "$HEAD_IP" "$RAY_PORT"
