#!/usr/bin/env bash
#SBATCH --job-name=kimi-h200
#SBATCH --partition=kempner_h200
#SBATCH --nodes=2
#SBATCH --gpus=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Kimi-K2.7-Code across two H200 nodes: Ray, TP4 inside each node, PP2 between them.
# serve.sh defaults EXTRA_ARGS to --skip-mm-profiling --mm-processor-cache-gb 0, without which the
# launch hangs indefinitely at cross-node multimodal profiling. Nothing needs to be set here for that.
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Kimi-K2.7-Code/h200-4-nodes2/serve.sbatch
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Kimi-K2.7-Code/h200-4-nodes2
RAY_PORT="${RAY_PORT:-6379}"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD="${nodes[0]}"
WORKER="${nodes[1]}"
# Ray needs a routable address rather than a hostname for both ranks, and it must be the interface the
# collectives use: env/env.sh pins NCCL and Gloo to ib0, so the Ray addresses are the ib0 addresses.
ipof () { srun --nodes=1 --ntasks=1 -w "$1" --overlap bash -c 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"
echo "serving kimi-k2.7-code: http://$HEAD:${API_PORT:-8000}/v1"

# RAY_BLOCK keeps each ray start in the foreground so Slurm does not tear its step down while vLLM is
# using the cluster. The sleeps let the head bind its port before the worker dials in, and let the
# worker register before the engine asks Ray for 8 GPUs.
srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/ray_head.sh" "$HEAD_IP" "$RAY_PORT" &
sleep 12
srun --export=ALL,RAY_BLOCK=1 --nodes=1 --ntasks=1 -w "$WORKER" --overlap bash "$R/ray_worker.sh" "$HEAD_IP" "$RAY_PORT" "$WORKER_IP" &
sleep 12

srun --export=ALL --nodes=1 --ntasks=1 -w "$HEAD" --overlap bash "$R/serve.sh" "$HEAD_IP" "$RAY_PORT"
