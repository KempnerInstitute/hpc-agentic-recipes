#!/usr/bin/env bash
#SBATCH --job-name=kimi-k3
#SBATCH --partition=kempner_h200
#SBATCH --nodes=4
#SBATCH --gpus=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Kimi-K3 across four H200 nodes under SGLang: TP16 with expert parallelism, one rank per node.
#
# Sixteen GPUs is the per-user cap on this cluster, so nothing else of yours can run alongside this.
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
#
# Slurm stages this script into its spool directory, so $0 does not point into the repo and the path
# below is resolved against the submit directory instead. The log lands in the submit directory.
# The endpoint runs on the FIRST allocated node; find it with `squeue --me`.
set -euo pipefail
R="${SLURM_SUBMIT_DIR:-$PWD}/recipes/Kimi-K3/h200-4-nodes4-sglang"
[ -f "$R/serve.sh" ] || { echo "submit from the repo root: sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch" >&2; exit 2; }
source "$R/env/env.sh"

[ -f "$SIF" ] || { echo "container not found: $SIF; run env/build.sh first" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" -eq 4 ] || { echo "expected 4 nodes, got ${#NODES[@]}" >&2; exit 1; }

# Every rank dials the head's InfiniBand address to form the 16-rank group, so read it from the node
# rather than assuming a naming scheme.
HEAD_IB="$(srun --nodes=1 --ntasks=1 -w "${NODES[0]}" bash -c "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")"
[ -n "$HEAD_IB" ] || { echo "could not read ib0 on ${NODES[0]}" >&2; exit 1; }
echo "head ${NODES[0]} at $HEAD_IB; endpoint will serve on ${NODES[0]}:${API_PORT:-8000}"

# srun launches one rank per node inside the allocation, so Slurm's cgroups own every process and
# scancel tears all sixteen ranks down. Nothing else is needed to stop this job.
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 --gpus-per-task=4 --kill-on-bad-exit=1 \
  bash -c "RANK=\$SLURM_PROCID HEAD_IB='$HEAD_IB' bash '$R/serve.sh'"
