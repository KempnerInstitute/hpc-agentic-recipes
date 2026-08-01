# Cluster defaults. Tracked on purpose, so a fresh clone works with no configuration.
# Source this, do not execute. Override precedence: this file, then common/site.conf, then the
# environment. Nothing here is user-specific.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/repo_root.sh"

# Where checkpoints live. This shared testbed copy is readable by any cluster user and is the permanent
# one. A copy on your own scratch space loads faster; override MODELS_DIR to use it.
: "${MODELS_DIR:=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models}"

# Where per-recipe environments are built. Scratch, because startup is dominated by page
# faulting the torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes, the
# interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
# seconds from scratch. A bare torch and vLLM import from scratch is 9.2 seconds, so most of that 58
# seconds is engine startup rather than filesystem cost.
# Scratch has a 90-day retention policy, so environments here are disposable and must be rebuilt
# periodically with common/tools/rebuild_envs.sh. This is a deliberate speed tradeoff.
# The default names one group's scratch space, which is the only part of this file that assumes
# anything about who you are. If you cannot write there, set ENV_ROOT in common/site.conf to any
# directory you can: lab or project space with no retention policy also makes environments
# permanent, at the cost of slower imports. The check below says so rather than letting a build
# fail later with a bare permission error.
: "${ENV_ROOT:=/n/netscratch/kempner_dev/Lab/$USER/agentic-coding-llm-env}"
if ! mkdir -p "$ENV_ROOT" 2>/dev/null; then
  echo "cannot create ENV_ROOT=$ENV_ROOT" >&2
  echo "set ENV_ROOT in common/site.conf to a directory you can write" >&2
fi

# Server logs. Node-local, because every rank writes stderr for the life of the endpoint: a log on a
# network filesystem puts a blocking write on the critical path, and a filesystem stall then freezes
# the server until its own NCCL watchdog kills it.
: "${LOG_DIR:=/tmp/$USER/vllm}"

# uv's package cache, deliberately placed on the same filesystem as ENV_ROOT. uv hardlinks package
# files from its cache into each environment; when the cache sits on a different filesystem it falls
# back to copying every file, so each recipe writes about 13 GB instead of linking it. Export it so
# every build inherits the same choice regardless of the caller's environment.
: "${UV_CACHE_DIR:=$ENV_ROOT/.uv-cache}"
export UV_CACHE_DIR

: "${API_PORT:=8000}"

# Partitions. All nodes of a given type share one hardware spec, so only the partition matters.
#   kempner_rtx   RTX PRO 6000 Blackwell, 8 GPUs per node, 97887 MiB each, sm_120, PCIe, CUDA 13
#   kempner_h200  H200, 4 GPUs per node, 143771 MiB each, NVLink, CUDA 12.9
#   kempner_h100  H100, 4 GPUs per node, 80 GB each, NVLink, CUDA 12.9
# Per-GPU allocation limits: 16 CPUs on kempner_rtx and kempner_h200, 24 on kempner_h100.
# Maximum wall time is 2 days on all three.
: "${PARTITION_RTX:=kempner_rtx}"
: "${PARTITION_H200:=kempner_h200}"
: "${PARTITION_H100:=kempner_h100}"

# Slurm account. There is no correct default for a public repo, so pass --account at submit time or
# set ACCOUNT in common/site.conf.
: "${ACCOUNT:=}"

# Optional local overrides: hostnames you hold, and ACCOUNT. Gitignored.
if [ -f "$REPO_ROOT/common/site.conf" ]; then
  source "$REPO_ROOT/common/site.conf"
fi

# Return success explicitly. A sourced file's exit status is that of its last command, and callers run
# under set -e: ending on a failed test for an optional file would abort them silently, with no output.
:
