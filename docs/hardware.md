# Hardware

Three GPU types are available. All nodes of a given type share one hardware specification, so any node
in a partition behaves like any other and recipes name partitions rather than hosts.

| Type | GPUs per node | Memory per GPU | Interconnect | CUDA | Partition |
| --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | 8 | 96 GiB, 97887 MiB | PCIe, no NVLink | 13, sm_120 | `kempner_rtx` |
| H200 | 4 | 140 GiB, 143771 MiB | NVLink | 12.9, driver 575 | `kempner_h200` |
| H100 | 4 | 80 GiB, 81559 MiB | NVLink | 12.9 | `kempner_h100` |

Allocation limits per GPU: 16 CPUs on `kempner_rtx` and `kempner_h200`, 24 on `kempner_h100`. Maximum
wall time is 2 days on all three.

## What each type is good for

**RTX PRO 6000** has the most aggregate memory per node, 8 times 97887 MiB, and it is Blackwell, so it
executes NVFP4 and MXFP4 natively and captures CUDA graphs cleanly for FP8 checkpoints that fail on
Hopper. Its weakness is the interconnect: there is no NVLink, so tensor-parallel all-reduce crosses
PCIe. Decode on a full node is communication-bound rather than bandwidth-bound, which is why expert
parallelism measured slower and why FP8 weights bought nothing on one model here.

**H200** has NVLink and the highest per-GPU bandwidth, which suits models that fit in four GPUs. Two
nodes give 8 GPUs and enough memory for very large checkpoints, with tensor parallelism inside each node
and pipeline parallelism between them. Hopper has no FP4 hardware, so FP4 expert weights fall back to
emulation, and some FP8 CUTLASS kernels fault during CUDA graph capture.

**H100** is the fallback for single-GPU models. At 80 GB it cannot hold the large checkpoints here.

## Topology rules that are not obvious

Keep tensor parallelism inside a node and use pipeline parallelism across nodes. Pure tensor parallelism
spanning two nodes hangs at NCCL initialization rather than failing with an error.

Pipeline parallelism disables speculative decoding in vLLM, so any multi-node vLLM recipe gives up MTP
or draft-model speedups, even when the checkpoint ships an MTP head.

Reserved nodes are removed from the Slurm scheduler. If you hold a reservation, submitting a job that
targets those nodes will not work, which is why every recipe documents a direct SSH launch path as well
as an sbatch one.
