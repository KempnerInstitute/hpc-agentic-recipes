# Hardware

Four GPU types are available. All nodes of a given type share one specification, so any node in a
partition behaves like any other and recipes name partitions rather than hosts.

| Type | Partition | Nodes | GPUs per node | Memory per GPU | Interconnect | Target |
| --- | --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | `kempner_rtx` | 24 | 8 | 97887 MiB | PCIe, no NVLink | sm_120, CUDA 13 |
| H200 | `kempner_h200` | 96 | 4 | 143771 MiB | NVLink | sm_90, CUDA 12.9 |
| H100 | `kempner_h100` | 96 | 4 | 81559 MiB | NVLink | sm_90, CUDA 12.9 |
| A100 | `kempner` | 28 | 4 | 40960 MiB | NVLink | sm_80, CUDA 12.9 |

## Allocation limits

One user may hold **16 GPUs at once** across all four partitions, which is 2 RTX nodes or 4 H200, H100 or
A100 nodes. The largest recipe here, Kimi-K3 on 4 H200 nodes, sits exactly at that cap.

Per GPU requested you may take 16 CPUs on `kempner_rtx`, `kempner_h200` and `kempner`, and 24 on
`kempner_h100`. Host memory per GPU is 180 GB on `kempner_rtx`, 360 GB on `kempner_h200` and `kempner_h100`,
and 240 GB on `kempner`. Maximum wall time is 2 days everywhere.

## What each type is good for

**RTX PRO 6000** has the most aggregate GPU memory per node, 8 times 97887 MiB, and it is Blackwell, so it
executes NVFP4 and MXFP4 natively and captures CUDA graphs cleanly for FP8 checkpoints that fail on Hopper.
Its weakness is the interconnect: with no NVLink, tensor-parallel all-reduce crosses PCIe. Decode on a full
node is communication-bound rather than bandwidth-bound, which is why expert parallelism measured slower
here and why FP8 weights bought nothing on one model.

**H200** has NVLink and the highest per-GPU bandwidth, which suits models that fit in four GPUs. Two nodes
give 8 GPUs and enough memory for very large checkpoints, with tensor parallelism inside each node and
pipeline parallelism between them. Hopper has no FP4 hardware, so FP4 expert weights fall back to
emulation, and some FP8 CUTLASS kernels fault during CUDA graph capture.

**H100** is the fallback for single-GPU models. At 80 GB it cannot hold the large checkpoints here.

**A100** is untried for this work and no recipe targets it. Two things bound what it could serve: at 40 GiB
it holds the least of any type here, and at sm_80 it has neither FP8 nor FP4 hardware, so every quantized
checkpoint in this repo would need a build that emulates its format or a different checkpoint entirely. Its
four GPUs are fully NVLink-connected, so tensor parallelism inside a node is cheap.

## Topology rules that are not obvious

Keep tensor parallelism inside a node and use pipeline parallelism across nodes. Pure tensor parallelism
spanning two nodes hangs at NCCL initialization rather than failing with an error.

Pipeline parallelism disables speculative decoding in vLLM, so any multi-node vLLM recipe gives up MTP or
draft-model speedups even when the checkpoint ships an MTP head. SGLang has no such restriction; see
[engines.md](engines.md).
