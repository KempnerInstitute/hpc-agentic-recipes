# Qwen3-Coder-480B-A35B-Instruct-FP8 on H200: blocked

Status: Blocked - this checkpoint does not run on H200 with CUDA graphs on vLLM 0.25.1

This directory holds a negative result. There are no launch scripts here on purpose: four
configurations were tried on H200 and all four failed at CUDA graph capture, while the same checkpoint
serves cleanly on an RTX node. If you came looking for a way to run this model on Hopper, this page is
the answer, so that the next person spends five minutes reading rather than a day rediscovering.

**Serve this model at `recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8` instead**, where its FP8
kernels run on sm_120, CUDA graphs capture, and the measured decode rate is 67.7 tok/s single stream at TP4 x PP2 on
one node, protocol slope(128,1152).

## What was tried

Measured on this cluster on vLLM 0.25.1, outside this recipe's scripts.
The checkpoint is 449 GiB (482.2 GB decimal), `Qwen3MoeForCausalLM`, 62 layers, 160 experts with 8
active per token, `moe_intermediate_size` 2560, 262144 native context.

| H200 attempt | Outcome |
| --- | --- |
| 4 GPUs, TP4, Triton MoE | crash during capture |
| 2 nodes, TP4 x PP2, DeepGEMM | `CUDA error: an illegal memory access was encountered` |
| 2 nodes, TP4 x PP2, Triton MoE | `cutlass_gemm_caller ... Error Internal`, then illegal memory access |
| 4 GPUs, TP4, `--enforce-eager` | works, 22.2 tok/s |

Graph capture was attempted at `--gpu-memory-utilization` 0.90 and again at 0.96; both faulted, which
is part of why memory was ruled out as the cause.

## Why it is not a memory problem

The obvious first hypothesis is that graph capture ran out of VRAM, and that was checked and rejected.
The two-node runs had 65 GiB of KV cache per GPU and a 2.2M-token cache, which is far more headroom
than the model needs at any context length anyone would use for agentic coding. Raising utilization
from 0.90 to 0.96 did not change the outcome either. The failure is not about how much memory is
available.

## Root cause, as far as it was traced

The CUTLASS w8a8 FP8 GEMM path faults on Hopper for this checkpoint during CUDA graph capture. The
`cutlass_gemm_caller ... Error Internal` message from the Triton MoE run names it directly, and the
illegal memory access that follows is the same fault surfacing later. Both MoE backends fail, so the
problem is below the MoE backend choice.

<!-- issue:coder480-h200-cutlass begin -->
**This FP8 checkpoint does not run on H200 with CUDA graphs.** Four configurations were tried and all
failed at graph capture: memory utilization 0.90 and 0.96 both faulted, `VLLM_USE_DEEP_GEMM=1` gave an
illegal memory access, and the Triton path gave `cutlass_gemm_caller ... Error Internal` followed by
an illegal memory access. Memory is not the constraint; the two-node runs had 65 GiB of KV per GPU and
a 2.2M-token cache. The root cause is the CUTLASS w8a8 FP8 GEMM path faulting on Hopper for this
checkpoint during capture. Eager works at 22.2 tok/s but costs roughly 3x, so serve this model on an
RTX node, where its FP8 kernels run on sm_120 and graphs capture cleanly.
<!-- issue:coder480-h200-cutlass end -->

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

Forcing `VLLM_USE_DEEP_GEMM=1` on H200 is what produced the second row of the table. That flag is kept
at 0 for every H200 recipe in this repo, and this checkpoint is one of the two models that independently
justified it.

## Why the eager fallback is not the answer

Eager mode does work: 22.2 tok/s at TP4 on 4 GPUs. That is roughly a third of the 67.7 tok/s the same
checkpoint reaches on one RTX node, so the fallback costs about 3x for a full H200 node held for a day.
Given that an RTX node runs the model properly, spending Hopper capacity on a 3x-slower configuration
is a poor trade unless there is a specific reason to hold H200 hardware.

## The other constraint, which is not hardware specific

<!-- issue:coder480-tp8-divisibility begin -->
**This FP8 checkpoint cannot run at TP8.** Its `moe_intermediate_size` is 2560 and its FP8
quantization block is 128. At TP8 each shard is 2560/8 = 320, which is not a multiple of 128, and vLLM
refuses to start:

```
output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128
```

TP4 gives 640, which is a multiple of 128, so on an 8-GPU node the working shape is TP4 with PP2.
<!-- issue:coder480-tp8-divisibility end -->

On a 4-GPU H200 node TP4 is the whole node, so this constraint never forced a shape change here; it is
what makes the working RTX configuration TP4 x PP2 rather than TP8. It is recorded here because
somebody trying to rescue the H200 path will reach for a different TP size early, and TP8 is not
available.

## What would have to change

Any of these would justify retrying, and none has happened:

- A vLLM release that fixes the CUTLASS w8a8 FP8 GEMM fault on Hopper for this checkpoint. The failure
  is in the kernel path, not in this repo's configuration, so an engine upgrade is the realistic fix.
- A bf16 twin of this checkpoint, which never touches the FP8 GEMM path at all. That is
  `recipes/Qwen3-Coder-480B-A35B-Instruct`, documentation-only and untested, and it is the cleanest way
  to find out whether Coder-480B can serve on H200 in any precision.
- An H200-friendly requantization of the experts.

If you do retry, the environment the attempts used was the Hopper one:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

## Gotchas that still apply if you retry

<!-- issue:lustre-watchdog begin -->
**A storage stall can kill the endpoint even after the storage recovers.** PyTorch kills the process
when the NCCL watchdog thread stops sending heartbeats, on the assumption that a collective hung. A
stalled network filesystem freezes every rank the same way, so at the 480 second default a transient
storage outage takes the endpoint down permanently rather than pausing it. The signature is every rank
reporting `Last enqueued NCCL work: -1`, meaning no collective was ever in flight, so the process was
frozen rather than genuinely hung on communication. `env/env.sh` sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` so a stall that resolves within an hour is survivable.
<!-- issue:lustre-watchdog end -->

<!-- issue:engine-ready-timeout begin -->
**Startup exceeds vLLM's default readiness timeout.** Weight load plus torch.compile plus CUDA graph
capture routinely takes longer than the 600 second default, so `env/env.sh` sets
`VLLM_ENGINE_READY_TIMEOUT_S=3600`. A first launch that looks hung is usually still loading; check
the log before killing it.
<!-- issue:engine-ready-timeout end -->

<!-- issue:node-local-logs begin -->
**Logs are written to node-local `/tmp`, not to the repo.** Every rank writes stderr for the life of
the endpoint, so a log on a network filesystem puts a blocking write on the critical path. During a
filesystem stall that write hangs, which freezes the server. `LOG_DIR` defaults to
`/tmp/$USER/vllm`, so read logs over SSH on the node that runs the server.
<!-- issue:node-local-logs end -->

## Client access, for the RTX endpoint

Nothing here serves anything, so there is no client configuration in this directory. When you connect
to the working RTX endpoint, two client facts still catch people out:

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** vLLM accepts only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which vLLM ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

<!-- issue:anthropic-hosted-tools-400 begin -->
**Anthropic's hosted tools fail against this endpoint with HTTP 400.** Claude Code's built-in web
search sends tool definitions of type `web_search_20250305` that carry no `input_schema`, and vLLM
rejects them:

```
API Error: 400 1 validation error: 'loc': ('body', 'tools', 0, 'input_schema'),
'msg': 'Field required', 'type': 'web_search_20250305'
```

Client-side tools (file edits, shell, and anything you define) work normally. For web access, install
the repo's keyless search tool and skill:

```
ln -sf "$REPO_ROOT/common/tools/search.sh" ~/.local/bin/search.sh
cp -r "$REPO_ROOT/common/skills/local-search" ~/.claude/skills/
```

Then the model searches through `search.sh` (web, arxiv, crossref, pubmed, openalex, wiki, fetch)
instead of the hosted tool.
<!-- issue:anthropic-hosted-tools-400 end -->
