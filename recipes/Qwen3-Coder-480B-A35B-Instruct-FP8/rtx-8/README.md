# Qwen3-Coder-480B-A35B-Instruct-FP8 on one RTX PRO 6000 node

Status: Untested (migrated) - numbers below were measured 2026-07-28 with the pre-restructure scripts, protocol: slope(128,1152)

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

## Configure once

Create the API key. The endpoint refuses requests without it, and the key is passed through the
environment rather than the command line so it never appears in `ps` output.

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/vllm_api_key
chmod 600 secrets/vllm_api_key
```

Nothing else is required. Cluster paths come from `common/defaults.sh`, which is tracked with working
defaults, so a fresh clone runs as is. Four optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `QWEN3_CODER_NODE` | unset | A node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Untested (migrated). The decode rate below was measured before the restructure, with the slope method,
but with the pre-restructure scripts rather than these recipe files, so the number is comparable to
other slope-measured numbers in this repo while the recipe itself has not been exercised.

## What this is

Qwen3-Coder-480B-A35B-Instruct, FP8 quantized: a mixture-of-experts coding model with 480B total
parameters and 35B active per token (62 layers, 160 experts, 8 routed per token, GQA with 96 query and
8 key-value heads), a native 262144-token context, and a tool call format built for agentic coding
clients. It is the largest coding model in this repo that fits a single node. It emits no reasoning
blocks: this is a non-thinking model, and that is a property of the weights rather than a setting. It
exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `Qwen3-Coder-480B-A35B-Instruct-FP8`
- Hugging Face repo: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3-Coder-480B-A35B-Instruct-FP8`

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

`qwen3_moe` is native to vLLM 0.25.1, so no out-of-tree model code and no `--trust-remote-code` are
needed here.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 8 per node, 97887 MiB each |
| Architecture | sm_120, CUDA 13 |
| Interconnect | PCIe, no NVLink |
| Nodes | 1 |
| Parallelism | TP4 x PP2 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, 180 GB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. The checkpoint is about 482 GB across 49 shards, so it needs the whole node. `serve.sbatch`
requests 96 CPUs and 1000 GB, both inside the per-GPU limits for a whole node.

This is the recommended hardware for this checkpoint, not just one option among several: its FP8
kernels run on sm_120 and CUDA graphs capture cleanly here, while on H200 every capture attempt
crashed. That failure is recorded under Gotchas below.

## Environment build

This recipe builds its own environment, shared with no other recipe: about 9.0 GB for the Python
environment plus 2.9 GB for a private CUDA 13.0 toolkit. Both land under `ENV_ROOT` on VAST scratch
rather than in the repo, because startup is dominated by page faulting the torch shared objects and
stat-ing tens of thousands of small package files: measured on one node, importing torch and vLLM took
about 14 minutes from Lustre and 9.2 seconds from VAST.

```
bash recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express and a conda step no Python environment can carry. What it does, and why:

<!-- issue:cuda13-toolkit begin -->
**The sm_120 JIT needs a complete CUDA 13.0 toolkit, and it must not reach `LD_LIBRARY_PATH`.** The
node's `/usr/local/cuda-13` is runtime-only, and the fragmented pip nvcc wheels mix 13.0 and 13.2
between `nvcc`, `cicc`, and `ptxas`, which breaks the JIT. The recipe installs a consistent toolkit
via conda. `env/env.sh` points `CUDA_HOME` at it and exposes its headers through `CPATH` and
`LIBRARY_PATH` for compilation only. Do **not** add its libraries to `LD_LIBRARY_PATH`: its
`libcudart` shadows torch's CUDA 13 runtime and pulls in a `libcupti.so.13` that is not present,
which breaks import entirely.
<!-- issue:cuda13-toolkit end -->

<!-- issue:flashinfer-cubin-skew begin -->
**FlashInfer must be 0.6.15, and its version check must be bypassed.** vLLM 0.25.1 pins
flashinfer-python 0.6.13, but the sm_120 attention backend passes a `kv_scale_format` argument that
0.6.13 does not accept, which fails at the first inference request. Install 0.6.15 with `--no-deps`
so torch is left untouched. No matching 0.6.15 cubin package exists, so `flashinfer-cubin` stays at
0.6.13 and `env/env.sh` sets `FLASHINFER_DISABLE_VERSION_CHECK=1`; kernels are then compiled from
source on first launch, which is why the first request after a fresh environment is slow.
<!-- issue:flashinfer-cubin-skew end -->

The vLLM wheel comes from `https://wheels.vllm.ai/nightly/cu130`, not from PyPI, because sm_120 needs
the CUDA 13 build and uv's `--torch-backend` maxes out at cu129. `build.sh` pins `vllm==0.25.1`
explicitly rather than leaving the version to resolve: the installed metadata reads a plain `0.25.1`
with no local version tag, so an unpinned install silently drifts, either to whatever the nightly
index holds that day or to the PyPI CUDA 13 wheel, and neither is the build these numbers were
measured on. Nightly wheels also rotate and are deleted, so a much later rebuild can fail outright
instead of drifting quietly.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, which is what makes a drifted rebuild visible.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f qwen3-coder-<jobid>.log
```

Advanced path, for a node you already hold. Most users should use the Slurm submission above; this exists for reservation holders and for administrators deploying an endpoint on behalf of others, because reserved nodes are removed from the scheduler and cannot be reached with sbatch. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/serve_ssh.sh <node>
```

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

Over bare SSH `nproc` reports 1 on these nodes while `Cpus_allowed_list` is the full set, so the
server is not actually CPU limited on that path even though it looks like it.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3-coder-480b","messages":[{"role":"user","content":"Write a bubble sort in Python."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior.

This model emits no reasoning, so the answer arrives in `content` directly and a small `max_tokens`
behaves normally, unlike the thinking models in this repo. That is also why `serve.sh` passes no
`--reasoning-parser`: a parser would look for reasoning delimiters that never appear and could move
plain output into a field most clients do not display. The omission is deliberate, and `serve.sh` uses
`${REASONING_PARSER-}` rather than `${REASONING_PARSER:-...}` so an explicitly empty value stays empty.

## Connect a client

```
export NODE=<the node serving it>
source recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/client.env
claude
```

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** vLLM accepts only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which vLLM ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

For an OpenAI-compatible client instead (Cline, Aider, Continue, OpenHands), use base URL
`http://<node>:8000/v1`, the same key, and model name `qwen3-coder-480b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3-Coder-480B-A35B-Instruct-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window, against a 262144 native maximum; larger costs KV cache memory |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is forced by the FP8 block size, do not raise it |
| `PP` | 2 | Pipeline parallel size; 2 is what puts the node's other four GPUs to work |
| `EXECUTOR` | `mp` | Distributed executor backend for the pipeline stages |
| `TOOL_PARSER` | `qwen3_coder` | Tool call parser, matching this model's own call format |
| `REASONING_PARSER` | empty | Deliberately unset; this is not a thinking model |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |

## Web search

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

## Measured performance

| Configuration | Rate | Per stream | Latency |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 67.2 tok/s | 67.2 tok/s | TTFT median 44 ms, n=3 spanning 66.4 to 67.3 |
| Saturated, concurrency 8 | 267.1 tok/s | 33.4 tok/s | TTFT median 63 ms, p90 65 ms, n=3 spanning 235.7 to 289.0 |
| Saturated, concurrency 32 | 682.8 tok/s | 21.3 tok/s | TTFT median 86 ms, p90 98 ms, n=3 spanning 661.6 to 682.9 |

Measured 2026-07-30 with `common/tools/bench.sh`, endpoint ready 3m 16s after launch. Full disclosure, without which a
tokens per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 17 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Hardware | one RTX PRO 6000 Blackwell node, 8 GPUs |

Quote 67.2 tok/s for interactive coding, where one person waits on one response,
and 682.8 tok/s at concurrency 32 when the endpoint serves several people at once.
Never compare one against the other.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

The eight GPUs are used as TP4 x PP2, not TP8, and that is forced by the checkpoint rather than chosen.
Its `moe_intermediate_size` is 2560 and its FP8 quantization block is 128. At TP8 each shard holds
2560/8 = 320 columns, which is not a multiple of 128, and vLLM refuses to start. TP4 gives 640, which
is a multiple of 128, so pipeline parallelism has to cover the other half of the node.

FP8 weights are what let 480B parameters fit one node with a 128K context. Both pipeline stages live on
the same node, so the stage-to-stage point-to-point traffic never leaves the machine, and the mp
executor is enough; no Ray cluster is involved.

The cost of that shape is speculative decoding: vLLM rejects a speculative config when pipeline
parallelism is active. This checkpoint ships no draft head anyway, so nothing is lost today, but a
future draft model could not be used in this configuration.

## Gotchas

<!-- issue:coder480-tp8-divisibility begin -->
**This FP8 checkpoint cannot run at TP8.** Its `moe_intermediate_size` is 2560 and its FP8
quantization block is 128. At TP8 each shard is 2560/8 = 320, which is not a multiple of 128, and vLLM
refuses to start:

```
output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128
```

TP4 gives 640, which is a multiple of 128, so on an 8-GPU node the working shape is TP4 with PP2.
<!-- issue:coder480-tp8-divisibility end -->

<!-- issue:coder480-h200-cutlass begin -->
**This FP8 checkpoint does not run on H200 with CUDA graphs.** Four configurations were tried and all
failed at graph capture: memory utilization 0.90 and 0.96 both faulted, `VLLM_USE_DEEP_GEMM=1` gave an
illegal memory access, and the Triton path gave `cutlass_gemm_caller ... Error Internal` followed by
an illegal memory access. Memory is not the constraint; the two-node runs had 65 GiB of KV per GPU and
a 2.2M-token cache. The root cause is the CUTLASS w8a8 FP8 GEMM path faulting on Hopper for this
checkpoint during capture. Eager works at 22.2 tok/s but costs roughly 3x, so serve this model on an
RTX node, where its FP8 kernels run on sm_120 and graphs capture cleanly.
<!-- issue:coder480-h200-cutlass end -->

<!-- issue:pp-forbids-spec-decode begin -->
**Pipeline parallelism disables speculative decoding.** vLLM rejects a speculative config when
pipeline parallelism is in use, so no MTP or draft-model speedup is available in any recipe that needs
PP to span nodes, even when the checkpoint ships an MTP head. This is why an SGLang recipe exists for
GLM-5.2: SGLang can run TP8 across two nodes with EAGLE speculative decoding, where vLLM would need PP
and therefore lose it. The guard is not visible in vLLM 0.25.1's config source, so treat it as behavior
for this version rather than a documented API contract, and re-check after an engine upgrade.
<!-- issue:pp-forbids-spec-decode end -->

<!-- issue:rtx-no-nvlink begin -->
**RTX PRO 6000 nodes have no NVLink, so peer-to-peer must be disabled.** `env/env.sh` sets
`NCCL_P2P_DISABLE=1`. Without it, NCCL initialization hangs on any multi-GPU job, with no error, and
the server never becomes ready.
<!-- issue:rtx-no-nvlink end -->

<!-- issue:rtx-comms-bound begin -->
**Decode on a full RTX node is limited by cross-GPU communication, not memory bandwidth.** All-reduce
traffic crosses PCIe rather than NVLink. Two consequences, both measured: do not enable
`--enable-expert-parallel`, which added all-to-all traffic and measured about 9 percent slower, and
FP8 weights bought nothing on Qwen3-235B because weight bandwidth was not the bottleneck.
<!-- issue:rtx-comms-bound end -->

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

## Stop the endpoint

If you launched with Slurm, which is the default path, `scancel <jobid>` is all you need: Slurm terminates the job step and reclaims the node. The tool below is only for the direct SSH path, which has no scheduler to clean up after it.

```
bash common/tools/stop.sh <node>
```

That kills the server processes and waits for GPU memory to be released. Both pipeline stages run on
the same node, so one host is all you need to clean up. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
eight GPUs before you relaunch, or the next start will fail on memory.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | to be measured | skipped |
| First-time FlashInfer JIT | to be measured | skipped, cached under `/tmp/$USER/flashinfer` |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold, and the sm_120 attention
kernels are compiled from source once because no matching cubin package exists. Both caches are
node-local, so a different node pays the JIT cost again. A launch that looks hung during this window
is usually still loading. Check the log before killing it. These numbers will be filled in when this
recipe is validated on hardware; they are deliberately blank rather than guessed.
