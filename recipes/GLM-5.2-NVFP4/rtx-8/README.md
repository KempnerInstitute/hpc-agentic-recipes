# GLM-5.2-NVFP4 on one RTX PRO 6000 node

Status: Validated - 2026-07-29, vLLM 0.25.1, protocol: slope(128,1152)

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
| `GLM52_NVFP4_NODE` | unset | A node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Validated on 2026-07-29 through the canonical Slurm path. The environment was built from
`env/build.sh` on a fresh netscratch tree, the job was submitted with
`sbatch --account=<acct> recipes/GLM-5.2-NVFP4/rtx-8/serve.sbatch`, and the rate was measured with
`common/tools/bench.sh`. Ready 14 minutes 51 seconds after the job started, which includes loading 47
shards and compiling the FlashInfer sm_120 kernels from source for the first time. Key gating and the
Anthropic endpoint were both confirmed.

Measured 101.6 tok/s, against the about 90 tok/s recorded before the restructure. The difference is the
measurement protocol, not a change in the model: the older figure used a single timed generation, which
counts prefill and fixed per-request cost as decode time. That bias is largest for fast models, and at
roughly 100 tok/s a 1152-token generation takes 12 seconds, so about a second of fixed cost was
suppressing the reported rate by more than 10 percent.

Untested (migrated). The decode rate below was measured before the restructure, with the older
single-generation protocol rather than the slope method, so treat it as conservative and not directly
comparable to slope-measured numbers elsewhere in this repo.

## What this is

GLM-5.2 quantized to NVFP4 by NVIDIA Model Optimizer: a 744B-parameter mixture-of-experts reasoning
and coding model that uses DeepSeek-style sparse attention for long context, at near-FP8 quality. It
exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy. This is
the fastest endpoint in this repo.

- Checkpoint directory: `GLM-5.2-NVFP4`
- Hugging Face repo: `nvidia/GLM-5.2-NVFP4`, quantized from `zai-org/GLM-5.2`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-NVFP4`

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 8 per node, 97887 MiB each |
| Architecture | sm_120, CUDA 13 |
| Interconnect | PCIe, no NVLink |
| Nodes | 1 |
| Parallelism | TP8 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, 180 GB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. The checkpoint is about 465 GB across 47 shards, which fits the node's VRAM with room for a
128K-token KV cache. `serve.sbatch` requests 96 CPUs and 500 GB, both inside the per-GPU limits for a
whole node.

## Environment build

This recipe builds its own environment, shared with no other recipe: about 9.0 GB for the Python
environment plus 2.9 GB for a private CUDA 13.0 toolkit. Both land under `ENV_ROOT` on VAST scratch
rather than in the repo, because startup is dominated by page faulting the torch shared objects and
stat-ing tens of thousands of small package files: measured on one node, importing torch and vLLM took
about 14 minutes from Lustre and 9.2 seconds from VAST.

```
bash recipes/GLM-5.2-NVFP4/rtx-8/env/build.sh
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
sbatch --account=<your-account> recipes/GLM-5.2-NVFP4/rtx-8/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f glm52-nvfp4-<jobid>.log
```

Advanced path, for a node you already hold. Most users should use the Slurm submission above; this exists for reservation holders and for administrators deploying an endpoint on behalf of others, because reserved nodes are removed from the scheduler and cannot be reached with sbatch. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/GLM-5.2-NVFP4/rtx-8/serve_ssh.sh <node>
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
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field, not `reasoning_content`. With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length`, and `content` is empty,
which looks like a broken endpoint but is not. Use at least 400 output tokens for a smoke test, and 800
or more for a model that reasons at length. If `content` is empty, raise the budget before suspecting
the endpoint.
<!-- issue:thinking-model-max-tokens end -->

## Connect a client

```
export NODE=<the node serving it>
source recipes/GLM-5.2-NVFP4/rtx-8/client.env
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
`http://<node>:8000/v1`, the same key, and model name `glm-5.2`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.2-NVFP4` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window; larger costs KV cache memory |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size; 8 is the node's GPU count |
| `MTP_TOKENS` | 3 | Speculative tokens per step |
| `NO_MTP` | unset | Set to disable speculative decoding |
| `ATTN_BACKEND` | unset | Override vLLM's attention backend selection |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
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

| Configuration | Rate | Per stream | Notes |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 94.3 tok/s | 94.3 tok/s | interactive latency, TTFT 89 ms |
| Saturated, concurrency 8 | 378.4 tok/s | 47.3 tok/s | aggregate across 8 requests |
| Saturated, concurrency 32 | 691.7 tok/s | 21.6 tok/s | peak measured |

Measured 2026-07-30 with `common/tools/bench.sh`. Full disclosure, without which a tokens per second
figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 21 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Protocol | slope(128,1152) |
| Hardware | one RTX PRO 6000 Blackwell node, 8 GPUs |

Per stream latency degrades sharply under load here, from 94.3 to 21.6 tok/s. That is the MTP
speculative decoding losing its advantage: speculation spends extra compute to save latency, which pays
off when the GPU is idle at concurrency 1 and competes with real work once the batch is full.

Quote the single stream figure for interactive coding and the aggregate for serving several people.
Never compare one against the other.

Context length turned out not to matter much for this model. Measured at concurrency 1:

| ISL, input tokens | Decode | TTFT |
| --- | --- | --- |
| 21 | 97.0 tok/s | 91 ms |
| 7052 | 97.8 tok/s | 123 ms |
| 26379 | 95.0 tok/s | 131 ms |

Decode is flat to 26K within run-to-run noise, and only time to first token grows, which is prefill
doing more work. That is a property of the attention design rather than a general rule: MLA compresses
the KV cache, it is stored as fp8, and sparse attention reads only a subset of the context, so the
per-step read barely grows. Do not carry this result over to a dense model with full attention.

## Parallelism and quantization

TP8 spans all eight GPUs of one node. Every all-reduce crosses PCIe rather than NVLink, which is the
ceiling on this configuration, so nothing that adds cross-GPU traffic is worth enabling here.

NVFP4 is what makes a 744B-parameter model fit one node: 4-bit weights and 4-bit input activations at
group size 16, with the embeddings, `lm_head` and the first layer left unquantized. The KV cache is
FP8 in DeepSeek's sparse-MLA layout (`--kv-cache-dtype fp8_ds_mla`), which is the format vLLM's sm_120
sparse-MLA backend reads, and it is what leaves room for a 128K context.

Pipeline parallelism is not used and not needed here, which matters because vLLM rejects speculative
decoding when pipeline parallelism is active. Staying single-node is what makes MTP available.

## Gotchas

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

That kills the server processes and waits for GPU memory to be released. Confirm with
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
