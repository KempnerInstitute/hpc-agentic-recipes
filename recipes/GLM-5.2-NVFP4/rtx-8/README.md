# GLM-5.2-NVFP4 on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) swept at concurrency 1 through 512

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
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on one RTX PRO 6000 Blackwell node, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256 and 512. Ready 18 minutes 1 second after launch. The endpoint was still answering after the sweep finished.

This run measured 93.4 tok/s single stream, against an earlier 101.6 tok/s. Both used the
slope method, so the protocol is not the difference. This recipe serves with MTP speculative decoding
at 3 draft tokens, and speculative gain depends on how predictable the generated text is, so its
single stream rate is less reproducible run to run than a non-speculative model's. Treat roughly 93
to 102 tok/s as the observed range rather than either endpoint as exact. Set `NO_MTP=1` to measure
the model without speculation if you need a stable baseline.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

GLM-5.2 quantized to NVFP4 by NVIDIA Model Optimizer: a 753B-parameter mixture-of-experts, 40B activated per token, reasoning
and coding model that uses DeepSeek-style sparse attention for long context, at near-FP8 quality. It
exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy. This is
the fastest large model in this repo; the small Gemma models are faster outright.

- Checkpoint directory: `GLM-5.2-NVFP4`
- Hugging Face repo: `nvidia/GLM-5.2-NVFP4`, quantized from `zai-org/GLM-5.2`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-NVFP4`

The testbed path works out of the box. Copying the checkpoint into your own scratch space loads
faster, because scratch outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the permanent copy.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 8 per node, 97887 MiB each |
| Architecture | sm_120, CUDA 13 |
| Interconnect | PCIe, no NVLink |
| Nodes | 1 |
| Parallelism | TP8 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, about 189 GiB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. The checkpoint is about 465 GB across 47 shards, which fits the node's VRAM with room for a
128K-token KV cache. `serve.sbatch` requests 96 CPUs and 500 GB, both inside the per-GPU limits for a
whole node.

## Environment build

This recipe builds its own environment, shared with no other recipe: about 9.0 GB for the Python
environment plus 2.9 GB for a private CUDA 13.0 toolkit. Both land under `ENV_ROOT` on scratch
rather than in the repo, because startup is dominated by page faulting the torch shared objects and
stat-ing tens of thousands of small package files: measured on one node, importing torch and vLLM took
about 14 minutes from Lustre and 9.2 seconds from scratch.

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

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/GLM-5.2-NVFP4/rtx-8/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f glm52-nvfp4-<jobid>.log
```

Direct path, for a node you already hold. Use the Slurm submission above unless you already have
the node, or you are deploying an endpoint on behalf of others:

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

| Configuration | Aggregate rate | Per stream | Latency |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 93.4 tok/s | 93.4 tok/s | TTFT median 121 ms, n=3 spanning 90.7 to 93.7 |
| Concurrency 8 | 368.3 tok/s | 46.0 tok/s | TTFT median 235 ms, p90 236 ms, n=3 spanning 361.3 to 378.9 |
| Concurrency 32 | 682.6 tok/s | 21.3 tok/s | TTFT median 628 ms, p90 638 ms, n=3 spanning 669.2 to 688.5 |
| Concurrency 64 | 1049.0 tok/s | 16.4 tok/s | TTFT median 1038 ms, p90 1068 ms, n=3 spanning 1047.0 to 1050.7 |
| Concurrency 128 | 1266.6 tok/s | 9.9 tok/s | TTFT median 1826 ms, p90 2035 ms, n=3 spanning 1266.0 to 1271.9 |
| Concurrency 256 (peak) | 1389.1 tok/s | 5.4 tok/s | TTFT median 3478 ms, p90 3533 ms, n=3 spanning 1357.6 to 1405.3 |
| Concurrency 512 | 1239.8 tok/s | 2.4 tok/s | TTFT median 5473 ms, p90 7606 ms, n=3 spanning 1179.5 to 1240.4 |

Measured with `common/tools/bench.sh`, endpoint ready 18m 1s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 21 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | one RTX PRO 6000 Blackwell node, 8 GPUs |

Quote 93.4 tok/s for interactive coding, where one person waits on one
response. Quote 1389.1 tok/s at concurrency 256 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput **peaks at concurrency 256** and falls to 1240 tok/s by concurrency 512, so 1389.1 tok/s is a
measured ceiling for this recipe rather than the edge of the sweep.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

TP8 spans all eight GPUs of one node. Every all-reduce crosses PCIe rather than NVLink, which is the
ceiling on this configuration, so nothing that adds cross-GPU traffic is worth enabling here.

NVFP4 is what makes a 753B-parameter model fit one node: 4-bit weights and 4-bit input activations at
group size 16. The `ignore` list in `quantization_config` keeps `lm_head`, `model.embed_tokens`, the
first two layers, and every layer's attention and shared experts at full precision. The KV cache is
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
