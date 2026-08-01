# gemma-4-26B-A4B-it on one RTX PRO 6000 GPU

Status: Validated - vLLM 0.26.0, protocol: slope(128,1152) swept at concurrency 1 through 1024

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
defaults, so a fresh clone runs as is. Optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `GEMMA26_RTX_NODE` | unset | An RTX node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on one RTX PRO 6000 Blackwell GPU, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024. Ready 10 minutes 48 seconds after launch. The endpoint was still answering after the sweep finished.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

Gemma 4 26B-A4B, instruction tuned: a mixture-of-experts model that activates 4B of its 26B
parameters per token, with native tool calling and a separate reasoning channel. It is the cheapest
endpoint in this repo, one GPU with no NCCL and no multi-node coordination, and the fastest by decode
rate, which makes it the best default for interactive work. It exposes vLLM's Anthropic-compatible
API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `gemma-4-26B-A4B-it`
- Hugging Face repo: `google/gemma-4-26B-A4B-it`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-26B-A4B-it`
- On disk: 51.6 GB, bf16, `Gemma4ForConditionalGeneration`, multimodal, 256K context
- Optional drafter: `gemma-4-26B-A4B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and
  currently unusable on this engine (see Gotchas)

The testbed path works out of the box. Copying the checkpoint into your own scratch space loads
faster, because scratch outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the permanent copy.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 1 of the 8 on the node, 97887 MiB |
| Nodes | 1 |
| Parallelism | TP1 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, about 189 GiB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. This recipe claims one GPU and leaves the other seven to other jobs, so expect to share the
node; on the SSH path set `GPU=<n>` to pin a device.

Weights are 51.6 GB and the KV cache costs 40 KiB per token, from 5 full attention layers plus 25
sliding-window layers of 1024 tokens. So 32K of context costs 1.5 GB and the full 256K costs 10 GB,
and one 96 GB card covers weights plus the maximum context with room left over.

sm_120 and CUDA 13 are why this variant has its own toolchain, and why the RTX, H200, and H100
variants of this checkpoint are three separate recipes rather than one recipe with a switch.

## Environment build

This recipe builds its own environment, shared with no other recipe: roughly 9.0 GB for the virtual
environment plus 2.9 GB for a CUDA 13.0 toolkit. Both land under `ENV_ROOT` on scratch rather
than in the repo, because startup is dominated by page faulting the torch shared objects and stat-ing
tens of thousands of small package files: measured on one node, importing torch and vLLM took about
14 minutes from Lustre and 9.2 seconds from scratch.

```
bash recipes/gemma-4-26B-A4B-it/rtx-1/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express plus a conda step. What it does, and why:

torch and vLLM come from the CUDA 13 nightly indexes, `https://wheels.vllm.ai/nightly/cu130` and the
PyTorch cu130 index, with `--prerelease=allow --index-strategy unsafe-best-match`. sm_120 needs the
CUDA 13 build, which does not ship as a stable release wheel, and the index strategy is what lets
torch resolve from the PyTorch index while vLLM resolves from the vLLM one.

<!-- issue:flashinfer-cubin-skew begin -->
**FlashInfer must be 0.6.15, and its version check must be bypassed.** vLLM 0.25.1 pins
flashinfer-python 0.6.13, but the sm_120 attention backend passes a `kv_scale_format` argument that
0.6.13 does not accept, which fails at the first inference request. Install 0.6.15 with `--no-deps`
so torch is left untouched. No matching 0.6.15 cubin package exists, so `flashinfer-cubin` stays at
0.6.13 and `env/env.sh` sets `FLASHINFER_DISABLE_VERSION_CHECK=1`; kernels are then compiled from
source on first launch, which is why the first request after a fresh environment is slow.
<!-- issue:flashinfer-cubin-skew end -->

<!-- issue:cuda13-toolkit begin -->
**The sm_120 JIT needs a complete CUDA 13.0 toolkit, and it must not reach `LD_LIBRARY_PATH`.** The
node's `/usr/local/cuda-13` is runtime-only, and the fragmented pip nvcc wheels mix 13.0 and 13.2
between `nvcc`, `cicc`, and `ptxas`, which breaks the JIT. The recipe installs a consistent toolkit
via conda. `env/env.sh` points `CUDA_HOME` at it and exposes its headers through `CPATH` and
`LIBRARY_PATH` for compilation only. Do **not** add its libraries to `LD_LIBRARY_PATH`: its
`libcudart` shadows torch's CUDA 13 runtime and pulls in a `libcupti.so.13` that is not present,
which breaks import entirely.
<!-- issue:cuda13-toolkit end -->

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. Record the exact resolution in `env/requirements.lock` and any non-PyPI artifact URLs with their
hashes in `env/WHEELS` after a build, which is what makes a drifted rebuild visible. Both matter more here than on
a Hopper recipe: the nightly index rotates and deletes wheels, and the installed metadata reads
`vllm 0.25.1` with no local version tag, so a naive `vllm==0.25.1` silently resolves to the PyPI
CUDA 13 wheel, which is a different environment from the one that was tested.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/rtx-1/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f gemma26-rtx-1-<jobid>.log
```

Direct path, for a node you already hold. Use the Slurm submission above unless you already have
the node, or you are deploying an endpoint on behalf of others:

```
bash recipes/gemma-4-26B-A4B-it/rtx-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-26B-A4B-it/rtx-1/serve_ssh.sh <node>    # pin one device of a busy node
```

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
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
source recipes/gemma-4-26B-A4B-it/rtx-1/client.env
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
`http://<node>:8000/v1`, the same key, and model name `gemma-4-26b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 32768 | Context window; 262144 is supported and costs 10 GB of KV cache |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; one GPU holds this checkpoint at full context |
| `QUANT` | unset, meaning bf16 | `fp8` quantizes weights on load; measured no change in rate for this MoE, so use it only to free VRAM |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, for debugging a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint. Must stay unset on vLLM 0.25.1 and 0.26.0, where it fails at the first request; see Gotchas |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, only read when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node, for example `GPU=1` |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |

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
| Single stream, concurrency 1 | 140.6 tok/s | 140.6 tok/s | TTFT median 35 ms, n=3 spanning 140.4 to 140.6 |
| Concurrency 8 | 837.6 tok/s | 104.7 tok/s | TTFT median 37 ms, p90 52 ms, n=3 spanning 837.5 to 837.6 |
| Concurrency 32 | 2188.6 tok/s | 68.4 tok/s | TTFT median 89 ms, p90 105 ms, n=3 spanning 2173.4 to 2188.8 |
| Concurrency 64 | 3457.8 tok/s | 54.0 tok/s | TTFT median 119 ms, p90 147 ms, n=3 spanning 3444.0 to 3482.2 |
| Concurrency 128 | 4637.9 tok/s | 36.2 tok/s | TTFT median 173 ms, p90 239 ms, n=3 spanning 4606.4 to 4638.0 |
| Concurrency 256 | 4431.0 tok/s | 17.3 tok/s | TTFT median 265 ms, p90 387 ms, n=3 spanning 4428.8 to 4431.5 |
| Concurrency 512 | 5403.5 tok/s | 10.6 tok/s | TTFT median 463 ms, p90 768 ms, n=3 spanning 5398.5 to 5465.2 |
| Concurrency 640 | 5463.9 tok/s | 8.5 tok/s | TTFT median 532 ms, p90 887 ms, n=3 spanning 5462.5 to 5466.9 |
| Concurrency 768 | 5750.4 tok/s | 7.5 tok/s | TTFT median 625 ms, p90 1065 ms, n=3 spanning 5740.4 to 5790.3 |
| Concurrency 896 | 5889.3 tok/s | 6.6 tok/s | TTFT median 724 ms, p90 1213 ms, n=3 spanning 5813.6 to 5931.9 |
| Concurrency 1024 (rising) | 5972.0 tok/s | 5.8 tok/s | TTFT median 701 ms, p90 1185 ms, n=3 spanning 5953.9 to 6004.2 |

Measured with `common/tools/bench.sh`, endpoint ready 10m 48s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 19 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | one RTX PRO 6000 Blackwell GPU |

Quote 140.6 tok/s for interactive coding, where one person waits on one
response. Quote 5972.0 tok/s at concurrency 1024 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput was **still rising at concurrency 1024**, 11 percent above its own concurrency 512 at the top of the sweep,
so 5972.0 tok/s is a floor rather than a ceiling. The sequence cap is not what stopped it:
`max_num_seqs` resolves to 1024 on this hardware and the sweep ran to that level, so finding the
true peak needs the cap raised, which is a different serving configuration.

Concurrency 512 was measured in both runs, at 5403.5 and 5401.2 tok/s, a -0.0 percent
difference. That is the check that the two halves of this curve are comparable.

Scheduler counters over the extended levels: KV cache usage reached 100 percent, the running
batch reached 1024 requests, and there were no preemptions at any level.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

TP1 is not a compromise here. One GPU holds the 51.6 GB of bf16 weights plus a 10 GB KV cache at the
full 256K context, and at TP1 there is no all-reduce at all, which matters on this node type because
it has no NVLink and tensor-parallel traffic would otherwise cross host memory.

bf16, with no `--quantization` flag, is the measured right answer for this checkpoint, and it is the
most interesting thing about it. FP8 weights changed the decode rate by nothing. The model activates
only 4B of its 26B parameters per token, so there is very little weight traffic for a narrower dtype
to save: during decode, GPU utilization measured 35 to 40 percent and power draw about 210 W of a
700 W budget. This endpoint is host overhead bound, not memory bandwidth bound, so quantization
addresses the wrong constraint. The same regime explains why the rate scales only about 1.3x per GPU
tier instead of tracking HBM bandwidth.

Set `QUANT=fp8` only when you want the VRAM back, for instance to run a second endpoint on the same
card. `KV_FP8=1` likewise measured no change in rate and only buys context, which this checkpoint does
not need on a 96 GB card.

The contrast with the dense 31B sibling is the point: there FP8 is worth 74 percent on this same GPU,
because a dense model reads all of its weights for every token. Same family, same flag, opposite
answer.

TP2 was not measured. The bottleneck is host overhead rather than weight bandwidth, so more GPUs
address the wrong constraint, but the honest statement is that it is untested.

## Gotchas

<!-- issue:gemma4-mtp-broken begin -->
**Speculative decoding is unavailable for Gemma 4 on this engine.** The drafter checkpoints
(`*-it-assistant`) are downloaded and wired through `SPEC_DRAFT`, but the `gemma4_mtp` method is
broken in both vLLM 0.25.1 and 0.26.0, failing with a shape mismatch:

```
a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]
```

The drafter expects twice 2816 rather than 5632. The fix (`_maybe_share_embeddings`) exists only on
vLLM's `main` branch, so this needs a build newer than 0.26.0. Do not enable `SPEC_DRAFT` on 0.25.1
or 0.26.0; it fails at the first request.
<!-- issue:gemma4-mtp-broken end -->

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
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on the
device this endpoint used before you relaunch, or the next start will fail on memory. On a shared node
check only that device: another job may legitimately be holding the other seven.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 20 min | skipped |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold, and the FlashInfer sm_120
kernels are compiled from source on the first request after a fresh environment because no matching
cubin package exists. A launch that looks hung during this window is usually still loading. Check the
log before killing it. These numbers will be filled in when this recipe is validated on hardware; they
are deliberately blank rather than guessed.
