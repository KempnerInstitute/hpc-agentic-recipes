# gemma-4-31B-it on one H100 GPU

Status: Validated - 2026-07-31, vLLM 0.25.1+cu129, protocol: slope(128,1152) swept at concurrency 1 through 512

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
| `GEMMA31_H100_NODE` | unset | An H100 node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Validated on 2026-07-31. The environment was built from `env/build.sh`, the endpoint was
launched with `serve.sbatch` through Slurm on one H100 GPU, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256 and 512. The endpoint was still answering after the sweep finished.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

Gemma 4 31B, instruction tuned: a dense model, so every parameter is read for every token, with native
tool calling and a separate reasoning channel. It is the higher-quality half of the single-GPU Gemma 4
pair and roughly a third the decode rate of its 26B mixture-of-experts sibling. It exposes vLLM's
Anthropic-compatible API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `gemma-4-31B-it`
- Hugging Face repo: not recorded before the restructure; the testbed copy is the system of record
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-31B-it`
- On disk: 62.6 GB, bf16, `Gemma4ForConditionalGeneration`, multimodal, 256K context
- Optional drafter: `gemma-4-31B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and currently
  unusable on this engine (see Gotchas)

This recipe serves the bf16 checkpoint with FP8 weight quantization applied at load time, which is why
there is no separate FP8 checkpoint directory.

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H100, 1 of the 4 on the node, 80 GB |
| Nodes | 1 |
| Parallelism | TP1 |
| Partition | `kempner_h100` |
| Per-GPU allocation limit | 24 CPUs |
| Maximum wall time | 2 days |

All H100 nodes on this cluster share one hardware specification, so any node in the partition works.
This recipe claims one GPU and leaves the other three to other jobs, so expect to share the node; on
the SSH path set `GPU=<n>` to pin a device.

The KV cache costs 160 KiB per token, from 10 full attention layers plus 50 sliding-window layers of
1024 tokens: 5.8 GB at 32K, 21 GB at 128K, and 41 GB at the full 256K. Weights are 62.6 GB in bf16 and
about half that in FP8, and this is the tightest of the three GPU types for this checkpoint. At the 0.90
utilization default an 80 GB card has roughly 72 GB to spend, so bf16 weights leave only about 9 GB of
KV cache, which is around 55K tokens; FP8 weights leave roughly 40 GB, about what the full 256K context
costs. So on this card the quantization choice decides both the rate and the usable context length.

Driver 575 and CUDA 12.9 are why this variant has its own toolchain, and why the RTX, H200, and H100
variants of this checkpoint are three separate recipes rather than one recipe with a switch.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on VAST scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes,
the interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
seconds from VAST. A bare torch and vLLM import from VAST is 9.2 seconds, so most of that 58 seconds
is engine startup rather than filesystem cost.

```
bash recipes/gemma-4-31B-it/h100-1/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. What it does, and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Ray is installed alongside vLLM. A TP1 endpoint never uses it, but it is what the pre-restructure
environment contained, so keeping it means the rates below were measured in this exact environment.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, and `env/WHEELS` records the non-PyPI artifact URLs with hashes.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/h100-1/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f gemma31-h100-1-<jobid>.log
```

Advanced path, for a node you already hold. Most users should use the Slurm submission above; this exists for reservation holders and for administrators deploying an endpoint on behalf of others, because reserved nodes are removed from the scheduler and cannot be reached with sbatch. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/gemma-4-31B-it/h100-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-31B-it/h100-1/serve_ssh.sh <node>    # pin one device of a busy node
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
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
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
source recipes/gemma-4-31B-it/h100-1/client.env
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
`http://<node>:8000/v1`, the same key, and model name `gemma-4-31b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-31B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 32768 | Context window; 262144 is supported and costs 41 GB of KV cache, which needs the FP8 default |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; one GPU holds this checkpoint |
| `QUANT` | `fp8` | Weight quantization applied at load. Set `QUANT=` (empty) for bf16, which measured 41 percent slower on this GPU and caps context near 55K |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate, and the cheapest way to buy context on this card |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, for debugging a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint. Must stay unset on vLLM 0.25.1 and 0.26.0, where it fails at the first request; see Gotchas |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, only read when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node, for example `GPU=1` |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

`QUANT` is forwarded as `${QUANT-fp8}` rather than `${QUANT:-}` by `serve_ssh.sh`, so that an unset
variable keeps the FP8 default instead of arriving on the far side as set and empty, which would
silently serve bf16 at 59 percent of the rate and with a fraction of the context.

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
| Single stream, concurrency 1 | 67.4 tok/s | 67.4 tok/s | TTFT median 45 ms, n=3 spanning 67.3 to 67.4 |
| Concurrency 8 | 491.3 tok/s | 61.4 tok/s | TTFT median 66 ms, p90 68 ms, n=3 spanning 491.2 to 491.3 |
| Concurrency 32 | 1427.9 tok/s | 44.6 tok/s | TTFT median 99 ms, p90 123 ms, n=3 spanning 1427.6 to 1428.0 |
| Concurrency 64 | 1824.9 tok/s | 28.5 tok/s | TTFT median 163 ms, p90 223 ms, n=3 spanning 1780.1 to 1845.6 |
| Concurrency 128 | 2273.0 tok/s | 17.8 tok/s | TTFT median 206 ms, p90 312 ms, n=3 spanning 2270.1 to 2289.0 |
| Concurrency 256 | 2592.9 tok/s | 10.1 tok/s | TTFT median 338 ms, p90 527 ms, n=3 spanning 2583.6 to 2597.1 |
| Concurrency 512 (highest measured) | 2680.3 tok/s | 5.2 tok/s | TTFT median 574 ms, p90 966 ms, n=3 spanning 2670.9 to 2696.2 |

Measured 2026-07-31 with `common/tools/bench.sh`. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 19 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Hardware | one H100 GPU |

Quote 67.4 tok/s for interactive coding, where one person waits on one
response. Quote 2680.3 tok/s at concurrency 512 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput was still rising at concurrency 512, the top of the sweep, so 2680.3 tok/s is a
floor and not a ceiling. The true saturation point is above what was measured, and the
sequence cap is not what stopped it: vLLM resolves `max_num_seqs` from device memory when the flag is
absent, which is 1024 on this hardware, so the sweep ran entirely below the cap. Per stream rate in the
table above shows what each added stream costs one user.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

TP1 works on an 80 GB card only because the weights are quantized: at TP1 there is no all-reduce at all,
so the node's NVLink fabric is left for jobs that need it, but bf16 weights would leave only about 9 GB
for KV cache.

FP8 weights are the default because this model is dense and therefore memory bandwidth bound. Every one
of its 31B parameters is read for every token, so halving the bytes per weight buys most of a
proportional speedup: 68.7 tok/s against 40.7 in bf16, 69 percent faster on this GPU. The same reasoning
shows up across GPU types, where the bf16 rates track HBM bandwidth: 1.77x from RTX to H100 and 1.38x
from H100 to H200, against bandwidth ratios of about 1.86x and 1.43x.

This is exactly the opposite of the 26B-A4B sibling, where FP8 measured no change at all. That model
activates only 4B of 26B parameters, so it is host overhead bound and there is almost no weight traffic
for a narrower dtype to save. Same family, same flag, opposite answer, and the reason is the
architecture rather than anything about the quantization implementation.

On this GPU type FP8 also decides how much context fits, which is not true on the 141 GB H200. Of the
roughly 72 GB available at the 0.90 default, bf16 weights leave about 9 GB, near 55K tokens, while FP8
leaves roughly 40 GB, about what the full 256K context costs. Setting `QUANT=` (empty) therefore costs
41 percent of the rate and most of the context at once, so do it only for a short-context fidelity
comparison. `KV_FP8=1` halves the 160 KiB per token that the KV cache costs at no measured cost in rate,
which makes it the cheapest way to stretch context on this card.

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

This is the model that would benefit most from a working drafter, since it is dense and bandwidth
bound, which is why the plumbing is kept rather than deleted.

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
check only that device: another job may legitimately be holding the other three.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 15 min | skipped |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold, any just-in-time kernel
compilation happens once, and FP8 quantization happens during weight load. A launch that looks hung
during this window is usually still loading. Check the log before killing it. These numbers will be
filled in when this recipe is validated on hardware; they are deliberately blank rather than guessed.
