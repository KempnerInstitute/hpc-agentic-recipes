# gemma-4-31B-it on one RTX PRO 6000 GPU

Status: Untested (migrated) - numbers below were measured 2026-07-27 with the pre-restructure scripts, protocol: slope(128,1152)

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
| `GEMMA31_RTX_NODE` | unset | An RTX node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Untested (migrated). The decode rates below were measured on this GPU type on 2026-07-27, with the
pre-restructure scripts whose serve flags this recipe reproduces exactly, using the slope protocol.
The recipe files themselves have not been run yet, so treat startup behavior as unverified while the
rates are trustworthy.

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
| GPU | RTX PRO 6000 Blackwell, 1 of the 8 on the node, 97887 MiB |
| Nodes | 1 |
| Parallelism | TP1 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, 180 GB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. This recipe claims one GPU and leaves the other seven to other jobs, so expect to share the
node; on the SSH path set `GPU=<n>` to pin a device.

The KV cache costs 160 KiB per token, from 10 full attention layers plus 50 sliding-window layers of
1024 tokens: 5.8 GB at 32K, 21 GB at 128K, and 41 GB at the full 256K. Weights are 62.6 GB in bf16 and
about half that in FP8, which is why the quantization choice decides how much context fits. Measured on
one 96 GB card: bf16 weights reached a 190K-token KV cache, and FP8 weights freed enough room for 503K
tokens, comfortably covering the model's full 256K context at no cost in speed.

sm_120 and CUDA 13 are why this variant has its own toolchain, and why the RTX, H200, and H100
variants of this checkpoint are three separate recipes rather than one recipe with a switch.

## Environment build

This recipe builds its own environment, shared with no other recipe: roughly 9.0 GB for the virtual
environment plus 2.9 GB for a CUDA 13.0 toolkit. Both land under `ENV_ROOT` on VAST scratch rather
than in the repo, because startup is dominated by page faulting the torch shared objects and stat-ing
tens of thousands of small package files: measured on one node, importing torch and vLLM took about
14 minutes from Lustre and 9.2 seconds from VAST.

```
bash recipes/gemma-4-31B-it/rtx-1/env/build.sh
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
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, and `env/WHEELS` records the non-PyPI artifact URLs with hashes. Both matter more here than on
a Hopper recipe: the nightly index rotates and deletes wheels, and the installed metadata reads
`vllm 0.25.1` with no local version tag, so a naive `vllm==0.25.1` silently resolves to the PyPI
CUDA 13 wheel, which is a different environment from the one that was tested.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/rtx-1/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f gemma31-rtx-1-<jobid>.log
```

Advanced path, for a node you already hold. Most users should use the Slurm submission above; this exists for reservation holders and for administrators deploying an endpoint on behalf of others, because reserved nodes are removed from the scheduler and cannot be reached with sbatch. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/gemma-4-31B-it/rtx-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-31B-it/rtx-1/serve_ssh.sh <node>    # pin one device of a busy node
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
source recipes/gemma-4-31B-it/rtx-1/client.env
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
| `MAX_MODEL_LEN` | 32768 | Context window; 262144 is supported and costs 41 GB of KV cache |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; one GPU holds this checkpoint |
| `QUANT` | `fp8` | Weight quantization applied at load. Set `QUANT=` (empty) for bf16, which measured 43 percent slower on this GPU |
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

`QUANT` is forwarded as `${QUANT-fp8}` rather than `${QUANT:-}` by `serve_ssh.sh`, so that an unset
variable keeps the FP8 default instead of arriving on the far side as set and empty, which would
silently serve bf16 at 57 percent of the rate.

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

| Configuration | Decode rate | Protocol |
| --- | --- | --- |
| TP1, FP8 weights, 32K context | 40.1 tok/s | slope(128,1152) |
| TP1, bf16 weights, 32K context | 23.0 tok/s | slope(128,1152) |

Sustained single-stream decode, greedy, measured on one RTX PRO 6000 on 2026-07-27. The slope protocol
times the same request at 128 and at 1152 output tokens and divides the difference, which cancels
prefill and per-request overhead; a single timed generation understates decode by up to 40 percent.
This is the slowest of the three GPU types for this checkpoint, which is expected: the model is memory
bandwidth bound and this card has the least of it.

What did and did not help, all measured rather than assumed:

| Change | Effect on decode rate |
| --- | --- |
| FP8 weights | 74 percent faster, and the reason it is the default |
| FP8 KV cache | no change in rate, but halves the cost of context |

To re-measure:

```
bash common/tools/bench.sh --host <node> --model gemma-4-31b
```

Both figures above are **single stream**, meaning one request at a time, which is what an interactive
coding session feels. That leaves the GPU far from saturated. To measure total throughput with
concurrent requests, and to find where it stops rising:

```
bash common/tools/bench.sh --host <node> --model gemma-4-31b --sweep 1,4,16,32
```

Aggregate throughput is several times the single stream figure, while per stream latency falls. On one
endpoint here, concurrency 8 delivered 404.6 tok/s aggregate against 90.0 tok/s single stream, with
each stream seeing 50.6 tok/s. Quote the single stream number for interactive use and the aggregate for
serving several people at once, and never compare one against the other.

Prompt length is a separate axis, and how much it costs depends entirely on the model's attention
design. The slope method cancels prefill, so a long prompt never distorts the measurement, but a larger
KV cache can slow every decode step because attention reads it on each one. How much is an empirical
question: measured on GLM-5.2-NVFP4, decode was flat from 21 to 26379 input tokens (97.0, 97.8 and 95.0
tok/s), because that model combines MLA compression, an fp8 KV cache and sparse attention that reads only
a subset of the context. A dense model with full attention and a bf16 KV cache should be expected to
degrade far more. Measure with `--prompt-tokens` rather than assuming either way.

## Parallelism and quantization

TP1 is enough: one 96 GB card holds the weights in either precision, and at TP1 there is no all-reduce
at all, which matters on this node type because it has no NVLink and tensor-parallel traffic would
otherwise cross host memory.

FP8 weights are the default because this model is dense and therefore memory bandwidth bound. Every
one of its 31B parameters is read for every token, so halving the bytes per weight buys most of a
proportional speedup: 40.1 tok/s against 23.0 in bf16, 74 percent faster on this GPU. The same
reasoning shows up across GPU types, where the bf16 rates track HBM bandwidth: 1.77x from RTX to H100
and 1.38x from H100 to H200, against bandwidth ratios of about 1.86x and 1.43x.

This is exactly the opposite of the 26B-A4B sibling, where FP8 measured no change at all. That model
activates only 4B of 26B parameters, so it is host overhead bound and there is almost no weight traffic
for a narrower dtype to save. Same family, same flag, opposite answer, and the reason is the
architecture rather than anything about the quantization implementation.

FP8 also decides how much context fits on this card, which matters more here than on the Hopper
variants. Measured on one 96 GB GPU: bf16 weights left room for a 190K-token KV cache, and FP8 weights
freed enough for 503K tokens, comfortably past the model's 256K maximum. So bf16 costs 43 percent of
the rate and caps context below the model's limit. Use `QUANT=` (empty) only when you specifically want
bf16 fidelity and can live with both.

`KV_FP8=1` halves the 160 KiB per token that the KV cache costs and measured no change in rate. It is
the right lever if you need long context in bf16; it is not a speed lever.

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
check only that device: another job may legitimately be holding the other seven.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 20 min | skipped |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold, the FlashInfer sm_120
kernels are compiled from source on the first request after a fresh environment because no matching
cubin package exists, and FP8 quantization happens during weight load. A launch that looks hung during
this window is usually still loading. Check the log before killing it. These numbers will be filled in
when this recipe is validated on hardware; they are deliberately blank rather than guessed.
