# Hy4-preview on four H200 nodes

Status: Validated - SGLang 0.5.18 in a container, protocol: slope(128,1152) at concurrency 1, 8 and 32

| | |
| --- | --- |
| Served name | `hy4-preview` |
| Checkpoint | `Hy4-preview`, Hugging Face `tencent/Hy4-preview` |
| On disk | 1452.8 GiB across 131 shards, `HYV4ForCausalLM` |
| Served precision | BF16 weights, BF16 KV pool with an FP8 indexer cache |
| Context | 131072. The checkpoint declares 1048576 |
| Hardware | 4 nodes, 4 H200 GPUs each, 16 GPUs total, pure TP16 |
| Engine | SGLang 0.5.18, DSA attention with `flashmla_sparse` |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

A 770B Mixture-of-Experts from Tencent with 49B active per token: 78 layers, the first dense and the
other 77 with 256 routed experts plus 1 shared, top-8 routed. Hidden size 6144, 64 attention heads,
vocabulary 120832. Attention is Gated DeepSeek Sparse Attention with an IndexCache, and the residual
path uses identity Hyper-Connections. Apache 2.0.

**This recipe cannot run on the upstream container unaided.** SGLang's load barrier is hardcoded at
480 s and this checkpoint cannot meet it, so `serve.sh` bind-mounts a patched file by default. See
Known limits and [patches/README.md](patches/README.md) before changing anything.

## 1. Create the API key

```
mkdir -p secrets
( umask 077; printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Hy4-preview-h200-4-nodes4.key )
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401, and `serve.sh` refuses to start without one.
- SGLang logs the parsed `server_args`, the key among them, so `serve.sh` creates the log directory
  mode 700 and each rank log mode 600, and aborts if it cannot.

## 2. Build the environment

No SGLang release carries this model, so the engine is the published image
`lmsysorg/sglang:hy4-preview`, which bundles the model code and the DSA kernels.

```
bash recipes/Hy4-preview/h200-4-nodes4/env/build.sh
```

That verifies a staged image and reports its SGLang version, failing if `sglang` does not import
inside it. `env/env.sh` resolves it in order: an exported `SIF`, then
`$MODELS_DIR/Hy4-preview/container/sglang-hy4-preview.sif`, then the shared testbed copy.

**The layers must be unpacked on node-local disk**, not on shared storage, for the same reason as the
Kimi-K3.1-Pro image: a whiteout whose removal walks a deep tree fails on a parallel filesystem. The
pull took 12 minutes locally.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 11.6 GiB on top
of the 1452.8 GiB.

## 3. Launch

On four nodes you already hold, head node first:

```
bash recipes/Hy4-preview/h200-4-nodes4/serve_ssh.sh <head> <w1> <w2> <w3>
```

Through Slurm:

```
sbatch --account=<your-account> recipes/Hy4-preview/h200-4-nodes4/serve.sbatch
```

The first node is the head: it is read for its `ib0` address, which becomes `--dist-init-addr`, and it
serves the endpoint. `serve_ssh.sh` refuses a node whose GPUs are busy, launches each rank, then
confirms all four are alive by the pid each one records.

**Startup is about an hour**, most of it weight loading. Measured per rank: 2459, 2497, 2959 and
3151 s, then graph capture. The log is `$HY4_LOG_DIR/hy4-rank<N>.log` on each node, default under
`/tmp/$USER/hy4`.

## 4. Verify

```
KEY=$(cat secrets/Hy4-preview-h200-4-nodes4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"hy4-preview","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":900,"temperature":0.9,"top_p":1.0}'
```

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in `ps`.
That is acceptable for a one-off check on nodes you hold alone; use `curl -K <file>` with a mode 600
file otherwise.

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "hy4-preview"` |
| `/v1/models` without a key | `401` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |
| the chat response | `reasoning_content` populated, and `content` free of `<think>` or `<\|` markers |

**Probe through the chat route, not `/v1/completions`.** This model is heavily instruction-tuned, and
a raw completion of "The capital of France is" returns a fill-in-the-blank quiz stem,
`' ________.\nA. London\nB. Paris\nC.'`, which looks like a failure and is not one. The card's
sampling recommendation is `temperature=0.9, top_p=1.0`.

Reasoning defaults to high. To turn off deep chain-of-thought, pass `reasoning_effort` set to
`no_think` through `chat_template_kwargs`.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/Hy4-preview/h200-4-nodes4/client.env
claude
```

- `client.env` exports the base URL, the key, the served model name and
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`.
- Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and
  yields 401 on every request.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `hy4-preview`, and
  `wire_api = "responses"`, which recent versions require. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill `sglang_launch.py` on each of the four nodes. Kill by that name rather than by
GPU, in case anything else on the node is not yours.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Hy4-preview` | Serve a different copy of the checkpoint |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port on the head node |
| `DIST_PORT` | 29501 | Rendezvous port for `--dist-init-addr` |
| `MAX_MODEL_LEN` | 131072 | Context window. The declared 1048576 exceeds the per-rank pool |
| `MEM_FRACTION` | 0.85 | Static VRAM fraction for weights plus KV |
| `TP` | 16 | Tensor parallel size. Must be a power of two dividing the 64 heads |
| `NNODES` | 4 | Node count |
| `HY4_LOAD_BARRIER_PATCH` | 1 | Bind the patched load barrier. Setting 0 restores the 480 s upstream value and the launch will fail |
| `SPEC_MODE` | `none` | Set to `nextn` for the native MTP layer, see Known limits |
| `SPEC_STEPS`, `SPEC_TOPK`, `SPEC_DRAFT` | 3, 1, 4 | NEXTN settings, only with `SPEC_MODE=nextn` |
| `MOE_RUNNER` | `auto` | MoE runner backend. Unset passes no flag, so the engine chooses; the cookbook pins none for this model |
| `ATTN` | `auto` | Attention backend. Leave it: unset passes no flag and SGLang selects DSA with `flashmla_sparse` |
| `TOOL_PARSER`, `REASONING_PARSER` | `auto` | Parsers. `auto` is what this model wants |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=131072`, `MEM_FRACTION=0.85`, `SPEC_MODE=none` |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Endpoint | idle, and the benchmark client ran on the head node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 48.8 tok/s | 48.8 tok/s | 48.7 to 48.8 |
| 8 | 299.1 tok/s | 37.4 tok/s | 299.0 to 299.1 |
| 32 | 836.3 tok/s | 26.1 tok/s | 836.1 to 840.4 |

| | |
| --- | --- |
| Engine facts | KV cache 278,400 tokens, 2.12 full-length requests at once; `max_running_requests` 2048 and 15.48 GB free after sizing |
| Label | rising. Aggregate throughput was still climbing at 32 and the request cap is 2048, so the ceiling is unmeasured |

836.3 tok/s at concurrency 32 is the highest aggregate of any four-node recipe here, and 48.8 tok/s on
a single stream beats both Kimi recipes on the same hardware, which is what 49B active out of 770B
buys. Speculative decoding is off, so the card's own latency figures are not reproduced here.

## Known limits

- **The upstream container cannot serve this model unaided.** SGLang sets
  `UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480` in `load_model_utils.py` and calls `monitored_barrier`
  with it after every rank loads. Sixteen ranks reading 1452.8 GiB from shared storage finish 600 to
  700 s apart, so correctly loading ranks get killed and the error blames OOM:
  `TP rank 0 could finish the model loading, but there are other ranks that didn't finish loading`.
  Two attempts died this way, spreads of 576 s and 589 s, the second on an otherwise idle filesystem,
  so it is neither memory nor contention. There is no environment variable for the constant, so
  `serve.sh` bind-mounts a copy with 3600 s. `HY4_LOAD_BARRIER_PATCH=0` restores upstream behavior and
  the launch will fail. Details in [patches/README.md](patches/README.md).
- **Pipeline parallelism is rejected by the model**, along with `--enable-prefill-cp`, so TP is the
  only axis. `TP` must be a power of two dividing the 64 heads, which with 4 GPUs per node makes TP16
  across four nodes the only workable layout here: 12 GPUs would need TP12 and 20 would need TP20,
  neither legal, and 8 GPUs cannot hold 1452.8 GiB.
- **Context is served at 131072, not the declared 1048576.** The cookbook states the 1M default
  exceeds the per-rank pool on H200. The pool holds 278,400 tokens, so 2.12 full-length requests.
- Do not override the attention backend. SGLang selects DSA with `flashmla_sparse` for prefill and
  decode over an FP8 indexer cache; the cookbook says to override "only with a kernel-specific reason".
- **Speculative decoding is untested here.** The model ships a native MTP layer and the cookbook drives
  it with `--speculative-algorithm NEXTN`, which `SPEC_MODE=nextn` sets. It is off in the measurement.
  The budget rule is `prompt_tokens + max_tokens + 4 <= context length`.
- The cookbook badges these multi-node BF16 recipes "In Progress" and says long-duration CUDA graph
  soak validation is incomplete, with `--disable-cuda-graph` as the fallback if the endpoint is
  unstable. Graph capture succeeded here.
- A raw `/v1/completions` probe returns a quiz stem rather than a continuation, see Verify.
- Throughput above concurrency 32 is unmeasured, and the curve was still climbing.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- **The MXFP8 sibling cannot run on H200.** `tencent/Hy4-preview-FP8` is 757.9 GiB with
  `quant_algo: MXFP8`, and the cookbook marks it "Not supported, the MXFP8 kernel path requires
  SM100". It would also need two RTX nodes rather than one, since 757.9 GiB against 764.7 GiB per node
  leaves nothing for a KV cache.
- Multimodal is not exercised.
