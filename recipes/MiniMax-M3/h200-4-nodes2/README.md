# MiniMax-M3 on two H200 nodes

Status: Validated - vLLM 0.32.0+cu129, protocol: slope(128,1152) at concurrency 1, 8 and 32

| | |
| --- | --- |
| Served name | `minimax-m3` |
| Checkpoint | `MiniMax-M3`, Hugging Face `MiniMaxAI/MiniMax-M3` |
| On disk | 795.5 GiB across 59 shards, `MiniMaxM3SparseForConditionalGeneration` |
| Served precision | BF16 weights, FP8 KV cache |
| Context | 131072 by default; the checkpoint declares 1048576 |
| Hardware | 2 nodes, 4 H200 GPUs each, 8 GPUs total, pure TP8 over Ray |
| Engine | vLLM 0.32.0, which registers this architecture |
| Allocation for the measurement | 2 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

A sparse-attention MoE, newer than the M2 line and a different architecture rather than a revision of
it. Upstream's default is TP8 on an 8-GPU node; the nodes here hold four GPUs each, so TP8 spans
two of them over Ray.

Two nodes are genuinely required: 795.5 GiB of weights against 561.6 GiB on a single node. Measured,
each GPU holds 99.94 GiB of weights and the engine reports only **13.49 GiB left for KV per GPU**, a
482,176 token pool. Subtracting weights from VRAM badly overestimates KV room on this model, so see
Known limits before planning for long contexts.

## 1. Create the API key

```
mkdir -p secrets
( umask 077; printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/MiniMax-M3-h200-4-nodes2.key )
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- `env/env.sh` resolves it through `common/lib/api_key.sh`, and vLLM reads `VLLM_API_KEY` from the
  environment, so the key never appears in a command line.

## 2. Build the environment

```
bash recipes/MiniMax-M3/h200-4-nodes2/env/build.sh
```

That creates a venv under `$ENV_ROOT` and is idempotent; pass `--force` to rebuild. The venv lives on
shared scratch so both nodes import the same interpreter. There is no container for this recipe.

## 3. Launch

Through Slurm:

```
sbatch --account=<your-account> recipes/MiniMax-M3/h200-4-nodes2/serve.sbatch
```

On two nodes you already hold, by ssh, head node first:

```
bash recipes/MiniMax-M3/h200-4-nodes2/serve_ssh.sh <head> <worker>
```

That reads `ib0` on each node, starts a Ray head and one worker, prints the cluster GPU count, which
must be 8, and then launches the server on the head. The head serves the endpoint. The log is
`$LOG_DIR/vllm-minimax-m3.log` on the head, default under `/tmp/$USER/vllm`. Startup is slow: the slowest rank takes 362 s to load its 99.94 GiB, then
`init engine (profile, create kv cache, warmup model)` adds 203.69 s, so allow about ten minutes.

## 4. Verify

```
KEY=$(cat secrets/MiniMax-M3-h200-4-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":800,"temperature":1.0,"top_p":0.95}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "minimax-m3"` |
| `/v1/models` without a key | `401` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |
| the chat response | a populated `reasoning` field, and `content` free of `<think>` markers |

Upstream recommends temperature 1.0, top_p 0.95 and top_k 40 for this model, which is why the sample
above does not use the temperature 0.6 other recipes here use.

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in `ps`.
That is acceptable for a one-off check on nodes you hold alone; use `curl -K <file>` with a mode 600
file otherwise.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/MiniMax-M3/h200-4-nodes2/client.env
claude
```

- `client.env` exports the base URL, the key, the served model name and
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`, matching the served context.
- Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and
  yields 401 on every request.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `minimax-m3`, and
  `wire_api = "responses"`. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, stop the server on the head and then Ray everywhere:

```
ssh <head> 'pkill -u $USER -f "vllm serve"'
for n in <head> <worker>; do ssh $n 'source <recipe>/env/env.sh; ray stop'; done
```

Leaving Ray running holds no GPU memory but does keep a cluster that a later launch will join, which
is confusing if the node set changed.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/MiniMax-M3` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port on the head node |
| `RAY_PORT` | 6379 | Ray head port |
| `BLOCK_SIZE` | 128 | **Mandatory at 128**, see Known limits |
| `MAX_MODEL_LEN` | 131072 | Context window. The checkpoint declares 1048576 |
| `GPU_UTIL` | 0.94 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, spanning both nodes |
| `KV_DTYPE` | `fp8_e4m3` | KV cache dtype |
| `SPEC_TOKENS` | 0 | Speculative tokens, off. Requires `SPEC_MODEL`, see Known limits |
| `SPEC_MODEL` | unset | EAGLE3 draft checkpoint. Required when `SPEC_TOKENS` is non-zero |
| `MTP_TOKENS` | 0 | Deprecated alias for `SPEC_TOKENS`, kept so older invocations still work |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode when `ENFORCE_EAGER` is unset |
| `VLLM_ALLREDUCE_USE_FLASHINFER` | 0 | **Leave at 0**, see Known limits |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | 0 | **Leave at 0**, see Known limits |
| `SPEC_METHOD` | `eagle3` | Speculative method if a draft is supplied |
| `ENFORCE_EAGER` | unset | Set to any value to skip CUDA graphs |
| `TOOL_PARSER` | `minimax_m3` | Tool-call parser. **Not** `minimax_m2` |
| `REASONING_PARSER` | `minimax_m3` | Reasoning parser. **Not** `minimax_m2` |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 182 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=131072`, `GPU_UTIL=0.94`, `KV_DTYPE=fp8_e4m3`, `BLOCK_SIZE=128`, no speculation |
| Allocation for the measurement | 2 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Endpoint | idle, and the benchmark client ran on the head node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 111.5 tok/s | 111.5 tok/s | 111.4 to 111.6 |
| 8 | 583.3 tok/s | 72.9 tok/s | 582.9 to 583.4 |
| 32 | 1582.2 tok/s | 49.4 tok/s | 1499.1 to 1583.2 |

| | |
| --- | --- |
| Engine facts | 99.94 GiB of weights per GPU; KV cache 13.49 GiB per GPU, a 482,176 token pool, 3.68 full-length requests at 131072 context; model load 362 s on the slowest rank; engine init 203.69 s |
| Label | rising. Aggregate throughput was still climbing at 32, so the ceiling is unmeasured |

111.5 tok/s on a single stream is the fastest figure in this repo for a checkpoint of this size, and it
is 5.4 times what the smaller MiniMax-M2.7 manages at 20.8 on one node. Per-stream decode does fall
sharply under load, from 111.5 to 49.4 between one caller and thirty-two, so the aggregate gain comes
partly from giving up single-stream latency. One of the three runs at concurrency 32 came in at 1499.1
against two near 1582, which is why the median rather than the mean is reported.

## Known limits

- **`--block-size 128` is mandatory and the recipe hard-codes it as the default.** This is upstream's
  statement, not an observation here: it calls 128 required on every platform, says the engine default
  of 16 misaligns this model's sparse attention, and reports outright failure on some backends. Do not
  lower it to save KV memory.
- **The parsers differ from the M2 line.** This model uses `minimax_m3` for both tool calls and
  reasoning. Passing `minimax_m2`, which the M2.7 recipe in this repo correctly uses, is a plausible
  copy-paste error and upstream flags the change explicitly.
- **Upstream says M3 support had not shipped in a stable vLLM release** and recommends a dedicated
  Docker image. The venv built here is 0.32.0 and its registry does list
  `MiniMaxM3SparseForConditionalGeneration`, `MiniMaxM3SparseForCausalLM` and `MiniMaxM3MTP`, so the
  code is present, but whether it is complete and correct at this version is untested. If load or
  generation fails, the image is the documented fallback.
- **Context is capped at 131072, not the declared 1048576, and the KV pool is the reason.** The engine
  reports 13.49 GiB of KV per GPU, a 482,176 token pool, which is **3.68 full-length requests** at
  131072. Eight concurrent callers at full context will not fit, and the declared 1048576 is far out of
  reach on this hardware: a single 1M-token request needs roughly twice the whole pool. The benchmark
  above used 182-token inputs, so its concurrency 32 result says nothing about long-context capacity.
  Raising `MAX_MODEL_LEN` lowers the concurrency figure proportionally.
- No speculative draft is used. Upstream uses EAGLE3 with 3 tokens and published third-party drafts,
  none of which is downloaded here, so `SPEC_TOKENS` defaults to 0. This architecture also has an
  `MiniMaxM3MTP` entry in vLLM, unexercised.
- `--kv-cache-dtype fp8_e4m3` is set by default. Upstream describes FP8 KV as optional and lossless
  in its own testing, giving about 1.5 times the KV pool. That claim is upstream's and is unverified
  here.
- **The MXFP8 checkpoint of this model is untested here**, which is why BF16 is served despite being
  twice the size. SGLang's cookbook states for a different model that its MXFP8 kernel path requires
  SM100, and vLLM 0.32.0 contains an `es_sm100_mxfp8_blockscaled_grouped_quant` kernel, but the same
  source also carries SM80-and-later Marlin and emulation paths. So MXFP8 on H200 (SM90) or RTX PRO
  6000 (SM120) is unverified rather than ruled out.
- Pure TP8 across two nodes puts tensor parallelism over the InfiniBand fabric rather than inside a
  node, unlike the TP4-per-node layouts elsewhere in this repo. The cost of that is unmeasured.
- Multimodal is not exercised. The architecture name ends in `ForConditionalGeneration` and vLLM also
  registers a `minimax_m3_vl` module, so vision inputs may be accepted; nothing here tests them.
