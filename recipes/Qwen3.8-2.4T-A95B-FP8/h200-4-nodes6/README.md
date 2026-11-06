# Qwen3.8-2.4T-A95B-FP8 on six H200 nodes

Status: Validated - vLLM 0.27.1 in a venv, protocol: slope(128,1152) at concurrency 1, 8 and 32

| | |
| --- | --- |
| Served name | `qwen3.8-2.4t` |
| Checkpoint | `Qwen3.8-2.4T-A95B-FP8`, Hugging Face `Qwen/Qwen3.8-2.4T-A95B-FP8` |
| On disk | 2324.7 GiB across 213 shards, `Qwen3_5MoeForCausalLM` |
| Served precision | FP8 weights, FP8 KV cache |
| Context | 262144, the checkpoint maximum |
| Hardware | 6 nodes, 4 H200 GPUs each, 24 GPUs total, TP4 inside a node and PP6 between them, over Ray |
| Engine | vLLM 0.27.1, FlashAttention, TRITON FP8 MoE |
| Allocation for the measurement | 6 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

A 2.4T parameter hybrid MoE with 95B active: 92 layers, 512 experts, hidden size 8192. This is the
largest checkpoint in the repo, and at 2324.7 GiB it does not fit in fewer than six of these nodes.

**Upstream sizes this at 32 H200 GPUs. It runs on 24.** See Known limits before trusting either number.

## 1. Create the API key

```
mkdir -p secrets
( umask 077; printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3.8-2.4T-A95B-FP8-h200-4-nodes6.key )
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- `env/env.sh` resolves it through `common/lib/api_key.sh`, and vLLM reads `VLLM_API_KEY` from the
  environment, so the key never appears in a command line.

## 2. Build the environment

```
bash recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/env/build.sh
```

That creates a venv with vLLM 0.27.1 and transformers 5.14.1 under `$ENV_ROOT`, and is idempotent;
pass `--force` to rebuild. There is no container for this recipe. The venv lives on shared scratch so
all six nodes import the same interpreter.

## 3. Launch

Through Slurm:

```
sbatch --account=<your-account> recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/serve.sbatch
```

On six nodes you already hold, by ssh, head node first:

```
bash recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/serve_ssh.sh <head> <w1> <w2> <w3> <w4> <w5>
```

That reads `ib0` on each node, starts a Ray head and five workers, prints the cluster GPU count, which
must be 24, and then launches the server on the head. The head serves the endpoint.

Startup is about 26 minutes end to end: Ray forms in seconds, the 2324.7 GiB of weights load across
the six nodes, then `init engine (profile, create kv cache, warmup model)` takes 163 s of which
compilation is 40 s, and graph capture 7 to 8 s per worker. The log is
`$LOG_DIR/vllm-qwen3.8-2.4t.log` on the head, default under `/tmp/$USER/vllm`.

## 4. Verify

```
KEY=$(cat secrets/Qwen3.8-2.4T-A95B-FP8-h200-4-nodes6.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/completions \
  -d '{"model":"qwen3.8-2.4t","prompt":"The capital of France is","max_tokens":14,"temperature":0}'

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3.8-2.4t","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":800,"temperature":0.6}'
```

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in `ps`.
That is acceptable for a one-off check on nodes you hold alone; use `curl -K <file>` with a mode 600
file otherwise.

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "qwen3.8-2.4t"` |
| `/v1/models` without a key | `401` |
| raw completion | a coherent continuation, for example ` Paris. It is located in the north-central part of the country` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |
| the chat response | a populated `reasoning` field, and `content` free of `<think>` or `<\|` markers |

Check the reasoning split rather than only that an answer came back. vLLM calls the field `reasoning`,
where SGLang calls it `reasoning_content`.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/client.env
claude
```

- `client.env` exports the base URL, the key, the served model name and
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`.
- Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and
  yields 401 on every request.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `qwen3.8-2.4t`, and
  `wire_api = "responses"`, which recent versions require. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, stop the server on the head and then Ray everywhere:

```
ssh <head> 'pkill -u $USER -f "vllm serve"'
for n in <head> <w1> <w2> <w3> <w4> <w5>; do ssh $n 'source <recipe>/env/env.sh; ray stop'; done
```

Leaving Ray running holds no GPU memory but does keep a cluster that a later launch will join, which
is confusing if the node set changed.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3.8-2.4T-A95B-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port on the head node |
| `RAY_PORT` | 6379 | Ray head port |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size, inside a node. Must divide the 64 attention heads |
| `PP` | 6 | Pipeline parallel size, between nodes |
| `KV_DTYPE` | `fp8` | KV cache dtype |
| `MTP_TOKENS` | 3 | Speculative tokens. **Set to 0 whenever `PP` is above 1**, see Known limits |
| `ENFORCE_EAGER` | unset | Set to any value to skip CUDA graphs |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode when not eager |
| `LOAD_FORMAT` | `fastsafetensors` | Weight loader |
| `TOOL_PARSER` | `qwen3_coder` | Tool-call parser |
| `REASONING_PARSER` | `qwen3` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `GPU_UTIL=0.90`, `KV_DTYPE=fp8`, `MTP_TOKENS=0` |
| Allocation for the measurement | 6 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Endpoint | idle, and the benchmark client ran on the head node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 52.6 tok/s | 52.6 tok/s | 52.5 to 52.6 |
| 8 | 217.5 tok/s | 27.2 tok/s | 217.5 to 217.5 |
| 32 | 534.7 tok/s | 16.7 tok/s | 531.1 to 535.1 |

| | |
| --- | --- |
| Engine facts | KV cache 9,294,877 tokens, 35.4 full-length requests at once; 19.71 to 26.05 GiB of KV per worker, and 541 to 556 GB held per node |
| Label | rising. Aggregate throughput was still climbing at 32, and the sequence cap is far above it, so the ceiling is unmeasured |

52.6 tok/s on a single stream is remarkable for a 2.4T parameter checkpoint, and that is the point of
a 95B active MoE: only a fraction of the weights runs per token. It is not the fastest model here,
since a dense 27B reaches 170.6 tok/s on a single H200 GPU. Per-stream throughput also falls faster
than on a dense model, from 52.6 to 16.7 between one caller and thirty-two, because pipeline
parallelism serializes a request across six stages.

## Known limits

- **Speculative decoding and pipeline parallelism are mutually exclusive.** SGLang's cookbook states
  that `pp_size > 1` rules out speculative decoding in aggregated serving, and the same holds here.
  `MTP_TOKENS` still defaults to 3 for a hypothetical single-stage deployment, so **set `MTP_TOKENS=0`
  for this six-node shape**, as the measurement did. The upstream latency figures that quote MTP are
  therefore not reproducible in this configuration.
- **Upstream sizes FP8 at 32 H200 GPUs; this recipe runs on 24.** The vLLM recipe's table gives 32,
  and it also calls pipeline parallelism "a fallback rather than a default", verified on GB300 with
  TP4 x PP3 rather than on Hopper. On 24 GPUs the engine still reported a 9,294,877 token KV pool at
  the full 262144 context, so 32 is conservative rather than necessary. Four nodes genuinely cannot
  work: 16 GPUs give 2246 GiB against 2324.7 GiB of weights.
- `TP` must divide the 64 attention heads, so only 1, 2, 4, 8, 16 and 32 are legal. With 4 GPUs per
  node, TP4 with PP6 is the only layout that keeps tensor parallelism inside a node.
- vLLM logs about thirty lines of `Port N is already in use, trying port N+1`, walking up from port
  100. Those are privileged ports it cannot bind; it clears 1024 within the same second and proceeds.
  It is noise, not a failure, and `VLLM_PORT` is not set by this recipe.
- An early warning reads `Tensor parallel size (24) exceeds available GPUs (4)`. That is vLLM naming
  the 24 GPU world size loosely; it is expected for multi-node Ray and not an error.
- Throughput above concurrency 32 is unmeasured, and the curve was still climbing.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- Multimodal is not exercised, and the BF16 sibling at 4556.4 GiB does not fit this cluster at all.
