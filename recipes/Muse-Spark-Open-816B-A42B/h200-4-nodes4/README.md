# Muse Spark Open 816B-A42B on four H200 nodes

Status: Validated - vLLM 0.33.0+cu130, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256,
512, 768 and 1024

| | |
| --- | --- |
| Served name | `muse-spark` |
| Checkpoint | `Muse-Spark-Open-816B-A42B`, Hugging Face `meta-models/Muse-Spark-Open-816B-A42B` |
| On disk | 1.5 TiB across 65 shards, BF16, `MuseSparkForConditionalGeneration` |
| Served precision | BF16, unquantized |
| Context | 262144. The checkpoint declares 1048576 |
| Hardware | 4 H200 nodes, 16 GPUs, 143771 MiB each, TP4 inside a node and PP4 between over Ray |
| Engine | vLLM 0.33.0+cu130, CUDA graphs at `FULL_AND_PIECEWISE`, no speculative decoding |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

Muse Spark Open is an 816B-parameter mixture-of-experts with 42B active per token, 62 layers and 256
routed experts at top-8, on a hybrid backbone that mixes full and sliding attention. The checkpoint
carries vision and audio towers; this recipe serves the text path only.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Muse-Spark-Open-816B-A42B-h200-4-nodes4.key
chmod 600 secrets/Muse-Spark-Open-816B-A42B-h200-4-nodes4.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.

## 2. Build the environment

```
bash recipes/Muse-Spark-Open-816B-A42B/h200-4-nodes4/env/build.sh
```

Builds into `$ENV_ROOT/Muse-Spark-Open-816B-A42B/h200-4-nodes4/venv` and pins vLLM 0.33.0+cu130 with
transformers 5.20.2. The same path must be reachable from all four nodes, which it is on shared scratch.
Run it on a compute node: a CPU allocation is enough, and it takes a few minutes.

## 3. Launch

```
sbatch --account=<your-account> recipes/Muse-Spark-Open-816B-A42B/h200-4-nodes4/serve.sbatch
```

The job starts a Ray head on the first node, joins the other three, then serves from the head. The
endpoint is `http://<head>:8000/v1`, and the job log names the head node on its first line.

To bring it up on nodes you already hold:

```
bash recipes/Muse-Spark-Open-816B-A42B/h200-4-nodes4/serve_ssh.sh <node0> <node1> <node2> <node3>
```

Weight load dominates startup: reading 1.5 TiB took about 45 minutes cold, and under 10 minutes once the
page cache was warm.

## 4. Verify

```
KEY=$(cat secrets/Muse-Spark-Open-816B-A42B-h200-4-nodes4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"muse-spark","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":4096,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "muse-spark"` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

Two things to know when probing this model. It reasons at length before answering, and the `muse`
reasoning parser consumes that text: it is billed in `completion_tokens` but returned in neither
`content` nor `reasoning_content`. A small `max_tokens` therefore returns `content: null` with
`finish_reason: length` and looks like a broken endpoint. Allow several thousand output tokens. Second,
prefer a temperature above 0, for the reason under Known limits.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/Muse-Spark-Open-816B-A42B/h200-4-nodes4/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`, without which Claude Code assumes and enforces a 200k window.
Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields
401 on every request.

- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `muse-spark`, and
  `wire_api = "responses"`, which recent versions require. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

Ray and the engine exit with the job. Confirm the GPUs are released before launching again, because
workers that outlive the job hold VRAM and the next launch then fails to allocate its pool.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Muse-Spark-Open-816B-A42B` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window. Fixed at init: raising it needs a reload, not a runtime change |
| `GPU_UTIL` | 0.94 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 4 | Pipeline parallel size; 4 is the node count |
| `TOOL_PARSER` | `muse` | Tool call parser |
| `REASONING_PARSER` | `muse` | Reasoning parser |
| `ENFORCE_EAGER` | unset | Set to fall back to eager instead of CUDA graphs |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported; concurrency 64 re-measured at 5 repeats |
| Input length | ISL 151 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread |
| --- | --- | --- | --- |
| 1 | 83.0 tok/s | 83.0 tok/s | 82.9 to 83.0 |
| 8 | 382.5 tok/s | 47.8 tok/s | 382.3 to 403.8 |
| 32 | 727.2 tok/s | 22.7 tok/s | 723.9 to 730.0 |
| 64 | 1293.6 tok/s | 20.2 tok/s | 1064.0 to 1313.4, n=5 |
| 128 | 2127.9 tok/s | 16.6 tok/s | 2118.1 to 2210.3 |
| 256 | 3445.6 tok/s | 13.5 tok/s | 3398.3 to 3604.3 |
| 512 | 5711.7 tok/s | 11.2 tok/s | 5693.0 to 5775.2 |
| 768 | 7809.7 tok/s | 10.2 tok/s | 7673.7 to 7839.4 |
| 1024 | 9458.8 tok/s | 9.2 tok/s | 9235.3 to 9549.8 |

| | |
| --- | --- |
| Engine facts | KV cache 4,303,585 tokens, 16.42 full-length requests at once; graph capture 5 to 6 s and 0.49 to 0.71 GiB per GPU |
| Label | rising. The curve is still climbing at 1024, the engine's own sequence cap, so 9458.8 is a floor |

## Known limits

- The ceiling is not measured. Throughput rises monotonically across every level and is still climbing at
  1024, so the real plateau needs `max_num_seqs` above the engine default.
- Concurrency 64 is bimodal rather than noisy. Five runs gave 1308.4, 1064.0, 1293.6, 1313.4 and 1069.9:
  two clean clusters near 1070 and 1300, not a spread around one value. The median is reported; treat the
  level as a range.
- Greedy decoding can fail to terminate. At temperature 0, "What is 2+2? Answer with just the number."
  did not stop within 8192 tokens in 2 of 3 attempts on the NVFP4 sibling, and this recipe shows the same
  nondeterminism: the same prompt returned `content: null` with `finish_reason: length` at `max_tokens`
  512 but answered in 276 tokens at 4096. A temperature above 0 answers in about 35 tokens. Prefer
  sampling over greedy decoding.
- The reasoning chain is billed but not returned, in either `content` or `reasoning_content`, so output
  token counts are much larger than the visible answer.
- Decode rate at a long input is not measured. Every figure here uses a 151-token input.
- The vision and audio towers are loaded but not exercised. This recipe is validated for text only.
