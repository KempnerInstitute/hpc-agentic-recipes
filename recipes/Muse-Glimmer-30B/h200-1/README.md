# Muse Glimmer 30B on one H200 GPU

Status: Validated - vLLM 0.33.0+cu130, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128 and 256

| | |
| --- | --- |
| Served name | `muse-glimmer` |
| Checkpoint | `Muse-Glimmer-30B`, Hugging Face `meta-models/Muse-Glimmer-30B` |
| On disk | 55.5 GiB across 2 shards, BF16, `MuseGlimmerForConditionalGeneration` |
| Served precision | BF16, unquantized |
| Context | 131072, the checkpoint's trained window |
| Hardware | 1 H200 GPU, 143771 MiB, TP1 |
| Engine | vLLM 0.33.0+cu130, CUDA graphs on, no speculative decoding by default |
| Allocation for the measurement | 1 GPU, 16 cores and 360 GB, `kempner_h200` |

Muse Glimmer is a dense 29.6B vision-language model distilled from Muse Spark for local use: a 52-layer
text decoder at hidden size 6656 with a repeating local, local, local, global attention pattern and a
2048-token sliding window, plus a ViT-G/14 perception encoder. This recipe serves the text path.

It emits channel-scoped reasoning and XML-style ATEM tool calls rather than JSON, so it needs the
dedicated `muse_glimmer` parsers, which this recipe passes.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Muse-Glimmer-30B-h200-1.key
chmod 600 secrets/Muse-Glimmer-30B-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.

## 2. Build the environment

```
bash recipes/Muse-Glimmer-30B/h200-1/env/build.sh
```

Builds into `$ENV_ROOT/Muse-Glimmer-30B/h200-1/venv` and pins vLLM 0.33.0+cu130 with transformers 5.20.2.
Run it on a compute node: a CPU allocation is enough, and it takes a few minutes.

`env/env.sh` loads `gcc/13.2.0-fasrc01` and `cuda/13.3.1-fasrc01`, which FlashInfer needs to compile its
sampling kernel on first launch. Without them the build fails with ninja exit 127 and the engine never
starts, so do not drop the module load when adapting this recipe.

## 3. Launch

```
sbatch --account=<your-account> recipes/Muse-Glimmer-30B/h200-1/serve.sbatch
```

On a node you already hold, naming the GPU index:

```
bash recipes/Muse-Glimmer-30B/h200-1/serve_ssh.sh <node> 0
```

Startup is about 4 minutes, most of it reading 55.5 GiB and compiling the FlashInfer sampling kernel the
first time. Later launches reuse the kernel cache.

## 4. Verify

```
KEY=$(cat secrets/Muse-Glimmer-30B-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":2048,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "muse-glimmer"` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

Reasoning arrives in `message.reasoning`, not folded into `content`. Tool calls arrive in
`message.tool_calls` despite being emitted as XML rather than JSON, because the `muse_glimmer` tool
parser converts them. Do not enable guided JSON: this model does not emit JSON tool calls.

## 5. Connect a client

```
export NODE=<the node>
source recipes/Muse-Glimmer-30B/h200-1/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`, without which Claude Code assumes and enforces a 200k window and
would overrun this model's trained context. Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `vllm serve` process on the node.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Muse-Glimmer-30B` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Muse-Glimmer-30B-assistant` | DFlash assistant, used only when `SPEC_MODE=dflash` |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window, fixed at init |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; the model fits one GPU |
| `MAX_NUM_SEQS` | 256, or 32 under `SPEC_MODE=dflash` | Sequence cap |
| `SPEC_MODE` | `none` | Set to `dflash` to draft from the assistant checkpoint |
| `ENFORCE_EAGER` | unset | Set to fall back to eager instead of CUDA graphs |
| `TOOL_PARSER` | `muse_glimmer` | Tool call parser |
| `REASONING_PARSER` | `muse_glimmer` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=131072`, `GPU_UTIL=0.90`, no speculative decoding |
| Allocation for the measurement | 1 GPU, 16 cores and 360 GB, `kempner_h200` |
| Sequence cap | `max_num_seqs` 256, which equals the top sweep level |
| Endpoint | idle; the benchmark client ran on another node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 61.0 tok/s | 61.0 tok/s | 61.0 to 61.0 |
| 8 | 484.5 tok/s | 60.6 tok/s | 484.5 to 484.7 |
| 32 | 1848.4 tok/s | 57.8 tok/s | 1848.1 to 1848.5 |
| 64 | 3526.3 tok/s | 55.1 tok/s | 3525.0 to 3527.2 |
| 128 | 6240.1 tok/s | 48.8 tok/s | 6235.8 to 6242.5 |
| 256 | 8925.0 tok/s | 34.9 tok/s | 8920.2 to 8932.6 |

| | |
| --- | --- |
| Engine facts | KV cache 3,803,090 tokens, 29.02 full-length requests at once; graph capture 6 s and 0.63 GiB |
| Label | rising. The curve is still climbing at 256, the recipe's own sequence cap, so 8925.0 is a floor |

The per-stream column barely moves from concurrency 1 to 64, falling only from 61.0 to 55.1 tok/s while
aggregate rises 58-fold. A single H200 GPU serves dozens of callers here without the latency each one
sees getting materially worse.

## Known limits

- The ceiling is not measured. Throughput rises monotonically across every level and is still climbing at
  256, so the real plateau needs `MAX_NUM_SEQS` above the recipe default.
- Speculative decoding is off by default and unmeasured. The DFlash assistant,
  `meta-models/Muse-Glimmer-30B-assistant` at 4.76 GiB, is downloaded and `SPEC_MODE=dflash` wires it up
  at 15 draft tokens, which is what the head's block size of 16 leaves once slot 0 re-presents the last
  accepted token. It also drops `max_num_seqs` to 32, so it trades aggregate throughput for latency.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- The perception encoder is loaded but not exercised. This recipe is validated for text only.
- Do not add token 200007 as an end-of-sequence token. It terminates the turn at the end of the reasoning
  channel and collapses tool calling; the recipe passes `--generation-config auto` so the checkpoint's own
  stop tokens, 200001 and 200008, are used.
