# gemma-4-26B-A4B-it on one RTX PRO 6000 GPU

Status: Validated - vLLM 0.26.0, protocol: slope(128,1152) at concurrency 1, 64, 128, 256, 640 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-26b` |
| Checkpoint | `gemma-4-26B-A4B-it`, Hugging Face `google/gemma-4-26B-A4B-it` |
| On disk | 51.6 GB, BF16, `Gemma4ForConditionalGeneration` |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 RTX PRO 6000 Blackwell, 97887 MiB, TP1, so no all-reduce |
| Engine | vLLM 0.26.0 with torch cu130, sm_120 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-26B-A4B-it-rtx-1.key
chmod 600 secrets/gemma-4-26B-A4B-it-rtx-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/gemma-4-26B-A4B-it/rtx-1/env/build.sh
```

- About 7.8 GB for the venv, plus a conda CUDA 13 toolkit alongside it under `ENV_ROOT`.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the pip
  wheels do not provide.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/rtx-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma26-rtx-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-26B-A4B-it/rtx-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-26B-A4B-it/rtx-1/serve_ssh.sh <node>
```

- The server log is node-local at `/tmp/$USER/vllm/`.
- Startup is 2 to 4 minutes once the node has read the environment. The first launch on a node can take
  far longer, 15 minutes 38 seconds measured once, with several minutes before the first log line, so an
  empty log on a first launch is not a hang. On later launches the first line appears within about 20 s.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-26B-A4B-it-rtx-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "gemma-4-26b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- Use at least 400 output tokens. Reasoning arrives in a separate `reasoning` field, so a smaller budget
  can end with `finish_reason: length` and empty `content`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-26B-A4B-it/rtx-1/client.env
claude
```

- No output cap is needed at 262144.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code does not recognize this served
  name and assumes a 200k window. Client 2.1.223 began enforcing that assumption, where earlier versions
  disabled auto-compact for an unrecognized model. Without the variable the client compacts at about
  167,000 input tokens; with it, about 229,144.
- That leaves roughly 1,000 tokens of headroom against the 262144 window, because the client also assumes a
  32000-token output request for an unrecognized model. Raising the output request needs the context
  variable lowered to match.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients: base URL `http://<node>:8000/v1`, same key, model `gemma-4-26b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | unset, BF16 | `fp8` quantizes weights on load; measured no change in rate |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint. Must stay unset on vLLM 0.25.1 and 0.26.0; see Known limits |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, read only when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR`, `CUDA_HOME` | under `ENV_ROOT` | The conda CUDA 13 toolkit the sm_120 JIT needs |
| `FLASHINFER_VERSION` | pinned in `env/build.sh` | FlashInfer build installed |
| `NODE`, `GEMMA26_RTX_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 19 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 1 GPU, 16 cores, 180 GB |
| Sequence cap | `max_num_seqs` 1024, which equals the top sweep level |
| Preemption | none up to 128; from 256 upward the KV cache saturates and requests are preempted |
| Endpoint | idle, no other traffic |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 141.1 tok/s | 141.1 tok/s | 140.1 to 141.1 |
| 64 | 3398.7 tok/s | 53.1 tok/s | 3286.6 to 3476.8 |
| 128 | 4503.0 tok/s | 35.2 tok/s | 4497.5 to 4509.9 |
| 256 | 4318.2 tok/s | 16.9 tok/s | 4316.1 to 4321.9 |
| 640 | 5301.4 tok/s | 8.3 tok/s | 5260.7 to 5303.2 |
| 1024 | 5797.5 tok/s | 5.7 tok/s | 5786.8 to 5800.1 |

| | |
| --- | --- |
| Label | rising. The highest measured value is 5797.5 tok/s at 1024, the top of the sweep, but the curve is not monotonic: 256 measures 4.1 percent below 128 |
| What binds at the top | the KV cache, not the sequence cap. One run at 1024 produced 11,070 preemptions with KV usage at 100 percent and 878 requests waiting, so raising `max_num_seqs` above 1024 would not lift this number |
| Quote for one caller | 141.1 tok/s |
| Quote for a shared endpoint | 5797.5 tok/s at concurrency 1024 |
| KV cache | 1,013,590 tokens from 32.18 GiB, 3.87 full-length requests at once |
| Cost of the larger context | concurrency 1024 measures 5960.8 tok/s at a 32768 context against 5797.5 at 262144, so 2.7 percent. Single stream is unaffected, 141.0 against 141.1 |
| Long prompt, cold | 30,047 tokens in 2.1 s, 120,048 in 19.1 s, 240,048 in 68.3 s |
| Power | peak 521 W of a 600 W limit, never throttled across 900 samples, clocks 2332 to 2422 MHz |

Reproduce:

```
KEY_NAME=gemma-4-26B-A4B-it-rtx-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b
KEY_NAME=gemma-4-26B-A4B-it-rtx-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b \
  --sweep 1,64,256,640,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- `SPEC_DRAFT` must stay unset. `gemma4_mtp` fails at engine startup on vLLM 0.25.1 and 0.26.0, so the
  server never comes up.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
