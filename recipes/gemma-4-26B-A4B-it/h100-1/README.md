# gemma-4-26B-A4B-it on one H100 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 64, 256, 640 and 1024

Serves `gemma-4-26b` on one H100 GPU at 256K context. One GPU, no NCCL, no multi-node coordination.

| | |
| --- | --- |
| Checkpoint | `gemma-4-26B-A4B-it`, Hugging Face `google/gemma-4-26B-A4B-it` |
| Size | 51.6 GB on disk, bf16, `Gemma4ForConditionalGeneration` |
| Context | 262144, the checkpoint maximum |
| Served name | `gemma-4-26b` |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-26B-A4B-it-h100-1.key
chmod 600 secrets/gemma-4-26B-A4B-it-h100-1.key
```

- The endpoint returns 401 without it.
- `secrets/vllm_api_key` is read when this file is absent.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/gemma-4-26B-A4B-it/h100-1/env/build.sh
```

- About 13 GB, built under `ENV_ROOT` on scratch.
- Installs the cu129 wheel: these nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- Scratch expires after 90 days; rebuild with the same command.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/h100-1/serve.sbatch
squeue --me                       # NODELIST gives the host
tail -f gemma26-h100-1-<jobid>.log
```

Direct, on a node you already hold:

```
bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>    # pin one device of a shared node
```

- Startup is 3 to 7 minutes, mostly weight loading. A launch that looks hung is still loading.
- On a shared node, pass `GPU=` the device Slurm gave you: `scontrol show job -d <jobid>`, field
  `GRES=...(IDX:n)`.
- `env/env.sh` sets `VLLM_ENGINE_READY_TIMEOUT_S=3600`, above vLLM's 600 second default.
- The server log is node-local at `/tmp/$USER/vllm/`; read it over SSH on that node.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-26B-A4B-it-h100-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

- 401 without a key is correct.
- Use at least 400 output tokens. This model reasons first, returned in a separate `reasoning` field, so a
  smaller budget leaves `content` empty with `finish_reason: length`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-26B-A4B-it/h100-1/client.env
claude
```

- No output-token cap is needed at 256K.
- Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`, which sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`, without which the client reaches for a hosted Haiku.
- OpenAI-compatible clients such as Codex: base URL `http://<node>:8000/v1`, same key, model
  `gemma-4-26b`. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

- On a shared node, confirm only your own device reads 0 MiB before relaunching.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | unset, meaning bf16 | `fp8` quantizes weights on load; measured no change in rate |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Must stay unset on vLLM 0.25.1 and 0.26.0; see Known limits |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node, for example `GPU=1` |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

`MODELS_DIR` and `VLLM_CACHE_ROOT` come from `common/defaults.sh`; override there or in `common/site.conf`.

## Benchmarking

vLLM 0.25.1+cu129, `MAX_MODEL_LEN=262144`, protocol slope(128,1152), 3 repeats per level, median, warm
endpoint. The node's other three GPUs ran unrelated jobs; the serving device held 1980 MHz throughout.

| Concurrency | Aggregate | Per stream |
| --- | --- | --- |
| 1 | 214.7 tok/s | 214.7 tok/s |
| 64 | 5245.6 tok/s | 82.0 tok/s |
| 256 | 6358.4 tok/s | 24.8 tok/s |
| 640 | 7048.6 tok/s | 11.0 tok/s |
| 1024 | 7070.7 tok/s | 6.9 tok/s |

- Saturated: 0.3 percent from 640 to 1024, so 7070.7 tok/s is a ceiling, not a rising curve.
- Quote 214.7 tok/s for one person coding, 7070.7 tok/s for a shared endpoint.
- Warm first. A fresh endpoint reads 204.5 tok/s at concurrency 1; concurrency 640 is unaffected.
- KV cache holds 661,098 tokens, 2.52 full-length requests at once.
- A 240,022-token prompt was served in 14 seconds.

Reproduce:

```
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b \
  --sweep 1,64,256,640,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`: the server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- `SPEC_DRAFT` must stay unset on vLLM 0.25.1 and 0.26.0. `gemma4_mtp` fails at the first request with
  `a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]`. The fix is on vLLM `main`.
- Anthropic's hosted tools do not work: `web_search_20250305`, `web_fetch_20250910` and
  `code_execution_20250522` run on Anthropic's servers, carry no `input_schema`, and vLLM rejects all
  three with HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md) instead.
- Client-side tools work normally: file edits, shell commands, Slurm submission.
