# <Model> on <hardware>

Status: Validated - <engine and version>, protocol: slope(128,1152) at concurrency <levels>

<!-- Steps and measured numbers only. Reasoning, history and anything that did not work stay out.
     Prefer tables to prose. Give an engine version, never a calendar date. Write about the recipe,
     not about yourself. -->

| | |
| --- | --- |
| Served name | `<served-model-name>` |
| Checkpoint | `<directory>`, Hugging Face `<org/repo>` |
| On disk | <size> across <n> shards, <precision> |
| Served precision | <precision and block size>, <GiB> of weights per GPU |
| Context | <max_model_len>, <how it relates to the checkpoint maximum> |
| Hardware | <n> <gpu type> <nodes or GPUs>, <MiB> each, <parallel shape and interconnect> |
| Engine | <engine and version>, <graphs or eager>, <speculative decoding or none> |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/<Checkpoint-Name>-<hardware>.key
chmod 600 secrets/<Checkpoint-Name>-<hardware>.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` on your PATH, and `mamba` where a recipe builds a
CUDA toolkit.

```
bash recipes/<Checkpoint-Name>/<hardware>/env/build.sh
```

- Size under `ENV_ROOT`, and anything the build needs beyond the venv.
- One bullet per non-obvious pin, saying what breaks without it.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/<Checkpoint-Name>/<hardware>/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f <job-name>-<jobid>.log
```

Direct, on a node you already hold:

```
bash recipes/<Checkpoint-Name>/<hardware>/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving | <range across launches, cold and warm> |
| Weight load, <n> shards across <n> ranks | <seconds> |

- Measure these. `to be measured` under a Validated status is a defect.

## 4. Verify

```
KEY=$(cat secrets/<Checkpoint-Name>-<hardware>.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"<served-name>","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "<served-name>"`, `"max_model_len": <n>` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- A thinking model needs at least 400 output tokens, or the budget goes on reasoning and `content` is empty.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/<Checkpoint-Name>/<hardware>/client.env
claude
```

- `client.env` must set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the served window. Claude Code assumes 200k
  for a name it does not recognize, so a smaller endpoint overflows and a larger one goes mostly unused.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `<served-name>`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

- Multi-node recipes must name every node, since there is no scheduler to clean up after the direct path.

## Tunable inputs

Every variable the scripts read. Extract the defaults from the scripts, never from memory.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/<Checkpoint-Name>` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | <n> | Context window |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| ... | | One row per variable, including those from `common/defaults.sh` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL <n> tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only |
| Context | `MAX_MODEL_LEN=<n>` |
| Allocation for the measurement | <GPUs, cores, memory, partition> |
| Sequence cap | `max_num_seqs` <n> |
| Preemption | <count>. Read `vllm:num_preemptions_total` from `/metrics`; the server log does not report it |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | <limit> enforced, median and peak across <n> samples |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | | | | |
| ... | | | | |

Verify every cell against the sweep log mechanically, column by column, before publishing.

| | |
| --- | --- |
| Label | peak, saturated, rising or capped, with the reason. See [docs/choosing-a-model.md](../../../docs/choosing-a-model.md) |
| Quote for one caller | <n> tok/s |
| Quote for a shared endpoint | <n> tok/s at concurrency <n> |
| KV cache | <n> tokens from <GiB> per GPU, <n> full-length requests at once |
| Long prompt | <n> tokens in <s> cold; <s> when the prefix is already cached |

Reproduce:

```
KEY_NAME=<Checkpoint-Name>-<hardware> bash common/tools/bench.sh --host <node> --model <served-name> \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the
  shared key and every request returns 401.

## Known limits

One bullet per failure a reader would otherwise walk into: the symptom and the rule, not the
investigation. Cover at least:

- Any parallelism shape the checkpoint refuses.
- Anything that must not be enabled.
- Any client-side limit.
