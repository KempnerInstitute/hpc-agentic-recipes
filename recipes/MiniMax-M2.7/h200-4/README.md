# MiniMax-M2.7 on one H200 node

Status: Untested - scripts written against upstream guidance, no bring-up has run on this cluster

| | |
| --- | --- |
| Served name | `minimax-m2.7` |
| Checkpoint | `MiniMax-M2.7`, Hugging Face `MiniMaxAI/MiniMax-M2.7` |
| On disk | 214.3 GiB across 125 shards, `MiniMaxM2ForCausalLM` |
| Served precision | FP8 weights |
| Context | 196608, the checkpoint maximum |
| Hardware | 1 node, 4 H200 GPUs, TP4 |
| Engine | vLLM 0.32.0, which registers this architecture |
| Allocation for the measurement | Not yet allocated. The intended request is 1 node, 4 GPUs, 64 cores and 1440 GB, `kempner_h200` |

An agentic-coding MoE. Upstream's default configuration is exactly TP4 on four H200 GPUs, which is
one node here, so this recipe needs no multi-node scaffolding and no container.

**No rate has been measured.** Everything below the launch section is upstream guidance or arithmetic,
not an observation from this cluster. The benchmarking section stays empty until the model has run
here, and this recipe is deliberately absent from the top-level tables for that reason.

## 1. Create the API key

```
mkdir -p secrets
( umask 077; printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/MiniMax-M2.7-h200-4.key )
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- `env/env.sh` resolves it through `common/lib/api_key.sh`, and vLLM reads `VLLM_API_KEY` from the
  environment, so the key never appears in a command line.

## 2. Build the environment

```
bash recipes/MiniMax-M2.7/h200-4/env/build.sh
```

That creates a venv under `$ENV_ROOT` and is idempotent; pass `--force` to rebuild. There is no
container for this recipe.

## 3. Launch

Through Slurm:

```
sbatch --account=<your-account> recipes/MiniMax-M2.7/h200-4/serve.sbatch
```

On a node you already hold, by ssh:

```
bash recipes/MiniMax-M2.7/h200-4/serve_ssh.sh <node>
```

The log is `$LOG_DIR/vllm-minimax-m2.7.log`, default under `/tmp/$USER/vllm`. Startup time on this
cluster is unmeasured.

## 4. Verify

```
KEY=$(cat secrets/MiniMax-M2.7-h200-4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"minimax-m2.7","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":800,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "minimax-m2.7"` |
| `/v1/models` without a key | `401` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |
| the chat response | a populated `reasoning` field, and `content` free of `<think>` markers |

Check the reasoning split, not only that an answer came back. **Read the first long generation in
full.** Upstream reports corrupted output on older stable releases, fixed only after a specific
commit, so garbled text here is a known failure mode rather than a sampling artifact. See Known
limits.

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in `ps`.
That is acceptable for a one-off check on a node you hold alone; use `curl -K <file>` with a mode 600
file otherwise.

## 5. Connect a client

```
export NODE=<the node>
source recipes/MiniMax-M2.7/h200-4/client.env
claude
```

- `client.env` exports the base URL, the key, the served model name and
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=196608`.
- Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and
  yields 401 on every request.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `minimax-m2.7`, and
  `wire_api = "responses"`. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

For an ssh launch:

```
ssh <node> 'pkill -u $USER -f "vllm serve"'
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/MiniMax-M2.7` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 196608 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size. **Do not set 8**, see Known limits |
| `PERF` | unset | Set to any value to enable CUDA graphs instead of eager |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode when `PERF` is set |
| `SPEC_CONFIG` | unset | Full speculative-config JSON, off by default |
| `TOOL_PARSER` | `minimax_m2` | Tool-call parser |
| `REASONING_PARSER` | `minimax_m2` | Reasoning parser |
| `VLLM_USE_RUST_FRONTEND` | 1 | Upstream sets this for the M2 line |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Not measured. No node has been assigned for this recipe, so there is no rate, no concurrency curve
and no saturation label. Running the standard protocol here would mean slope(128,1152) with 3 repeats
per level and the median reported, as every other recipe in this repo does.

| | |
| --- | --- |
| Label | none. No rate is measured, so this recipe contributes no label and is absent from the counts in [docs/benchmarking.md](../../../docs/benchmarking.md) |

## Known limits

- **Pure TP8 is not supported by this architecture.** Upstream states it explicitly: above four GPUs
  the expert tensors do not shard evenly, so TP8 requires `--enable-expert-parallel`, and DP8 with
  expert parallelism is the other documented option. TP4 on one node sidesteps this entirely, which
  is why it is the default here.
- **Older vLLM releases produce corrupted output for this model.** Upstream reports the fix landing
  in nightlies after a specific commit, and pins a different commit for reproducing its accuracy
  numbers. The venv here is 0.32.0, far newer than the 0.20.0 the recipe page badges, so it very
  likely contains the fix, but that is an inference and not a test. Read the first long generation
  before trusting the endpoint.
- **The two upstream sources disagree on the reasoning parser.** The vLLM recipe page gives
  `minimax_m2`, which this recipe uses; the checkpoint's own deployment guide gives
  `minimax_m2_append_think`. If the `reasoning` field comes back empty or `content` carries `<think>`
  markers, set `REASONING_PARSER=minimax_m2_append_think` and compare.
- Eager is the default, as everywhere else in this repo, because CUDA graphs have changed output
  correctness on some recipes here. `PERF=1` opts in and is unmeasured.
- No speculative draft is used. vLLM registers `Eagle3MiniMaxM2ForCausalLM`, so a draft is possible,
  but none is downloaded and none is measured.
- A third-party guide claims this model serves on 2 H200 GPUs. That contradicts upstream's TP4
  default and is untested here; 214.3 GiB of weights would leave very little KV room on 280.8 GiB.
- KV cache dtype is left at the engine default. Upstream specifies none for this model.
