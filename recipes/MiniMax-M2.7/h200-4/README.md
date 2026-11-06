# MiniMax-M2.7 on one H200 node

Status: Validated - vLLM 0.32.0+cu129, protocol: slope(128,1152) at concurrency 1, 8 and 32

| | |
| --- | --- |
| Served name | `minimax-m2.7` |
| Checkpoint | `MiniMax-M2.7`, Hugging Face `MiniMaxAI/MiniMax-M2.7` |
| On disk | 214.3 GiB across 125 shards, `MiniMaxM2ForCausalLM` |
| Served precision | FP8 weights |
| Context | 196608 as served. The checkpoint declares 204800 |
| Hardware | 1 node, 4 H200 GPUs, TP4 |
| Engine | vLLM 0.32.0, which registers this architecture |
| Allocation for the measurement | 1 node, 4 GPUs, 64 cores and 1440 GB, `kempner_h200` |

An agentic-coding MoE. Upstream's default configuration is exactly TP4 on four H200 GPUs, which is
one node here, so this recipe needs no multi-node scaffolding and no container.

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

The log is `$LOG_DIR/vllm-minimax-m2.7.log`, default under `/tmp/$USER/vllm`. Startup is quick for a checkpoint this size: the 214.3 GiB loads in about four minutes, then
`init engine (profile, create kv cache, warmup model)` takes 8.87 s.

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
| `MAX_MODEL_LEN` | 196608 | Context window as served. The checkpoint declares 204800 |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size. **Do not set 8**, see Known limits |
| `PERF` | unset | Set to any value to enable CUDA graphs instead of eager |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode when `PERF` is set |
| `SPEC_CONFIG` | unset | Full speculative-config JSON, off by default |
| `TOOL_PARSER` | `minimax_m2` | Tool-call parser |
| `REASONING_PARSER` | `minimax_m2` | Reasoning parser |
| `VLLM_USE_RUST_FRONTEND` | 1 | Upstream sets this for the M2 line |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 44 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=196608`, `GPU_UTIL=0.90`, eager, no speculation |
| Allocation for the measurement | 1 node, 4 GPUs, 64 cores and 1440 GB, `kempner_h200` |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 20.8 tok/s | 20.8 tok/s | 20.8 to 20.9 |
| 8 | 165.7 tok/s | 20.7 tok/s | 164.2 to 168.7 |
| 32 | 641.5 tok/s | 20.0 tok/s | 641.2 to 642.9 |

| | |
| --- | --- |
| Engine facts | 137.86 of 139.79 GiB free per GPU at startup, `GPU_UTIL=0.90` giving 125.81 GiB; 130483 to 130485 MiB held per GPU while serving; engine init 8.87 s |
| Label | rising. Aggregate throughput was still climbing at 32 and per-stream had barely moved, so the ceiling is unmeasured |

Scaling is close to linear: 32 times the concurrency returns 30.8 times the throughput, and per-stream
decode falls only from 20.8 to 20.0 tok/s between one caller and thirty-two. That is what a sparse MoE
with few active parameters should do, and no turnover was observed through concurrency 32. The flip side
is that 20.8 tok/s on a single stream is modest, close to GLM-4.6-FP8 at 19.1 on the same hardware, so
this recipe earns its place on aggregate throughput rather than on latency.

## Known limits

- **Pure TP8 is not supported by this architecture.** Upstream states it explicitly: above four GPUs
  the expert tensors do not shard evenly, so TP8 requires `--enable-expert-parallel`, and DP8 with
  expert parallelism is the other documented option. TP4 on one node sidesteps this entirely, which
  is why it is the default here.
- **Older vLLM releases produce corrupted output for this model**, per upstream, fixed in nightlies
  after a specific commit. **No corruption was observed in the checks run here** at 0.32.0: a long
  code generation returned a correct iterative Fibonacci function with a real docstring, and
  `The capital of France is` continued with ` Paris.` That is two generations, not a systematic
  accuracy evaluation, so it is evidence the fix is present rather than proof. The symptom is garbled
  text rather than a failure to start, so read a long generation after any version change.
- **`minimax_m2` is the reasoning parser that worked here.** The vLLM recipe page gives `minimax_m2`
  and the checkpoint's own deployment guide gives `minimax_m2_append_think`. For the requests tested,
  `minimax_m2` populated `reasoning` and left `content` free of `<think>` markers. If a different
  prompt shape leaks markers, `REASONING_PARSER=minimax_m2_append_think` is the alternative to try.
- **The model emits U+2011 non-breaking hyphens inside code.** Generated text contains `F(n‑1)`,
  `non‑negative` and `n‑th` with a non-ASCII hyphen. Harmless in prose, but an identifier or a shell
  command copied straight out of a response can carry an invisible character that fails to parse.
- The server logged `No available shared memory broadcast block found in 60 seconds` twice while
  idle between benchmark levels. It served every request correctly before, during and after, and all
  nine measurement runs completed, so it is noise here rather than a fault, but it is unexplained.
- Eager is the default here because CUDA graphs have changed output correctness on some recipes in
  this repo. `PERF=1` opts in and is unmeasured. Note that the MiniMax-M3 recipe does serve with
  graphs, so this is not a repo-wide rule.
- No speculative draft is used. vLLM registers `Eagle3MiniMaxM2ForCausalLM`, so a draft is possible,
  but none is downloaded and none is measured.
- A third-party guide claims this model serves on 2 H200 GPUs. That contradicts upstream's TP4
  default and is untested here; 214.3 GiB of weights would leave very little KV room on 280.8 GiB.
- KV cache dtype is left at the engine default. Upstream specifies none for this model.
