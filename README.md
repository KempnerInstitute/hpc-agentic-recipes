# Local Agentic Coding

Self-hosted open-weight LLMs for agentic coding on Kempner GPU nodes, served as OpenAI- and
Anthropic-compatible endpoints for tools like Claude Code, Cline, Aider, and OpenHands. Recipes run
on either vLLM or SGLang, whichever serves a given model best, and every endpoint is API-key gated.

## Models

| Model | Precision | Resources | Parallelism | Decode | Context |
|-------|-----------|-----------|-------------|--------|---------|
| GLM-5.2 | NVFP4 | 1 RTX PRO 6000 node (8 GPUs) | TP8 | ~90 tok/s | 128K |
| GLM-4.6 | FP8 | 1 H200 node (4 GPUs) | TP4 | ~18 tok/s | 200K |
| GLM-5.2 | FP8 | 2 H200 nodes (4 GPUs each) | TP4 x PP2 | ~13 tok/s | 1M |
| Kimi-K2.7-Code | INT4 | 1 RTX PRO 6000 node (8 GPUs) | TP8 | ~21 tok/s | 32K |
| Kimi-K2.7-Code | INT4 | 2 H200 nodes (4 GPUs each) | TP4 x PP2 | ~29 tok/s | 32K |
| Gemma-4-26B-A4B | bf16 | **1 GPU** (any type) | TP1 | 140 to 236 tok/s | 32K (256K max) |
| Gemma-4-31B | FP8 | **1 GPU** (any type) | TP1 | 40 to 85 tok/s | 32K (256K max) |
| Qwen3-235B-A22B | bf16 | 1 RTX PRO 6000 node (8 GPUs) | TP8 | ~63 tok/s | 40K |
| Qwen3-Coder-480B-A35B | FP8 | 1 RTX PRO 6000 node (8 GPUs) | TP4 x PP2 | ~64 tok/s | 128K (256K max) |

Each serves on `http://<node>:8000/v1` with model name `glm-4.6`, `glm-5.2`, `kimi-k2.7-code`,
`gemma-4-26b`, `gemma-4-31b`, `qwen3-235b`, or `qwen3-coder-480b`. Gemma-4-26B-A4B is the fastest and
by far the cheapest, needing only **one GPU**, so it is the best default for interactive work.
Qwen3-Coder-480B is the largest coding model that fits a single node. Where a range is given it
spans the GPU types; see "Measured decode rates" below for the per-GPU numbers and the winning config.
Kimi-K2.7-Code (1T-parameter MoE, native INT4, thinking-mode) is the strongest coder; it fits on one
RTX node or two H200 nodes. GLM-5.2 NVFP4 is near-FP8 quality and suits reasoning. Models supporting
longer context accept a higher `MAX_MODEL_LEN`, subject to KV memory.

## Engines

Recipes target one of two inference engines; pick whichever serves a given model best.

- **vLLM** (default): exposes both the OpenAI (`/v1`) and native Anthropic (`/v1/messages`) APIs, so
  Claude Code connects directly. It backs every endpoint in the table above.
- **SGLang**: OpenAI-compatible, used where it wins on a specific model. The included GLM-5.2 FP8
  two-node recipe (`serve_sglang_glm_ssh.sh`, served as `glm-5.2`) drives GLM-5.2's MTP/EAGLE
  speculative head, which vLLM cannot use when it spans two nodes with pipeline parallelism. It runs
  from a separate `.venv-sglang`.

> [!NOTE]
> The live endpoints currently run on vLLM. SGLang is the alternate engine: prefer a vLLM endpoint for
> Claude Code (Anthropic-native); SGLang suits OpenAI-compatible clients (Cline, Aider, Continue,
> OpenHands) or an Anthropic-to-OpenAI proxy.

## Repository layout

```
local-agentic-coding/
├── scripts/
│   ├── config.sh                 # node names + checkpoint paths (all overridable)
│   ├── setup_env.sh              # build the H200 vLLM environment (.venv, cu129)
│   ├── lib_env.sh                # H200 vLLM runtime env (sourced)
│   ├── lib_env_cu130.sh          # RTX 6000 vLLM runtime env (sourced)
│   ├── lib_env_sglang.sh         # SGLang runtime env (sourced)
│   ├── serve_glm46_ssh.sh        # GLM-4.6 FP8   (vLLM, H200, 1 node)
│   ├── serve_glm_ssh.sh          # GLM-5.2 FP8   (vLLM, H200, 2 nodes, Ray)
│   ├── serve_glm52_nvfp4_ssh.sh  # GLM-5.2 NVFP4 (vLLM, RTX 6000, 1 node)
│   ├── serve_kimi_rtx_ssh.sh     # Kimi-K2.7     (vLLM, RTX 6000, 1 node)
│   ├── serve_kimi_h200_ssh.sh    # Kimi-K2.7     (vLLM, H200, 2 nodes, Ray)
│   ├── serve_gemma4_ssh.sh       # Gemma-4-26B and 31B (vLLM, 1 GPU)
│   ├── serve_qwen3_ssh.sh        # Qwen3-235B    (vLLM, RTX 6000, 1 node)
│   ├── serve_sglang_glm_ssh.sh   # GLM-5.2 FP8   (SGLang, H200, 2 nodes)
│   ├── slurm_*.sbatch            # Slurm batch jobs (one per model / config)
│   ├── vllm_*.sh                 # per-model vLLM serve commands
│   ├── sglang_glm.sh             # GLM-5.2 SGLang serve command (MTP/EAGLE)
│   ├── ray_head.sh, ray_worker.sh# Ray cluster for the 2-node models
│   ├── chat.sh, bench.sh, smoke_test.sh  # query / throughput / health (all send the API key)
│   ├── search.sh                 # web and literature search (no API key needed)
│   ├── download_model.sh         # fetch a checkpoint into MODELS_DIR
│   └── stop_ssh.sh               # tear down
├── .claude/skills/local-search/  # skill so Claude Code uses search.sh automatically
├── clients/                      # Claude Code environment files (one per endpoint)
├── plan-new-models.md            # sizing and engine-support study for models not yet served
├── secrets/vllm_api_key          # API key, mode 600 (gitignored)
├── .venv/                        # H200 vLLM environment (cu129)
├── .venv-cu130/                  # RTX 6000 vLLM environment (cu130)
├── .venv-sglang/                 # SGLang environment (cu129)
└── cuda13/                       # CUDA 13.0 toolkit for the RTX FlashInfer JIT
```

## Hardware

Two node types. All nodes of a type are identical, so only hostnames change between reservations.

| Type | Per node | Serves |
|------|----------|--------|
| RTX PRO 6000 Blackwell | 8x 96 GB, sm_120, PCIe (no NVLink), CUDA 13 | GLM-5.2 NVFP4, Kimi-K2.7-Code, Qwen3-235B, Gemma 4 |
| H200 | 4x 141 GB, CUDA 12.9 | GLM-4.6, GLM-5.2 FP8, Kimi-K2.7-Code, Gemma 4 |
| H100 | 80 GB per GPU, CUDA 12.9 | Gemma 4 (single-GPU models only) |

> [!NOTE]
> Reserved nodes are taken out of the scheduler, so serving is started over direct SSH. Slurm
> equivalents are in the serve-script comments. Example hostnames live in `scripts/config.sh`.

## Configuration

`scripts/config.sh` holds every tunable input; the serve scripts and client envs source it. Override
any value with an environment variable at launch.

| Variable | Purpose |
|----------|---------|
| `RTX_NODE` | node for GLM-5.2 NVFP4 |
| `GLM46_NODE` | node for GLM-4.6 |
| `GLM52_HEAD`, `GLM52_WORKER` | the two nodes for GLM-5.2 FP8 |
| `KIMI_NODE` | RTX node for Kimi-K2.7-Code |
| `KIMI_HEAD`, `KIMI_WORKER` | the two H200 nodes for Kimi-K2.7-Code |
| `GEMMA4_NODE` | node for Gemma-4-26B and 31B (single GPU; set `GPU=<n>` to pin a device) |
| `QWEN3_NODE` | RTX node for Qwen3-235B (uses all 8 GPUs) |
| `CODER_NODE` | RTX node for Qwen3-Coder-480B (uses all 8 GPUs) |
| `TP`, `PP` | tensor and pipeline parallel sizes (Coder-480B needs TP4, so PP2 on 8 GPUs) |
| `QUANT` | quantization for the Gemma and Qwen scripts (`fp8`, or empty for bf16) |
| `MODELS_DIR` | base directory holding the checkpoint folders |
| `MODEL` | exact checkpoint path to serve (overrides the per-model default) |
| `API_PORT` | serve port (default 8000) |

```
# serve on a different node / from a different checkpoint copy
RTX_NODE=holygpu7c0801 MODEL=/scratch/GLM-5.2-NVFP4 bash scripts/serve_glm52_nvfp4_ssh.sh
```

## Access control

Endpoints require an API key, which vLLM and SGLang read from the `VLLM_API_KEY` environment variable.

1. Store the key in `secrets/vllm_api_key` (gitignored, mode 600).
2. `lib_env.sh`, `lib_env_cu130.sh`, and `lib_env_sglang.sh` export `VLLM_API_KEY` from that file at
   serve time.
3. `clients/*.env`, `chat.sh`, and `bench.sh` read the same file.

Requests without the key return HTTP 401. To rotate the key, write a new value to the file and
restart the servers.

## Setup

Requires `uv` (Python environment manager) and a Kempner Slurm account or reservation. Model
checkpoints are expected under `MODELS_DIR` (see Weights).

> [!NOTE]
> On Kempner, VAST scratch (`/n/netscratch`) outperforms Lustre for this workload, most noticeably
> for the Python environments, which are large sets of small files. For best throughput, build the
> `.venv*` environments and stage the checkpoints on VAST, then point `MODELS_DIR`/`MODEL` there.
> Scratch is not permanent storage, so treat it as a fast cache rather than the system of record.

### H200 environment (one-time)

```
bash scripts/setup_env.sh
mkdir -p secrets && printf '%s' '<api-key>' > secrets/vllm_api_key && chmod 600 secrets/vllm_api_key
```

### RTX 6000 environment (one-time)

The RTX PRO 6000 (sm_120, CUDA 13) needs a dedicated environment and a CUDA 13.0 toolkit for
FlashInfer's just-in-time kernel compilation.

```
# 1. Python environment with the CUDA 13 build of vLLM
uv venv .venv-cu130 --python 3.12
uv pip install --python .venv-cu130/bin/python --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  vllm

# 2. FlashInfer 0.6.15 (installed with --no-deps to leave torch untouched)
uv pip install --python .venv-cu130/bin/python --no-deps -U flashinfer-python==0.6.15

# 3. CUDA 13.0 toolkit for the FlashInfer JIT
module load Mambaforge/23.3.1-fasrc01
mamba create -y -p ./cuda13 -c nvidia \
  cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
```

> [!WARNING]
> `flashinfer-python` must be **0.6.15**, not the 0.6.13 pinned by vLLM 0.25.1. The sm_120 attention
> backend passes a `kv_scale_format` argument that 0.6.13 does not accept, which fails at the first
> inference request. `lib_env_cu130.sh` sets `FLASHINFER_DISABLE_VERSION_CHECK=1` because no matching
> 0.6.15 cubin package exists; kernels are compiled on first launch.

> [!IMPORTANT]
> `lib_env_cu130.sh` points `CUDA_HOME` at `cuda13` and adds its headers to `CPATH` for the JIT.
> Do not add the toolkit's libraries to `LD_LIBRARY_PATH`; its `libcudart` shadows the torch runtime
> and breaks import.

> [!NOTE]
> The RTX PRO 6000 has no NVLink. `NCCL_P2P_DISABLE=1` (set in `lib_env_cu130.sh`) is required for
> multi-GPU NCCL initialization.

### SGLang environment (one-time, optional)

SGLang uses its own venv so it coexists with the vLLM environments:

```
uv venv .venv-sglang --python 3.12
uv pip install --python .venv-sglang/bin/python --prerelease=allow "sglang>=0.5.10"
```

## Serving

Request a node through Slurm; the batch scripts select the partition and GPUs, then run the server
on the allocated node. Submit from the repo root:

```
sbatch scripts/slurm_glm52_nvfp4.sbatch   # GLM-5.2 NVFP4  -> kempner_rtx,  1 node,  8 GPUs
sbatch scripts/slurm_glm46.sbatch         # GLM-4.6 FP8    -> kempner_h200, 1 node,  4 GPUs
sbatch scripts/slurm_glm52_fp8.sbatch     # GLM-5.2 FP8    -> kempner_h200, 2 nodes, 4 GPUs each
sbatch scripts/slurm_kimi_rtx.sbatch      # Kimi-K2.7-Code -> kempner_rtx,  1 node,  8 GPUs
sbatch scripts/slurm_kimi_h200.sbatch     # Kimi-K2.7-Code -> kempner_h200, 2 nodes, 4 GPUs each
sbatch scripts/slurm_gemma4.sbatch        # Gemma-4-26B    -> kempner_rtx,  1 node,  1 GPU
sbatch scripts/slurm_gemma31.sbatch       # Gemma-4-31B    -> kempner_rtx,  1 node,  1 GPU (FP8)
sbatch scripts/slurm_qwen3.sbatch         # Qwen3-235B     -> kempner_rtx,  1 node,  8 GPUs
sbatch scripts/slurm_qwen3_coder.sbatch   # Qwen3-Coder-480B -> kempner_rtx, 1 node, 8 GPUs (TP4xPP2)
```

The two Gemma jobs are single-GPU and also run on H200 or H100. Switch partition and runtime env:

```
sbatch --partition=kempner_h200 --export=ALL,ENV_LIB=lib_env.sh scripts/slurm_gemma4.sbatch
sbatch --partition=kempner_h100 --export=ALL,ENV_LIB=lib_env.sh scripts/slurm_gemma31.sbatch
```

> [!NOTE]
> `kempner_rtx` allows 16 CPUs and 180 GB of host memory per GPU, which is what
> `slurm_gemma4.sbatch` requests for its single GPU.

Set your account (and optionally the checkpoint):

```
sbatch --account=<your-account> --export=ALL,MODEL=<path> scripts/slurm_glm46.sbatch
```

Find the serving host and follow startup:

```
squeue --me                              # the NODELIST column is the serving host
tail -f logs/slurm-glm46-<jobid>.log
```

The endpoint is then `http://<host>:8000/v1`. First launch takes several minutes: weight load plus
first-time JIT of the attention, FlashInfer, and Triton kernels (cached afterward).

> [!NOTE]
> Partitions are `kempner_h200` for H200 and `kempner_rtx` for RTX PRO 6000. Set `--account` to your
> own allocation. Add `--reservation=<name>` only if you hold one. The RTX job also requires the
> one-time RTX environment (see the Setup section).

> [!NOTE]
> Kimi-K2.7-Code is multimodal. Across two H200 nodes its serve scripts default to `--skip-mm-profiling`
> and `--mm-processor-cache-gb 0`, because the cross-node multimodal profiling step otherwise deadlocks.
> No action needed; the scripts set this.

### Direct SSH (only for nodes you already hold)

If you already hold the nodes (an interactive allocation, or a reservation with the nodes taken out
of the scheduler), start the server over SSH instead of sbatch:

```
bash scripts/serve_glm52_nvfp4_ssh.sh    # or serve_glm46_ssh.sh / serve_glm_ssh.sh
bash scripts/serve_kimi_rtx_ssh.sh       # or serve_kimi_h200_ssh.sh  (Kimi-K2.7-Code)
GPU=1 bash scripts/serve_gemma4_ssh.sh   # Gemma-4-26B on one GPU of a shared node
bash scripts/serve_qwen3_ssh.sh          # Qwen3-235B across all 8 GPUs of an RTX node

# Qwen3-Coder-480B reuses the same script with TP4 x PP2 and its own parsers:
QWEN3_NODE="$CODER_NODE" MODEL="$CODER_MODEL" SERVED_NAME=qwen3-coder-480b \
  TP=4 PP=2 TOOL_PARSER=qwen3_coder REASONING_PARSER= bash scripts/serve_qwen3_ssh.sh

# Gemma-4-31B (dense) reuses the same script, with FP8 and its own served name:
MODEL="$GEMMA31_MODEL" SERVED_NAME=gemma-4-31b QUANT=fp8 GPU=1 bash scripts/serve_gemma4_ssh.sh
bash scripts/serve_sglang_glm_ssh.sh     # GLM-5.2 FP8 via SGLang (alternate engine)
```

Target node and checkpoint come from `scripts/config.sh`.

## Use

Claude Code speaks the Anthropic API, so point it at a **vLLM** endpoint. The client env reads the
node from `config.sh`; for a Slurm-allocated host, set the matching node variable first (`RTX_NODE`,
`GLM46_NODE`, `GLM52_HEAD`, or `KIMI_NODE`):

```
RTX_NODE=<host> source clients/claude-code-glm52-nvfp4.env
# others: claude-code-kimi.env, claude-code-gemma4.env, claude-code-gemma31.env,
#         claude-code-qwen3.env, claude-code-qwen3-coder.env
claude
```

> [!IMPORTANT]
> The client envs export `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`. vLLM accepts only the
> `Authorization: Bearer` header, whereas `ANTHROPIC_API_KEY` makes Claude Code send `x-api-key`,
> which is rejected with HTTP 401. `ANTHROPIC_SMALL_FAST_MODEL` is set for the same reason it matters
> elsewhere: without it Claude Code tries to reach a Haiku model that these servers do not host.

OpenAI-compatible tools (Cline, Aider, Continue, OpenHands) work with either engine: base URL
`http://<host>:8000/v1`, API key from `secrets/vllm_api_key`, model `glm-4.6`, `glm-5.2`,
`kimi-k2.7-code`, `gemma-4-26b`, `gemma-4-31b`, `qwen3-235b`, or `qwen3-coder-480b`.

### Measured decode rates

Sustained single-stream decode, greedy, measured on this cluster. Each number is a slope: the same
request is timed at 128 and at 1152 output tokens and the rate is `(1152-128)/(t2-t1)`, which cancels
prefill and per-request overhead. Measuring a single short generation instead inflates the fixed cost
into the rate and understates decode by up to 40 percent, so prefer the slope when comparing configs.

| Model | Config | RTX PRO 6000 96 GB | H100 80 GB | H200 141 GB |
|-------|--------|-------------------:|-----------:|------------:|
| Gemma-4-26B-A4B | bf16, TP1 (best) | 140.5 | 183.9 | **236.0** |
| Gemma-4-31B | **FP8**, TP1 (best) | 40.1 | 68.7 | **85.3** |
| Gemma-4-31B | bf16, TP1 | 23.0 | 40.7 | 56.3 |
| Qwen3-235B-A22B | bf16, TP8, full node | **63.0** | n/a | n/a |
| Qwen3-Coder-480B-A35B | FP8, TP4 x PP2, full node | **63.9** | n/a | see note |

What does and does not help, all measured rather than assumed:

| Change | Gemma-4-26B-A4B (MoE, 4B active) | Gemma-4-31B (dense) | Qwen3-235B (MoE, TP8) |
|--------|----------------------------------|---------------------|-----------------------|
| FP8 weights | no change | **+52 to +74 percent** | no change |
| FP8 KV cache | no change | no change | not tested |
| Expert parallelism | n/a at TP1 | n/a | **9 percent slower** |
| n-gram speculative decoding | **halves throughput** | not tested | not tested |

The three models sit in different regimes, which is why the same flag helps one and not another:

- **Gemma-4-31B is memory bandwidth bound.** FP8 halves the bytes read per token and buys most of a
  proportional speedup, and bf16 rates track HBM bandwidth across the three GPUs (1.77x from RTX to
  H100, 1.38x from H100 to H200, against bandwidth ratios of about 1.86x and 1.43x).
- **Gemma-4-26B-A4B is host overhead bound.** It activates only 4B parameters, so there is little
  weight traffic to save: GPU utilization measured 35 to 40 percent and power draw about 210 W of a
  700 W limit. Rates therefore scale only about 1.3x per GPU tier, and quantization does nothing.
- **Qwen3-235B at TP8 is communication bound.** Decode costs about 16 ms per token where weight
  bandwidth alone implies about 3 ms. The RTX node has no NVLink and requires `NCCL_P2P_DISABLE=1`, so
  the tensor-parallel all-reduces in all 94 layers cross host memory. Expert parallelism adds
  all-to-all traffic on the same path and measured slower.

> [!WARNING]
> Qwen3-Coder-480B-FP8 **cannot run at TP8**. Its `moe_intermediate_size` is 2560 and its FP8
> quantization block is 128, so TP8 leaves 2560/8 = 320 per shard, which is not divisible by 128, and
> vLLM refuses to start. Use **TP4**, which on an 8-GPU node means adding `PP2`. This is why the model
> is served as TP4 x PP2 rather than TP8.

> [!NOTE]
> Qwen3-Coder-480B-FP8 **does not run on H200 with CUDA graphs** on vLLM 0.25.1. Four configurations
> were tried and all failed during graph capture, while the same checkpoint works on the RTX node:
>
> | H200 attempt | Outcome |
> |--------------|---------|
> | 4 GPUs, TP4, Triton MoE | crash during capture |
> | 2 nodes, TP4 x PP2, DeepGEMM | `CUDA error: an illegal memory access was encountered` |
> | 2 nodes, TP4 x PP2, Triton MoE | `cutlass_gemm_caller ... Error Internal`, then illegal memory access |
> | 4 GPUs, TP4, `--enforce-eager` | works, 22.2 tok/s |
>
> Memory is not the constraint: the two-node runs had 65 GiB of KV per GPU and a 2.2M-token cache. The
> root cause is the CUTLASS w8a8 FP8 GEMM path, which faults on Hopper for this checkpoint during
> capture. Note that forcing `VLLM_USE_DEEP_GEMM=1` on H200 reproduces the crash `lib_env.sh` disables
> it for, so leave that flag alone. The eager fallback works but costs roughly 3x, so **serve this
> model on the RTX node**, where its FP8 kernels are exercised on sm_120 and CUDA graphs capture
> cleanly.

Context length is the other thing VRAM decides. Gemma-4-31B in bf16 reached a 190K-token KV cache on
one 96 GB card, while FP8 weights freed enough room for 503K tokens, comfortably covering the model's
full 256K context at no cost in speed.

> [!NOTE]
> Google ships an official MTP drafter per Gemma 4 checkpoint (`*-it-assistant`, under 1 GB) that
> promises a large lossless speedup, and `vllm_gemma4.sh` supports it through `SPEC_DRAFT`. It does not
> work on vLLM 0.25.1 or 0.26.0: the drafter's `pre_projection` expects two backbone-width vectors
> (2 x 2816) but the engine feeds it backbone plus draft width (2816 + 1024), so startup fails with
> `a and b must have same reduction dim, but got [s, 3840] X [5632, 1024]`. Upstream `main` fixes this
> by sharing the target's embedding table with the drafter, so it needs a vLLM newer than 0.26.0.

### Web search

Claude Code's built-in WebSearch runs on Anthropic infrastructure, so a local endpoint rejects it
with `400 ... tools.0.input_schema Field required`. Use `scripts/search.sh`, which searches from the
cluster and needs no API key:

```
bash scripts/search.sh web "flow field reconstruction deep learning" 5
bash scripts/search.sh arxiv "velocity field reconstruction" 5
bash scripts/search.sh fetch https://arxiv.org/abs/2105.09450
```

Modes are `web`, `arxiv`, `crossref`, `pubmed`, `openalex`, `wiki`, and `fetch`. The last argument is
the result count (default 5). The first `web` call installs `ddgs` into `.venv-tools`.

To let Claude Code call it on its own, put the script on your PATH and install the skill:

```
ln -sf "$PWD/scripts/search.sh" ~/.local/bin/search.sh
mkdir -p ~/.claude/skills && cp -r .claude/skills/local-search ~/.claude/skills/
```

> [!NOTE]
> Web search scrapes a search engine, so it is less stable than a paid API and the shared cluster
> address can be rate-limited. The literature modes use official APIs and are reliable. Fetching a
> URL always works, because it runs on the client rather than on Anthropic's servers.

Query and benchmark (pass the target hostname from `config.sh` as `<node>`):

```
bash scripts/chat.sh "Write a bubble sort in Python" <node> 8000 glm-5.2
bash scripts/bench.sh <node> 8000 256 4 glm-5.2
```

Stop: `bash scripts/stop_ssh.sh`.

## Weights

Checkpoint folders live under `MODELS_DIR` (default
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`):

```
GLM-5.2-NVFP4/   GLM-5.2-FP8/   GLM-4.6-FP8/   Kimi-K2.7-Code/
gemma-4-26B-A4B-it/   gemma-4-31B-it/   Qwen3-235B-A22B/
Qwen3-Coder-480B-A35B-Instruct-FP8/
```

Each Gemma 4 checkpoint also has an optional speculative-decoding drafter, under 1 GB:
`gemma-4-26B-A4B-it-assistant/` and `gemma-4-31B-it-assistant/`, which `GEMMA4_DRAFT` and
`GEMMA31_DRAFT` point at. See the note under "Measured decode rates" before relying on them.

The folder names are the same across locations, so serving a copy elsewhere is just
`MODEL=<dir>/<folder> ...` or a different `MODELS_DIR`.

> [!NOTE]
> For large Hugging Face downloads, run on a compute node with `HF_HUB_DISABLE_XET=1`.
