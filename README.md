# Local Agentic Coding

Self-hosted open-weight LLMs for agentic coding on Kempner GPU nodes — served as OpenAI- and
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

Each serves on `http://<node>:8000/v1` with model name `glm-4.6`, `glm-5.2`, or `kimi-k2.7-code`.
GLM-5.2 NVFP4 is the fastest and near-FP8 quality; prefer it for reasoning and coding. Kimi-K2.7-Code
(1T-parameter MoE, native INT4, thinking-mode) is the strongest coder; it fits on one RTX node or two
H200 nodes and supports up to 256K context (raise `MAX_MODEL_LEN`, subject to KV memory).

## Engines

Recipes target one of two inference engines; pick whichever serves a given model best.

- **vLLM** (default) — exposes both the OpenAI (`/v1`) and native Anthropic (`/v1/messages`) APIs, so
  Claude Code connects directly. It backs every endpoint in the table above.
- **SGLang** — OpenAI-compatible, used where it wins on a specific model. The included GLM-5.2 FP8
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
│   ├── serve_sglang_glm_ssh.sh   # GLM-5.2 FP8   (SGLang, H200, 2 nodes)
│   ├── slurm_*.sbatch            # Slurm batch jobs (one per model / config)
│   ├── vllm_*.sh                 # per-model vLLM serve commands
│   ├── sglang_glm.sh             # GLM-5.2 SGLang serve command (MTP/EAGLE)
│   ├── ray_head.sh, ray_worker.sh# Ray cluster for the 2-node models
│   ├── chat.sh, bench.sh, smoke_test.sh  # query / throughput / health (all send the API key)
│   └── stop_ssh.sh               # tear down
├── clients/                      # Claude Code environment files (one per endpoint)
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
| RTX PRO 6000 Blackwell | 8x 96 GB, sm_120, PCIe (no NVLink), CUDA 13 | GLM-5.2 NVFP4, Kimi-K2.7-Code |
| H200 | 4x 141 GB, CUDA 12.9 | GLM-4.6, GLM-5.2 FP8, Kimi-K2.7-Code |

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
printf '%s' '<api-key>' > secrets/vllm_api_key && chmod 600 secrets/vllm_api_key
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
```

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
bash scripts/serve_sglang_glm_ssh.sh     # GLM-5.2 FP8 via SGLang (alternate engine)
```

Target node and checkpoint come from `scripts/config.sh`.

## Use

Claude Code speaks the Anthropic API, so point it at a **vLLM** endpoint. The client env reads the
node from `config.sh`; for a Slurm-allocated host, set the matching node variable first (`RTX_NODE`,
`GLM46_NODE`, `GLM52_HEAD`, or `KIMI_NODE`):

```
RTX_NODE=<host> source clients/claude-code-glm52-nvfp4.env   # or clients/claude-code-kimi.env
claude
```

OpenAI-compatible tools (Cline, Aider, Continue, OpenHands) work with either engine: base URL
`http://<host>:8000/v1`, API key from `secrets/vllm_api_key`, model `glm-4.6`, `glm-5.2`, or
`kimi-k2.7-code`.

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
```

The folder names are the same across locations, so serving a copy elsewhere is just
`MODEL=<dir>/<folder> ...` or a different `MODELS_DIR`.

> [!NOTE]
> For large Hugging Face downloads, run on a compute node with `HF_HUB_DISABLE_XET=1`.
