# Adding a model

A recipe is one model on one hardware shape. Adding one means creating a directory and updating the
indexes that point at it. Everything else lives inside the directory.

## Before you write anything

**1. Find the checkpoint and decide which variant you want.** Quantized variants matter more than the
parameter count: an FP8 or NVFP4 build may fit hardware the BF16 build cannot. Record the Hugging Face
repo id, because the recipe README must state it.

**2. Confirm your engine can actually load it.** Do this first, and do it by asking the engine rather
than trusting a model card or a blog post:

```
python3 -c "
import json
from vllm.model_executor.models.registry import ModelRegistry
supported = set(ModelRegistry.get_supported_archs())
arch = json.load(open('<checkpoint>/config.json'))['architectures']
print(arch, [a in supported for a in arch])
"
```

A `False` here means no amount of configuration will help, and it is worth an hour of your time to learn
that before you spend a day of GPU time. Announcements of day-zero support routinely precede the code
landing in a release.

**3. Read the checkpoint's config for the numbers that constrain parallelism.** You need
`num_hidden_layers`, `moe_intermediate_size`, `max_position_embeddings`, the quantization method and
`weight_block_size` if quantized, and whether it has a vision tower. These decide the legal TP values,
the memory budget, and whether multimodal profiling will bite you.

**4. Download it if needed.** [downloading-weights.md](downloading-weights.md), from a compute node.

## Scaffold

Copy an existing recipe directory and edit it:

```
cp -r recipes/<existing-recipe> recipes/<Checkpoint-Name>/<hardware>
```

Hardware directory names are `<gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>]`. Rename the served model,
the key name, the log file and the checkpoint path throughout, and delete anything the new model does not
need.

Pick what to copy from by toolchain, not by model similarity:

| If your target is | Copy from a recipe on | Because |
| --- | --- | --- |
| One or more RTX GPUs | any `rtx-*` recipe | torch cu130, conda CUDA 13 toolkit, FlashInfer 0.6.15, no NVLink |
| One or more H200 or H100 GPUs | any `h200-*` recipe | cu129 release wheel, driver 575 constraint |
| More than one node | any `*-nodes2` recipe | Ray bring-up, InfiniBand settings, PP across nodes |
| SGLang | `Kimi-K3/h200-4-nodes4-sglang` | different launcher and flags entirely; the other SGLang recipe ships no scripts |

## Pin an environment

Each recipe builds its own environment and shares with nobody, so a torch bump in one recipe cannot break
another. Pin the versions that matter in `env/build.sh`, as a variable the tunables table documents:
`VLLM_VERSION` for the engine, and `FLASHINFER_VERSION` where the recipe needs a specific build.

A full `uv pip freeze` lock file is optional and most recipes do not carry one. It only rebuilds an
environment if every non-PyPI artifact also has its resolved URL and sha256 in `env/WHEELS`, so add the pair
or neither.

Some builds cannot be pinned at all. The sm_120 RTX wheels exist only on a nightly index that deletes old
builds within days, so a version there resolves to something else within a week. State the build the rates
were measured on in the recipe instead, and say that a rebuild will differ.

## Write the flags

`env/env.sh` holds the runtime environment. Every flag needs a provenance comment, and a review will ask
for any that is missing:

```
# verified: <symptom>, <engine version>
# inherited from <source>, untested for this model
# required: <reason>
```

This exists because the alternative is folklore. A flag copied from another recipe with a confident
one-line justification looks measured, and the next person cannot tell the difference between a setting
that prevents a crash and one that was inherited by accident.

Candidates to consider: attention backend, `VLLM_USE_DEEP_GEMM`, `NCCL_P2P_DISABLE`,
`NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`,
`VLLM_ENGINE_READY_TIMEOUT_S`, `FLASHINFER_DISABLE_VERSION_CHECK`, JIT cache directories,
`HF_HUB_OFFLINE`, `CUDAHOSTCXX`.

## Write the serve invocation

`serve.sh` holds the engine command. The decisions that actually take time:

- `--served-model-name` must match `ANTHROPIC_MODEL` in `client.env`, because a mismatch produces a
  confusing model-not-found at first use. Recipes for the same model on different
  hardware share one name on purpose, so a client does not change when the endpoint moves.
- `--tool-call-parser` and `--reasoning-parser` must be names the engine has registered. Find them by
  listing the registry rather than guessing from filenames, which do not match the registered names.
- `--enable-auto-tool-choice` is required for agentic use under vLLM. SGLang has no such flag.
- To omit a parser entirely for a non-thinking model, the variable must use `${VAR-default}` rather than
  `${VAR:-default}`, so that passing an empty value omits the flag instead of substituting the default.
  A model with no reasoning parser that gets one will have its plain output silently parsed as reasoning.
- TP must divide the model cleanly. For a block-quantized MoE, each shard of `moe_intermediate_size` must
  be a multiple of the quantization block, or the engine refuses to start.
- Keep TP inside a node and use PP across nodes. Note that PP disables speculative decoding.

## Launch, verify, then measure

Start it, confirm it answers, confirm a keyless request returns 401, and only then benchmark. Measure with
the slope method and record the protocol label:

```
bash common/tools/bench.sh --host <node> --model <served-name>
```

## Write the README and check it

Follow `common/templates/recipe-README.md`, which carries the section order every recipe here uses.

A recipe page carries the instructions and the measured numbers, and nothing else. Reasoning, history, and
anything that was tried and did not work stay out of the repo. Prefer a table to prose, and put failure
modes in Known limits as one bullet each: the symptom and the rule, not the investigation.

Two conventions the README text has to follow. Give an engine version rather than a calendar date, because a
date does not tell a reader whether a number still holds. And write about the recipe rather than about
yourself: no first person, and no citing notes that are not in the repo.

## Update the indexes

- One row in the model table in the top-level README
- A row in the model's own `README.md` if the model has more than one hardware variant
- A mention in [choosing-a-model.md](choosing-a-model.md) if it is a model someone should reach for

Coverage is checked in both directions when your pull request is reviewed, so a recipe missing from a table
and a table row pointing at a missing recipe are both caught. Adding every row above avoids a round trip.
