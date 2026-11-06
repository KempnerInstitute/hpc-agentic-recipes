# Ripple-65B-Preview-NVFP4

The NVFP4 build of Ripple-65B-Preview, a dense 65B transformer from Ricursive Intelligence. It serves
as `ripple`, the same name as the BF16 build, so only one of them can run on a node at a time.

Only the MLP is quantized. The checkpoint's own implementation calls this an
"NVFP4-MLP/BF16-attention" model, which is why it is 53.3 GiB rather than the 32.5 GiB a uniform 4-bit
build would be, and why its KV cache costs the same per token as the BF16 build.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | SGLang | 1 H200 GPU, TP1 | 36.2 tok/s | 517.9 at c=16, capped | Validated |

**Prefer this over the BF16 recipes unless you need the last 11 percent of throughput.** One GPU
serves the full 262144 context, where [BF16 on one GPU](../Ripple-65B-Preview/h200-1/README.md) caps
near 33,000 tokens and [BF16 on two](../Ripple-65B-Preview/h200-2/README.md) needs twice the hardware
to reach the same window.

| | BF16, 1 GPU | BF16, 2 GPUs | NVFP4, 1 GPU |
| --- | --- | --- | --- |
| Usable context | ~33,000 | 262,144 | 262,144 |
| Single stream | 25.1 tok/s | 40.5 tok/s | 36.2 tok/s |
| Aggregate at its cap | 188.8 at c=8 | 587.7 at c=16 | 517.9 at c=16 |
| Weights | 120.7 GiB | 120.7 GiB | 53.71 GB |

NVFP4 is Blackwell-native, and the other NVFP4 recipes here serve on RTX PRO 6000, so it is worth
stating plainly that this one runs on Hopper: SGLang's ModelOpt path accepts capability 89 and above,
and an H200 is 90. It misses the sm_120 kernel path, not correctness.

## Checkpoint provenance

- Checkpoint directory: `Ripple-65B-Preview-NVFP4`
- Hugging Face repo: `ricursiveAI/Ripple-65B-Preview-NVFP4`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Ripple-65B-Preview-NVFP4`
- Size: 53.3 GiB across 6 shards, `quant_algo` NVFP4 through ModelOpt
- Architecture: `RicursiveForCausalLM`, needs `--trust-remote-code`

The checkpoint ships its SGLang implementation under `inference/local`, where the BF16 build uses
`inference/server`. The model and parser modules are otherwise equivalent, so
`common/tools/ripple_launch.py` is shared between them and only `RIPPLE_IMPL` changes.

There is no FP8 build: `ricursiveAI` publishes the base model, the BF16 instruct, the DFlash2 draft
and this NVFP4 checkpoint, and nothing else.
