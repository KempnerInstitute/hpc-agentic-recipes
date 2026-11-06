# Load barrier patch for the container

One file from `lmsysorg/sglang:hy4-preview`, bind-mounted over the copy inside the image. Set
`HY4_LOAD_BARRIER_PATCH=1` to apply it, which `serve.sh` does by default for this recipe.

`load_model_utils.py` carries a single local change: `UNBALANCED_MODEL_LOADING_TIMEOUT_S` goes from
480 to 3600.

## Why

After every rank finishes loading weights, SGLang calls `dist.monitored_barrier` with that constant as
its timeout, and raises if any rank has not arrived:

```
ValueError: TP rank 0 could finish the model loading, but there are other ranks that didn't finish
loading. It is likely due to unexpected failures (e.g., OOM) or a slow node.
```

The message names OOM first, which is misleading. Loading this 1452.8 GiB checkpoint across 16 ranks
from shared storage simply does not finish within 480 s of itself. Two measured attempts:

| Attempt | Fastest rank | Slowest rank that finished | Spread |
| --- | --- | --- | --- |
| 1 | 1410 s | 1986 s | 576 s |
| 2 | 1853 s | 2442 s | 589 s |

Both exceed 480 s, and in each case two of the four ranks had not finished at all when the barrier
fired. GPU memory peaked near 97 GB of 140 GB per card, so this is not memory pressure. The second
attempt ran with no other job touching the filesystem and was slower than the first, so it is not
contention either; it is the size of the checkpoint against a barrier sized for smaller ones.

There is no environment variable for the constant, so a bind mount is the only route short of building
a new image. Upstream sizes it for a model that loads in minutes; at 1452.8 GiB over a shared
filesystem the load alone takes 30 to 40 minutes per rank.

## Keeping it current

The file is a verbatim copy of the image's own, with the one constant changed. If the image is
rebuilt, re-extract and re-apply rather than carrying this copy forward:

```
singularity exec <sif> cat /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py > load_model_utils.py
```

then change `UNBALANCED_MODEL_LOADING_TIMEOUT_S` and keep the comment explaining why.
