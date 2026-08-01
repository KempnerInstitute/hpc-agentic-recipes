# GLM-5.2-FP8 on two H200 nodes, SGLang engine

Status: Blocked - has never loaded weights; documentation only, no scripts ship in this directory

This recipe is a recorded negative result. It ships a README and nothing else: no `serve.sh`, no
`serve.sbatch`, no environment build. There is nothing to run, because nothing here has ever run. Use
`recipes/GLM-5.2-FP8/h200-4-nodes2`, the vLLM variant of the same checkpoint, which works.

## Why this variant exists at all

One capability, and it is a real one. GLM-5.2's checkpoint ships an MTP speculative head
(`num_nextn_predict_layers` is 1), and on the vLLM path that head is unusable: the model needs two H200
nodes, spanning two nodes needs pipeline parallelism, and vLLM rejects a speculative config whenever
pipeline parallelism is active. So the fastest available shape for this checkpoint on Hopper leaves a
free decode speedup on the disk.

SGLang can express the shape that would use it: TP8 across both nodes with no pipeline parallelism at
all, plus EAGLE speculative decoding driven by the checkpoint's own head. That is the entire argument
for keeping this directory. Every other axis favors vLLM, including the Anthropic-compatible endpoint
that lets Claude Code connect with no proxy, which SGLang does not serve.

Whether the shape would actually be faster is unknown, and this recipe cannot say, because it has never
produced a token. Two things would have to be true: the weights would have to load, and TP8 across two
nodes would have to initialize. The second is doubtful on its own evidence, since pure cross-node tensor
parallelism hangs at NCCL initialization on this cluster under vLLM, which is why the working recipe uses
pipeline parallelism to cross nodes. SGLang uses its own distributed initialization, so it is not
automatically subject to the same failure, but nobody has got far enough to find out.

## What was attempted

The configuration that was attempted:

| Setting | Value |
| --- | --- |
| Engine | SGLang `0.5.11.dev20260420+g3063d640d`, in its own virtual environment |
| Launcher | `python -m sglang.launch_server`, rank 0 on the head, rank 1 on the worker |
| Parallelism | `--tp 8 --nnodes 2 --node-rank <0 or 1>`, no pipeline parallelism |
| Rendezvous | `--dist-init-addr <head_ib0>:20000` |
| Context | `--context-length 131072`, `--mem-fraction-static 0.90` |
| Speculative decoding | `--speculative-algorithm EAGLE`, 5 steps, `eagle-topk` 1, 6 draft tokens |
| Parsers | `--reasoning-parser glm45 --tool-call-parser glm47` |
| Attention backend | `nsa`, with `flashmla_sparse` for prefill and `fa3` for decode |

Weight loading began at 19:36:36 and the schedulers died at 19:39:45 on the head and 19:39:55 on the
worker. All eight ranks failed the same way, TP0 through TP3 on the head and TP4 through TP7 on the
worker, each with two sigquits reported to the parent process.

## The failure

Every rank raised the same shape assertion while loading weights. Abbreviated to the frames that
matter, with the site-packages prefix shortened:

```
[2026-07-18 19:39:45 TP1] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/managers/scheduler.py", line 3773, in run_scheduler_process
    scheduler = Scheduler(
  File ".../sglang/srt/managers/scheduler.py", line 425, in __init__
    self.init_model_worker()
  File ".../sglang/srt/model_executor/model_runner.py", line 1259, in load_model
    self.model = self.loader.load_model(
  File ".../sglang/srt/model_loader/loader.py", line 708, in load_weights_and_postprocess
    model.load_weights(weights)
  File ".../sglang/srt/models/deepseek_v2.py", line 2323, in load_weights
    self.do_load_weights(weights, is_nextn)
  File ".../sglang/srt/models/deepseek_common/deepseek_weight_loader.py", line 361, in do_load_weights
    future.result()
  File ".../concurrent/futures/thread.py", line 59, in run
    result = self.fn(*self.args, **self.kwargs)
  File ".../sglang/srt/layers/linear.py", line 269, in weight_loader
    assert param.size() == loaded_weight.size()
AssertionError
```

```
[2026-07-18 19:39:45] Received sigquit from a child process. It usually means the child failed.
```

<!-- issue:sglang-weight-load-fails begin -->
**This recipe has never successfully loaded weights.** The only recorded attempt fails during weight
loading with a shape assertion inside the DeepSeek weight loader:

```
assert param.size() == loaded_weight.size()
AssertionError
```

followed by the parent receiving sigquit from a child. No decode rate has ever been measured for this
configuration. It is kept as a starting point for someone who wants to finish the work, and because
SGLang is the only engine that can use GLM-5.2's MTP head at TP8 across two nodes, which vLLM cannot
because it would need pipeline parallelism.
<!-- issue:sglang-weight-load-fails end -->

Two details in that traceback are the lead for anyone who wants to finish this work, and neither is a
diagnosis:

- **The model is being loaded as DeepSeek-V2.** The failing frames are `models/deepseek_v2.py` and
  `models/deepseek_common/deepseek_weight_loader.py`, so this SGLang build has no GLM-5.2-specific
  implementation and maps the architecture onto its DeepSeek family loader. The assertion is a
  parameter-versus-checkpoint shape mismatch inside that loader, raised in a worker thread and surfaced
  through `future.result()`, which is consistent with a checkpoint whose tensor shapes do not match what
  the DeepSeek path expects.
- **Shared-experts fusion was on.** The log records `Shared experts fusion optimization enabled.`
  immediately before the load began. That optimization rewrites expert weight shapes, so it is the first
  thing worth disabling on a retry, with `--disable-shared-experts-fusion`. This is a hypothesis from
  log adjacency, not a tested fix.

A retry would also want `--speculative-algorithm` dropped for the first attempt, so that a weight
loading failure cannot be confused with a draft-head problem, and a newer SGLang build, since the one
used was a dated dev snapshot from a git hash rather than a release.

## No measured performance

No decode rate has ever been measured for this configuration, and none is estimated here. The engine
never reached a state where it could answer a request. For reference, the vLLM variant of the same
checkpoint on the same two nodes measured 13.0 tok/s with protocol `slope(128,1152)`, and that is
the number to beat if this is ever made to work.

## Environment build

There is no `env/build.sh` here, because no environment for this recipe has ever been built to a state
worth recording. The attempt ran from a separate virtual environment holding SGLang
`0.5.11.dev20260420+g3063d640d`, kept apart from the vLLM one so both engines could coexist. Whoever
picks this up will build a fresh one and, this being Hopper, will hit the same wheel constraint every
other H200 recipe here hits:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

That text is written for the vLLM recipes and their `--torch-backend=cu129` install, but the constraint
is the driver, not the engine: driver 575 on these nodes is CUDA 12.9, so any SGLang build here needs a
torch built against CUDA 12.9 as well. Pin the SGLang version explicitly rather than taking a dated dev
snapshot, which is what the failed attempt used.

## Gotchas

Everything below applies to this configuration on this hardware, and all of it is inherited rather than
observed here, because this recipe has never run long enough to observe anything but the weight load
failure. The shared text refers to `env/env.sh` and `serve.sh`, which this directory does not ship;
read those references as what an implementation would have to set, not as what is set today.

<!-- issue:cross-node-tp-hangs begin -->
**Keep tensor parallelism inside a node and use pipeline parallelism across nodes.** Pure tensor
parallelism spanning two nodes hangs at NCCL initialization. The working shape is TP within each node,
where all-reduce uses NVLink, and PP between nodes.
<!-- issue:cross-node-tp-hangs end -->

This one deserves emphasis, because this recipe deliberately does the thing it warns against. TP8 across
two nodes is the entire point of the SGLang variant, and it is the shape that hangs at NCCL
initialization under vLLM on this cluster. SGLang brings up its own process group through
`--dist-init-addr` rather than through vLLM's path, so it is not automatically subject to the same
failure, but it is not known to be immune either: the weight load fails first, so the question has never
been reached. Assume this is the second obstacle, not a solved problem.

<!-- issue:pp-forbids-spec-decode begin -->
**Pipeline parallelism disables speculative decoding.** vLLM rejects a speculative config when
pipeline parallelism is in use, so no MTP or draft-model speedup is available in any recipe that needs
PP to span nodes, even when the checkpoint ships an MTP head. This is why an SGLang recipe exists for
GLM-5.2: SGLang can run TP8 across two nodes with EAGLE speculative decoding, where vLLM would need PP
and therefore lose it. The guard is not visible in vLLM 0.25.1's config source, so treat it as behavior
for this version rather than a documented API contract, and re-check after an engine upgrade.
<!-- issue:pp-forbids-spec-decode end -->

That constraint is the reason this directory exists. It is a vLLM constraint, and avoiding it is exactly
what a TP8 SGLang deployment would buy.

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

`VLLM_USE_DEEP_GEMM` is a vLLM environment variable and SGLang does not read it, so the flag itself is
not the point here. The finding underneath it is: a DeepGEMM-style FP8 MoE kernel path takes an illegal
memory access on this checkpoint's sparse attention on this hardware. SGLang has its own FP8 MoE runner
selection, `--moe-runner-backend`, and if a retry gets past weight loading and then faults inside a fused
MoE kernel, this is the prior to reach for rather than a fresh investigation.

<!-- issue:jit-cache-node-local begin -->
**JIT caches must be node-local.** Concurrent multi-node compiles against a shared NFS home hit stale
file handles. `env/env.sh` points `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
`/tmp/$USER`, which also means the first launch on a fresh node pays the compile cost again.
<!-- issue:jit-cache-node-local end -->

The earlier SGLang environment library already set both cache variables to `/tmp/$USER`, so a
reimplementation should keep that.

<!-- issue:lustre-watchdog begin -->
**A storage stall can kill the endpoint even after the storage recovers.** PyTorch kills the process
when the NCCL watchdog thread stops sending heartbeats, on the assumption that a collective hung. A
stalled network filesystem freezes every rank the same way, so at the 480 second default a transient
storage outage takes the endpoint down permanently rather than pausing it. The signature is every rank
reporting `Last enqueued NCCL work: -1`, meaning no collective was ever in flight, so the process was
frozen rather than genuinely hung on communication. `env/env.sh` sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` so a stall that resolves within an hour is survivable.
<!-- issue:lustre-watchdog end -->

<!-- issue:node-local-logs begin -->
**Logs are written to node-local `/tmp`, not to the repo.** Every rank writes stderr for the life of
the endpoint, so a log on a network filesystem puts a blocking write on the critical path. During a
filesystem stall that write hangs, which freezes the server. `LOG_DIR` defaults to
`/tmp/$USER/vllm`, so read logs over SSH on the node that runs the server.
<!-- issue:node-local-logs end -->

Both nodes write their own log on this path, which is worth knowing before debugging: the failed attempt
put rank 0 output on the head and rank 1 output on the worker, and the two files did not contain the same
information. Four of the eight ranks failed in each.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field, not `reasoning_content`. With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length`, and `content` is empty,
which looks like a broken endpoint but is not. Use at least 400 output tokens for a smoke test, and 800
or more for a model that reasons at length. If `content` is empty, raise the budget before suspecting
the endpoint.
<!-- issue:thinking-model-max-tokens end -->

Field names differ by engine. That text describes vLLM's response shape; SGLang exposes only the
OpenAI-compatible API, so a client here would read `reasoning_content` rather than a `reasoning` field.
The budget advice is the part that carries over.

## Security note

<!-- issue:sglang-ungated begin -->
**This SGLang recipe does not gate the endpoint with an API key.** Unlike the vLLM recipes, the SGLang
launcher passes no `--api-key`, so anyone who can reach the port can use it. Do not run it on a shared
node without adding key gating or restricting the port.
<!-- issue:sglang-ungated end -->

This matters even for a recipe that does not work, because an ungated launcher is easy to copy. Every vLLM recipe here reads
`secrets/vllm_api_key` and passes it to the engine through the environment, so a keyless request gets
HTTP 401. The SGLang launcher passed no `--api-key` and its environment library loaded no key file, so
had this ever started serving, the port would have been open to anyone who could reach the node. Add key
gating before running it, not after.

## If you pick this up

The work is: build an SGLang environment, get the weights to load, confirm TP8 initializes across two
nodes, add API key gating, and only then measure. Any one of the first three failing is a useful result
worth recording here in place of this text. The vLLM variant is the fallback the whole time, and it is
what the repo recommends today.
