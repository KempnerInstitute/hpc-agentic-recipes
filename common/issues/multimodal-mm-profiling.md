**Multimodal profiling deadlocks across nodes.** With a multimodal checkpoint on more than one node,
startup completes weight loading and then hangs at multimodal profiling with 0 percent GPU
utilization, indefinitely. Raw two-node NCCL all-reduce is healthy at that point, so it is not a
fabric problem. `serve.sh` passes `--skip-mm-profiling --mm-processor-cache-gb 0`, which is required,
not optional.
