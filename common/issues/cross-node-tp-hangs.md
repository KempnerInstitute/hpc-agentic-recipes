**Keep tensor parallelism inside a node and use pipeline parallelism across nodes.** Pure tensor
parallelism spanning two nodes hangs at NCCL initialization. The working shape is TP within each node,
where all-reduce uses NVLink, and PP between nodes.
