**A storage stall can kill the endpoint even after it recovers.** PyTorch kills the process when the
NCCL watchdog thread stops sending heartbeats, on the assumption that a collective hung. A stalled
network filesystem freezes every rank the same way, so at the 480 second default a storage outage
that later recovers still takes the endpoint down permanently. Observed on 2026-07-29: a holylfs06
OSS failover froze two unrelated endpoints on two nodes within one second of each other, and both
were killed by their own watchdog eight minutes later while reporting `Last enqueued NCCL work: -1`,
meaning no collective was ever in flight. `env/env.sh` sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` so a transient stall is survivable.
