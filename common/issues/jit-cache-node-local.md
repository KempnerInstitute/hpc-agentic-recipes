**JIT caches must be node-local.** Concurrent multi-node compiles against a shared NFS home hit stale
file handles. `env/env.sh` points `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
`/tmp/$USER`, which also means the first launch on a fresh node pays the compile cost again.
