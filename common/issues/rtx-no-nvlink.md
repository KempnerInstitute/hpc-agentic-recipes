**RTX PRO 6000 nodes have no NVLink, so peer-to-peer must be disabled.** `env/env.sh` sets
`NCCL_P2P_DISABLE=1`. Without it, NCCL initialization hangs on any multi-GPU job, with no error, and
the server never becomes ready.
