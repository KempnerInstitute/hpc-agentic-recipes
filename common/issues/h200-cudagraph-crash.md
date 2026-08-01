**CUDA graphs and torch.compile crash on H200 for this model, so eager is the default.** Graph capture
hits an illegal memory access on vLLM 0.25.1, so `serve.sh` passes `--enforce-eager`. Set `PERF=1` to
retry the compile path after a vLLM upgrade.
