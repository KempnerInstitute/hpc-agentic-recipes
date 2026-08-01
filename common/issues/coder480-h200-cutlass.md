**This FP8 checkpoint does not run on H200 with CUDA graphs.** Four configurations were tried and all
failed at graph capture: memory utilization 0.90 and 0.96 both faulted, `VLLM_USE_DEEP_GEMM=1` gave an
illegal memory access, and the Triton path gave `cutlass_gemm_caller ... Error Internal` followed by
an illegal memory access. Memory is not the constraint; the two-node runs had 65 GiB of KV per GPU and
a 2.2M-token cache. The root cause is the CUTLASS w8a8 FP8 GEMM path faulting on Hopper for this
checkpoint during capture. Eager works at 22.2 tok/s but costs roughly 3x, so serve this model on an
RTX node, where its FP8 kernels run on sm_120 and graphs capture cleanly.
