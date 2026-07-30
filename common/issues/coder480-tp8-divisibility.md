**This FP8 checkpoint cannot run at TP8.** Its `moe_intermediate_size` is 2560 and its FP8
quantization block is 128. At TP8 each shard is 2560/8 = 320, which is not a multiple of 128, and vLLM
refuses to start:

```
output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128
```

TP4 gives 640, which is a multiple of 128, so on an 8-GPU node the working shape is TP4 with PP2.
