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
