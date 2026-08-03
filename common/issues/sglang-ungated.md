**Anything authored for this recipe has to gate the endpoint itself.** SGLang accepts a key only as a
`--api-key` argument, which any user on the node can read out of `/proc`, so the working recipe here starts
through `common/tools/sglang_launch.py` to supply it after exec. Scripts added to this directory should do
the same rather than passing the key directly, and an endpoint with no key at all accepts any request that
reaches the port.
