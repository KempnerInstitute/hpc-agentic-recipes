**This SGLang recipe does not gate the endpoint with an API key.** Unlike the vLLM recipes, the SGLang
launcher passes no `--api-key`, so anyone who can reach the port can use it. Do not run it on a shared
node without adding key gating or restricting the port.
