**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field (not `reasoning_content`). With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length` or `stop_reason` is
`max_tokens`, and `content` is empty, which looks like a broken endpoint but is not. Measured on
2026-07-29: GLM-4.6 consumed a full 400-token budget on reasoning alone and returned no answer. Use at
least 400 output tokens for a smoke test and 800 or more for a model that reasons at length. If
`content` is empty, raise the budget before suspecting the endpoint.
