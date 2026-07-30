# Web search for local models

## The problem

Claude Code's built-in web search does not work against a vLLM endpoint. The client sends a tool
definition of type `web_search_20250305` that carries no `input_schema`, and vLLM's request validation
rejects it:

```
API Error: 400 1 validation error: 'loc': ('body', 'tools', 0, 'input_schema'),
'msg': 'Field required', 'type': 'web_search_20250305'
```

The same applies to `web_fetch_20250910` and `code_execution_20250522`. These are Anthropic server-side
tools: the hosted API executes them, and no local engine implements them. Client-side tools, which is
everything the model actually drives (file edits, shell commands, and any tool you define), work
normally.

## The replacement

`common/tools/search.sh` is a keyless search tool with seven modes:

```
bash common/tools/search.sh web      "vllm pipeline parallel speculative decoding"
bash common/tools/search.sh arxiv    "mixture of experts routing"
bash common/tools/search.sh crossref "attention is all you need"
bash common/tools/search.sh pubmed   "protein language model"
bash common/tools/search.sh openalex "scaling laws"
bash common/tools/search.sh wiki     "tensor parallelism"
bash common/tools/search.sh fetch    https://example.com/page
```

Web search needs the `ddgs` package, which is installed into a small local environment on first use. The
literature modes need nothing but network access.

## Making the model use it

Install the skill so Claude Code reaches for this tool on its own, including when it hits the 400 error:

```
ln -sf "$PWD/common/tools/search.sh" ~/.local/bin/search.sh
cp -r common/skills/local-search ~/.claude/skills/
```

The skill tells the model what the tool does, when to use each mode, and specifically to fall back to it
when a hosted web search fails.

## A caveat worth knowing

Web mode scrapes a search engine, so it is the least stable of the seven: results can change format or
rate-limit without notice. The literature modes (arxiv, crossref, pubmed, openalex) call documented
APIs and are considerably more reliable. Prefer them when the question is academic.
