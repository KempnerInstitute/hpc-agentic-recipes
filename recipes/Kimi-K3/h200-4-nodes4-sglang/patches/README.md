# Parser patch for the container

Two files taken unmodified from SGLang's `kimi-k3` branch and bind-mounted over the copies inside
`sglang-kimi-k3-cu12.sif`. Set `K3_PARSER_PATCH=1` to apply them.

The container was built on 2026-07-27. On 2026-08-01 the branch reworked the K3 reasoning and
tool-call path in commit `e2cf21b9`, "[Kimi K3] Add reasoning, tool-call, and OpenAI serving
support (#33025)". No release of SGLang carries it: 0.5.16 predates it, and the `kimi-k3-cu12` tag
has not been rebuilt since. The CUDA 13 tag cannot run on the H200 driver, which caps at CUDA 12.9.

## What it fixes

Without it, the detector ends the reasoning channel only on the tool marker. A reply that opens the
response channel instead is never closed, so its markers reach the text a client displays.

This is a display defect, not a functional one. Measured on this endpoint before patching, Codex still
wrote a file, ran it and reported the right output, and Claude Code did too. What was wrong is that
every Codex reply arrived wrapped in `<|close|>think<|open|>response<|sep|>` and
`<|close|>message<|sep|>`, and Claude Code showed the same intermittently.

Three changes address it: the detector now also breaks on `RESPONSE_OPEN`, a new
`strip_partial_marker_suffix` trims truncated markers, and `_tools_passthrough` is set only when the
channel really is tools rather than unconditionally.

## Measured, patched against unpatched

Same four nodes, same task, same key. Markers gone from Codex output and from the raw
`/v1/responses` body, with nothing else moved: request cap 67, token pool 383,223, context 262,144,
decode 33.4 to 39.3 tok/s against 33.8 to 39.3 before. Claude Code still writes files, runs commands
and reports results. An 8 trial two-turn replay on `/v1/messages` was clean both before and after, so
that route showed no defect to fix here.

## Provenance

Pinned to the commit in `UPSTREAM_SHA`, fetched from
`raw.githubusercontent.com/sgl-project/sglang/<sha>/python/sglang/srt/...`. Both files are upstream
Apache-2.0 code, held here only so a bind mount has a source on a filesystem every node can read.
`kimik3_format.py` is needed because the parser imports three new helpers from it.

Retire this directory once a `kimi-k3-cu12` image dated after 2026-08-01 exists, or once the fix
reaches a numbered SGLang release. Check with
`curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags?name=kimi-k3-cu12"`.
