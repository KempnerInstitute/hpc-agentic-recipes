# Parser patch for the container

Two files from SGLang's `kimi-k3` branch, bind-mounted over the copies inside
`sglang-kimi-k3-cu12.sif`. Set `K3_PARSER_PATCH=1` to apply them. `kimik3_format.py` is verbatim.
`reasoning_parser.py` carries one local change on top of upstream, described under Second reasoning
block below.

The image predates the commit named in `UPSTREAM_SHA`, which reworked K3 reasoning and tool-call
handling. No release of SGLang carries that work: 0.5.16 is older than it, and the `kimi-k3-cu12` tag
has not been rebuilt since. The CUDA 13 tag would carry more of it but cannot run on the H200 driver,
which caps at CUDA 12.9, so a bind mount over the CUDA 12 image is the only route on this hardware.

## What it fixes

Without it, the detector ends the reasoning channel only on the tool marker. A reply that opens the
response channel instead is never closed, so its markers reach the text a client displays.

This is a display defect, not a functional one. Measured on this endpoint before patching, an OpenAI
client still wrote a file, ran it and reported the right output, and so did Claude Code. What was wrong
is that every reply from that client arrived wrapped in `<|close|>think<|open|>response<|sep|>` and
`<|close|>message<|sep|>`, and Claude Code showed the same intermittently.

Three changes address it: the detector now also breaks on `RESPONSE_OPEN`, a new
`strip_partial_marker_suffix` trims truncated markers, and `_tools_passthrough` is set only when the
channel really is tools rather than unconditionally.

## Second reasoning block, the local change

A response can hold more than one reasoning block: the model thinks, calls a tool, then thinks again
about the result. Closing the first block latches `_reasoning_done`, and every gate that strips an
opening marker sits behind that latch, so upstream emits the second block verbatim as visible text.
It shows up only after a tool call, which is why single-shot requests look clean.

Two changes in `KimiK3Detector`:

- `parse_streaming_increment` re-enters the reasoning channel when content after a completed block
  opens a new one, and holds back a marker split across chunks instead of emitting it.
- `_find_think_open` replaces an exact string match, so a repeated separator such as
  `<|open|>think<|sep|<|sep|>` is still recognized.

Verified by running the detector inside the image with both parsers over the same streams. Upstream
leaks on four shapes, this one on none:

| Stream | Upstream | Patched |
| --- | --- | --- |
| single block | clean | clean |
| two blocks, tool shape | leaks | clean |
| second block, repeated separator | leaks | clean |
| open marker split across chunks | leaks | clean |
| three blocks | leaks | clean |
| no reasoning, or reasoning only | clean | clean |

End to end, Claude Code at maximum effort driving a `search.sh` round-trip showed the markers twice
before and none after.

## Measured, patched against unpatched

Same four nodes, same task, same key. Markers gone from the client's displayed output and from the raw
`/v1/responses` body, with nothing else moved: request cap 67, token pool 383,223, context 262,144,
decode 33.4 to 39.3 tok/s against 33.8 to 39.3 before. Claude Code still writes files, runs commands and
reports results. An eight trial two-turn replay on `/v1/messages` was clean both before and after, so
that route showed no defect for this patch to fix.

## Provenance

Pinned to the commit in `UPSTREAM_SHA`, fetched from
`raw.githubusercontent.com/sgl-project/sglang/<sha>/python/sglang/srt/...`. Both files are upstream
Apache-2.0 code, held here only so a bind mount has a source on a filesystem every node can read, and they
keep upstream's style rather than this repository's. `kimik3_format.py` is needed because the parser
imports three new helpers from it. The local change to
`reasoning_parser.py` is confined to `KimiK3Detector`; nothing shared with other models is touched.

## When to retire it

Drop this directory and the `K3_PARSER_PATCH` block in `serve.sh` once the fix ships in something the
recipe can pin directly. Two ways to check:

```bash
# has the CUDA 12 tag been rebuilt?
curl -s "https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags?name=kimi-k3-cu12" \
  | python3 -c 'import json,sys; [print(t["name"], t["last_updated"]) for t in json.load(sys.stdin)["results"]]'

# does a rebuilt image already contain the fix?
singularity exec "$SIF" grep -c strip_partial_marker_suffix \
  /sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_format.py
```

A non-zero count from the second command means the image no longer needs this.
