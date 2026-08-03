#!/usr/bin/env python3
"""Offline tests for verify_checkpoint.py. No GPU, no network, no real weights.

Run: python3 repo-checks/test_verify_checkpoint.py

Builds small but genuine safetensors files, then breaks them one way at a time. The case that matters most
is the truncated shard: counting files reports it as present, which is why this checks the length each file
declares in its own header.
"""
import importlib.util
import json
import pathlib
import shutil
import struct
import sys
import tempfile

TOOLS = pathlib.Path(__file__).resolve().parents[1] / "common" / "tools"

spec = importlib.util.spec_from_file_location(
    "verify_checkpoint", TOOLS / "verify_checkpoint.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAILURES = []


def check(name, got, want):
    if got == want:
        print("  ok    %s" % name)
        return
    FAILURES.append("%s: got %r, want %r" % (name, got, want))
    print("  FAIL  %s: got %r, want %r" % (name, got, want))


def shard(path, nbytes=16):
    header = {"w": {"dtype": "F32", "shape": [nbytes // 4], "data_offsets": [0, nbytes]}}
    hb = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"\0" * nbytes)


def build(n_shards, with_index=True):
    d = pathlib.Path(tempfile.mkdtemp())
    names = ["model-%05d-of-%05d.safetensors" % (i + 1, n_shards) for i in range(n_shards)]
    for nm in names:
        shard(d / nm)
    if with_index:
        (d / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": 16 * n_shards},
             "weight_map": {"w%d" % i: nm for i, nm in enumerate(names)}}))
    return d, names


def problems(d):
    return mod.check(str(d))


d, names = build(3)
check("an intact sharded checkpoint is clean", problems(d), [])
shutil.rmtree(str(d))

d, names = build(1, with_index=False)
check("a single-shard checkpoint with no index is clean", problems(d), [])
shutil.rmtree(str(d))

d, names = build(3)
(d / names[1]).unlink()
got = problems(d)
check("a missing shard is reported", (len(got), "missing 1 of 3 shards" in got[0]), (1, True))
shutil.rmtree(str(d))

d, names = build(3)
p = d / names[2]
with p.open("r+b") as f:
    f.truncate(p.stat().st_size - 1)
got = problems(d)
check("a shard short by one byte is reported", (len(got), "bytes on disk" in got[0]), (1, True))
shutil.rmtree(str(d))

d, names = build(2)
(d / names[0]).write_bytes(b"\x00" * 200)
got = problems(d)
check("a shard with an unreadable header is reported", (len(got), "truncated or corrupt" in got[0]), (1, True))
shutil.rmtree(str(d))

d, names = build(2)
for nm in names:
    q = d / nm
    with q.open("r+b") as f:
        f.truncate(q.stat().st_size - 2)
check("every broken shard is listed, not just the first", len(problems(d)), 2)
shutil.rmtree(str(d))

d = pathlib.Path(tempfile.mkdtemp())
check("a directory with no safetensors is reported", ["no safetensors" in problems(d)[0]], [True])
shutil.rmtree(str(d))

check("a path that is not a directory is reported",
      ["not a directory" in problems(pathlib.Path(tempfile.mkdtemp()) / "nope")[0]], [True])

d, names = build(2)
(d / "model.safetensors.index.json").write_text("{ not json")
check("an unreadable index is reported", ["unreadable" in problems(d)[0]], [True])
shutil.rmtree(str(d))

# A shard larger than its header declares is also wrong: trailing bytes mean a botched write.
d, names = build(2)
p = d / names[0]
p.write_bytes(p.read_bytes() + b"\x00")
check("a shard longer than declared is reported", (len(problems(d)), "bytes on disk" in problems(d)[0]),
      (1, True))
shutil.rmtree(str(d))

if FAILURES:
    print("%d FAILURE(S):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests pass")
