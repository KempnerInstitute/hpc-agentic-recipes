#!/usr/bin/env python3
"""Check that a downloaded checkpoint is complete before serving it.

    python3 common/tools/verify_checkpoint.py <checkpoint-dir>

An incomplete download fails deep inside weight loading with an error that points at the model rather than
at the file, so it is worth ruling out first. Two things go wrong: a shard is missing, or a shard is
present but short.

Counting files catches only the first. Every safetensors file states its own length in its header, so this
compares that against the size on disk, which catches a shard truncated by as little as one byte. Only
headers are read, so a 1.5 TiB checkpoint verifies in seconds.

Missing shards are found from model.safetensors.index.json when it exists. A single-shard checkpoint ships
no index, which is not an error.

Exits non-zero and names the files if anything is wrong.
"""
import json
import pathlib
import struct
import sys


def read_declared_size(path):
    """Bytes this file says it should be: the 8-byte length, the JSON header, and the tensor data."""
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    return 8 + header_len + end


def check(directory):
    d = pathlib.Path(directory)
    if not d.is_dir():
        return ["%s is not a directory" % d]
    shards = sorted(d.glob("*.safetensors"))
    if not shards:
        return ["no safetensors files in %s" % d]

    index = d / "model.safetensors.index.json"
    if index.exists():
        try:
            want = set(json.loads(index.read_text())["weight_map"].values())
        except Exception as e:
            return ["%s is unreadable (%s)" % (index.name, type(e).__name__)]
        missing = sorted(want - {p.name for p in shards})
        if missing:
            return ["missing %d of %d shards, starting with %s"
                    % (len(missing), len(want), ", ".join(missing[:5]))]

    problems = []
    for p in shards:
        size = p.stat().st_size
        try:
            declared = read_declared_size(p)
        except Exception as e:
            problems.append("%s: header unreadable, so the file is truncated or corrupt (%s)"
                            % (p.name, type(e).__name__))
            continue
        if size != declared:
            problems.append("%s: %d bytes on disk, header declares %d" % (p.name, size, declared))
    return problems


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: verify_checkpoint.py <checkpoint-dir>")
    found = check(sys.argv[1])
    if found:
        print("incomplete:")
        for f in found:
            print("  " + f)
        sys.exit(1)
    n = len(sorted(pathlib.Path(sys.argv[1]).glob("*.safetensors")))
    print("%d shard%s, all complete" % (n, "" if n == 1 else "s"))
