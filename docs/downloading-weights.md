# Downloading weights

Every model a recipe references is already staged in the shared testbed, which `MODELS_DIR` points at by
default. Check there before downloading hundreds of gigabytes.

That directory is **read-only** unless you are in the data administrators group, so it is where you read
from, not where you download to. If a checkpoint you need is missing from it, either ask an administrator
to stage it, or download your own copy anywhere you can write and point `MODELS_DIR` at that instead.

## Choosing where to put it

Anywhere on the cluster you have write access works: your lab's main storage, your lab's scratch space,
or your own directory under either. Two things to weigh:

- **Speed.** Scratch (`/n/netscratch`) is faster than Lustre (`/n/holylfs*`) for this workload.
- **Retention.** Scratch has a 90-day retention policy, so a copy there is a cache, not storage. Lab
  storage keeps it.

Directory names are identical wherever you put it, so switching between copies only changes `MODELS_DIR`.

## Fetching a checkpoint

```
bash common/tools/download_model.sh <hf-repo> <dest-parent-dir> [local-name]
```

For example, into a lab scratch directory:

```
bash common/tools/download_model.sh zai-org/GLM-5.2-FP8 /n/netscratch/<your_lab>/<you>/models
```

Run it from a compute node, not a login node. These are large transfers and login nodes are shared. The
tool refuses to start if the destination does not exist or is not writable by you, rather than failing
several minutes into the transfer. Pass `--dry-run` first to see the resolved destination and the free
space there without transferring anything.

It turns off both accelerated Hugging Face transfer backends, `HF_HUB_ENABLE_HF_TRANSFER` and the Xet
backend. The plain path resumes cleanly after an interruption, which matters more than peak speed for a
several-hundred-gigabyte repository. An interrupted transfer can simply be rerun.

## Verify before serving

A truncated shard fails deep inside weight loading with a confusing error, so confirm the download is
complete first. Compare the shard count against the index:

```
python3 -c "
import json, glob, sys
d = sys.argv[1]
idx = json.load(open(d + '/model.safetensors.index.json'))
want = set(idx['weight_map'].values())
have = {p.split('/')[-1] for p in glob.glob(d + '/*.safetensors')}
print('shards expected', len(want), 'present', len(have))
print('missing:', sorted(want - have)[:5])
" /path/to/checkpoint
```

## Point a recipe at it

Set `MODELS_DIR` to the parent directory, or `MODEL` to the checkpoint itself, either exported or set in
`common/site.conf`:

```
export MODELS_DIR=/n/netscratch/<your_lab>/<you>/models
```
