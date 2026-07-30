# Downloading weights

Checkpoints are read from `MODELS_DIR`, which defaults to the shared testbed location:

```
/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/<Checkpoint-Name>
```

Most models a recipe references are already there. Check before downloading hundreds of gigabytes.

## Fetching a new checkpoint

```
bash common/tools/download_model.sh <hf-repo> [local-name]
```

Do this from a compute node, not a login node. These are large transfers, and login nodes are shared.

The tool sets `HF_HUB_ENABLE_HF_TRANSFER=0` deliberately. The accelerated transfer path has been
unreliable for very large repositories here, and the plain path resumes cleanly, which matters more than
peak speed for a 500 GB download. For the same reason `HF_HUB_DISABLE_XET=1` is worth setting: the Xet
backend has caused stalls on multi-hundred-gigabyte checkpoints.

Downloads resume, so an interrupted transfer can simply be rerun.

## Verify before serving

A truncated shard produces a confusing failure deep inside weight loading, so confirm the download is
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

## Faster storage

Copying a checkpoint to your own VAST scratch space loads measurably faster than Lustre, and the
directory names are identical in both locations, so only `MODELS_DIR` changes. Scratch has a 90-day
retention policy: treat it as a fast cache and keep the testbed copy as the system of record.
