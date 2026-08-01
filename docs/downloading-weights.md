# Downloading weights

Checkpoints are read from `MODELS_DIR`, which defaults to the shared testbed location:

```
/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/<Checkpoint-Name>
```

Every model a recipe references is already there. Check before downloading hundreds of gigabytes.

## Fetching a new checkpoint

```
bash common/tools/download_model.sh <hf-repo> [local-name]
```

Run it from a compute node, not a login node. These are large transfers and login nodes are shared.

The tool turns off both accelerated Hugging Face transfer backends, `HF_HUB_ENABLE_HF_TRANSFER` and the Xet
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

## Faster storage

Scratch (`/n/netscratch`) is faster than the Lustre testbed path (`/n/holylfs06`) for this workload, and
directory names are identical in both, so only `MODELS_DIR` changes. Scratch has a 90-day retention policy:
treat a copy there as a cache and keep the testbed copy as the permanent one.
