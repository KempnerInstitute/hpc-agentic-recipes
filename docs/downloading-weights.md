# Downloading weights

Every model a recipe references is already staged in the shared model repository on Kempner AI Cluster,
which `MODELS_DIR` points at by default. Check there before downloading hundreds of gigabytes.

That directory is **read-only**. If a checkpoint you need is missing from it, either submit a ticket to
request it, or download your own copy anywhere you can write and point `MODELS_DIR` at that instead.

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
several minutes into the transfer. Pass `--dry-run` first to see the resolved destination without
transferring anything.

Check that you have room before starting. The tool does not, because free space on the filesystem is not the
limit that applies to you: what binds is your group's quota. Read it with `lfs quota -g <group>
/n/holylfs06` on Lustre, or `quota -g <group> /n/netscratch` on scratch, which is not Lustre and so has no
`lfs quota`.

It turns off both accelerated Hugging Face transfer backends, `HF_HUB_ENABLE_HF_TRANSFER` and the Xet
backend. The plain path resumes cleanly after an interruption, which matters more than peak speed for a
several-hundred-gigabyte repository. An interrupted transfer can simply be rerun.

## Verify before serving

An incomplete download fails deep inside weight loading with an error that points at the model rather than
at the file, so rule it out first:

```
python3 common/tools/verify_checkpoint.py /path/to/checkpoint
```

It reports a missing shard, and also a shard that is present but short, which counting files cannot see.
Every safetensors file states its own length in its header, so that is compared against the size on disk,
down to a single byte. Only headers are read, so a 1.5 TiB checkpoint verifies in seconds. It exits non-zero
and names the files when something is wrong.

## Point a recipe at it

Set `MODELS_DIR` to the parent directory, or `MODEL` to the checkpoint itself, either exported or set in
`common/site.conf`:

```
export MODELS_DIR=/n/netscratch/<your_lab>/<you>/models
```
