# Repo restructure plan (v2)

Turn this repo into a recipe collection for running open-weight agentic coding models on the Kempner
AI Cluster. Every model variant becomes a self-contained recipe: its own environment build, its own
hardware notes, its own launch scripts, its own measured numbers.

Status: plan for review, revision 2. Nothing has been implemented beyond one commit on branch
`repo-restructure` carrying this session's serving hardening.

Revision 2 incorporates two independent audits, one for factual correctness against the repo and one
for usability against the standalone principle. They produced 36 findings between them. Every
finding is either folded in below or listed in section 15 with a reason. Three of the findings
corrected claims in v1 that were simply wrong, and those corrections are called out inline.

## 0. Governing principle: recipes are standalone

A user who opens one recipe reads everything they need and never has to open a second file to
succeed. This outranks brevity and anti-duplication.

- Every issue affecting a recipe is written out in full in that recipe's README, even when identical
  text appears in eight others.
- No recipe README may say "see the top-level README" or "see docs/troubleshooting.md" for anything
  required to build, launch, verify, connect to, or debug that model.
- The full environment build, launch, verification, client configuration, tunable inputs, measured
  numbers, and complete gotcha list all appear inline.

### How duplication is kept correct rather than rotting

Duplicating text by hand across eleven recipes guarantees drift. So duplication is generated and
verified, not typed:

- `common/issues/<slug>.md` holds the canonical prose for each known issue.
- `common/fragments/<slug>.sh` holds canonical shell fragments, for example the InfiniBand HCA
  autodetection loop or the CUDA 13 toolkit wiring.
- `common/issues/matrix.tsv` maps each slug to the recipes it applies to.
- Recipe files carry `# issue:<slug> begin` and `# issue:<slug> end` markers.
- `common/tools/audit_recipes.sh --fix` injects canonical content into every marked block. A plain
  run verifies byte equality and fails on drift, on a block the matrix does not list, and on a
  matrix entry with no block.

This is why recipe `env/env.sh` files are complete standalone preambles that source nothing at
runtime, yet cannot silently diverge.

### Resolving the audit disagreement about shared flags

The correctness audit argued that `NCCL_P2P_DISABLE=1` and `FLASHINFER_DISABLE_VERSION_CHECK=1` are
node and toolchain properties, not model properties, so they belong in a shared `common/lib/rtx_node.sh`
that every RTX recipe sources, because the new failure mode of duplication is one recipe omitting
`NCCL_P2P_DISABLE` and hanging at NCCL init. The usability audit classified the same two as
hardware-wide issues that must nonetheless appear in every affected recipe.

Decision: duplicate them into each recipe's `env/env.sh` via the canonical-fragment mechanism, so the
standalone principle holds and the file a user reads is complete. The omission risk the correctness
audit identified is real, so `audit_recipes.sh` asserts that every RTX recipe sets
`NCCL_P2P_DISABLE=1`, every multi-rank recipe sets `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`, and the H200
GLM recipes set `VLLM_USE_DEEP_GEMM=0`. Duplication plus a mandatory-presence check is strictly safer
than a shared file, because a recipe that fails to source the shared file fails the same way and no
check would catch it.

## 1. Decisions

| Question | Decision |
| --- | --- |
| Environment isolation | Strict physical isolation. One environment per recipe, never shared, even when pins are identical. |
| Directory naming | Exact downloaded checkpoint name, then a hardware directory. Verified: all eleven checkpoint names in this plan exist in both the testbed and netscratch trees with no case or spelling mismatch. |
| Hardware directory naming | One axis: `<gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>]`. See section 3. |
| Environment location | Under `ENV_ROOT`, defaulting to `/n/netscratch/kempner_dev/Lab/mmsh/agentic-coding-llm-env`, one isolated subdirectory per recipe. |
| Documented launch path | Slurm sbatch canonical, direct SSH secondary. |
| Test bar for the PR | PR first, then validate H200 recipes over direct SSH on the four idle reservation nodes. RTX and Slurm validated later, with the exception in section 14. |
| Shared flags | Duplicated per recipe, enforced by audit. See section 0. |
| Hardening commit | Already committed as `41f57fd` on `repo-restructure`, so it is reviewed in this PR rather than landing on `main` separately. v1 left this as an open question. |

## 2. Why restructure

50 tracked files: a flat `scripts/` holding 37, `clients/` holding 8, and a 431-line README with 10
top-level sections. Three real problems.

**Adding a model touches five scattered places.** A new model needs `scripts/vllm_<m>.sh`,
`scripts/serve_<m>_ssh.sh`, `scripts/slurm_<m>.sbatch`, entries in `scripts/config.sh`, a
`clients/claude-code-<m>.env`, and edits across at least five README sections. Nothing tells a
contributor that list exists. Worse, five existing scripts each serve two models through internal
branching, so the real coupling is invisible.

**Model-specific facts and hardware facts are tangled in three shared env libs.** `lib_env.sh`,
`lib_env_cu130.sh`, and `lib_env_sglang.sh` each mix venv activation, module loads, toolchain wiring,
cache directories, InfiniBand detection, API key loading, and model-specific workarounds. A
contributor cannot tell which lines exist for which model.

**Correction to v1.** v1 claimed `VLLM_USE_DEEP_GEMM=0` in `lib_env.sh` was set purely for GLM-5.2
and penalized every other H200 model. That is wrong. `README.md:358-360` records that forcing
`VLLM_USE_DEEP_GEMM=1` on H200 reproduces the same crash for Qwen3-Coder-480B-FP8, so the flag is
load-bearing for at least two models. The real problem is not that the flag is wrong, it is that
nothing records which models it protects, so a future recipe author cannot tell whether flipping it
is safe. That is an argument for per-recipe provenance, not for per-recipe freedom, and section 5
handles it with mandatory provenance tags.

**One README serves two audiences.** A user choosing a model reads the same document as an operator
building a CUDA 13 toolchain for FlashInfer's sm_120 JIT.

## 3. Naming

Recipe directories use the exact checkpoint name, then a hardware directory on a single axis:

```
recipes/<Checkpoint-Name>/<gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>]/
```

Examples: `rtx-8`, `h200-4`, `h100-1`, `h200-1`, `h200-4-nodes2`, `rtx-8-nodes2`,
`h200-4-nodes2-sglang`.

v1 used `rtx`, `h200x2`, and `single-gpu`, which both audits rejected. `single-gpu` is not a GPU type
and collapsed three separately measured variants with two incompatible toolchains, and `<type>x<N>`
would have produced `rtxx2`.

The consequence is that the two Gemma recipes split by GPU type, because they genuinely need
different environments: on RTX they need torch cu130, the conda CUDA 13 toolkit, and FlashInfer
0.6.15, while on H200 and H100 they need the cu129 release wheel. The current repo expresses this
through `--export=ALL,ENV_LIB=lib_env.sh`, and the README already publishes three separate decode
rates per Gemma model.

## 4. Target structure

```
README.md                     index: fastest path, model table, how to contribute
recipes/
  <Checkpoint-Name>/
    README.md                 model overview, variant index, checkpoint provenance, HF repo id
    <hardware>/
      README.md               the complete standalone document, 15 required sections
      env/
        build.sh              the only supported build path
        env.sh                complete runtime preamble, sources nothing
        requirements.lock     uv pip compile output with hashes
        WHEELS                exact URLs plus sha256 for every non-PyPI artifact
      serve.sh                the engine invocation
      serve.sbatch            canonical launch
      serve_ssh.sh            secondary launch
      ray_head.sh             multi-node recipes only, parameterized by GPUS_PER_NODE
      ray_worker.sh           multi-node recipes only
      client.env              client configuration for this endpoint
common/
  defaults.sh                 tracked, cluster-correct defaults so a fresh clone works
  site.conf.example           optional overrides: hostnames, ACCOUNT
  lib/
    repo_root.sh              single anchored REPO_ROOT resolution
    api_key.sh                key resolution from REPO_ROOT
  issues/
    <slug>.md                 canonical issue prose
    matrix.tsv                slug to recipe mapping, with a must-not-apply column
    readme-only.txt           allowlist of recipes that are documentation only
  fragments/
    <slug>.sh                 canonical shell fragments
  templates/
    recipe-README.md
  tools/
    bench.sh                  slope method, to be written, see section 9
    smoke_test.sh
    chat.sh
    stop.sh                   takes a node list
    download_model.sh
    search.sh
    audit_recipes.sh
    rebuild_envs_scratch_space.sh
    new_recipe.sh
docs/
  quickstart.md               connect to an endpoint someone else is already serving
  choosing-a-model.md         comparison table plus the model-choice prose
  engines.md                  vLLM versus SGLang, Anthropic versus OpenAI APIs
  clients.md                  OpenAI-compatible clients
  hardware.md
  adding-a-model.md
  benchmarking.md             slope method plus the three-regime analysis
  downloading-weights.md
  web-search.md
  troubleshooting.md          aggregated view only
  roadmap.md                  from plan-new-models.md
common/skills/local-search/SKILL.md    moved out of .claude, see section 9
logs/.gitkeep
```

Recipes, eleven runnable plus three documentation-only:

| Recipe | Status intent |
| --- | --- |
| `GLM-5.2-NVFP4/rtx-8` | Untested (migrated) until an RTX run |
| `GLM-5.2-FP8/h200-4-nodes2` | validate in phase 7 |
| `GLM-5.2-FP8/h200-4-nodes2-sglang` | Blocked, see section 9 |
| `GLM-4.6-FP8/h200-4` | validate in phase 7 |
| `Kimi-K2.7-Code/rtx-8` | Untested (migrated) |
| `Kimi-K2.7-Code/h200-4-nodes2` | validate in phase 7 |
| `Qwen3-235B-A22B/rtx-8` | Untested (migrated) |
| `Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8` | Untested (migrated) |
| `Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4` | Blocked, holds the four-attempt CUTLASS table |
| `gemma-4-26B-A4B-it/{rtx-1,h200-1,h100-1}` | validate `h200-1` in phase 7 |
| `gemma-4-31B-it/{rtx-1,h200-1,h100-1}` | validate `h200-1` in phase 7 |
| `DeepSeek-V4-Pro/rtx-8-nodes2` | Untested, primary target, see section 9 |
| `DeepSeek-V4-Pro/h200-4-nodes2` | Untested, documented as expected to fail |
| `Kimi-K3/` | Blocked, README only |
| `Qwen3-Coder-480B-A35B-Instruct/` | Untested, README only, bf16 |

## 5. Recipe anatomy: 15 required sections

Enforced by `audit_recipes.sh`. v1 had 11 and both audits found content with no home.

0. **Configure once.** Inline: copy `common/site.conf.example` if overriding anything, create the API
   key with `mkdir -p secrets && printf '%s' '<key>' > secrets/vllm_api_key && chmod 600 ...`, and the
   exact variables this recipe reads. A fresh clone must work without this step, per section 7.
1. **Status.** Structured and dated: `Status: Validated - 2026-07-29, vLLM 0.25.1, commit <sha>, protocol: slope(128,1152)`.
   `Untested (migrated)` means the numbers were measured with the pre-restructure scripts and the new
   recipe has not been run. `Blocked` states the blocker. Audit refuses `Validated` without a date, an
   engine version, and a protocol label.
2. **What this is.** Model, checkpoint, HF repo id, testbed path, and the note about copying to
   personal netscratch for speed.
3. **Hardware.** GPU type, GPUs per node, node count, total VRAM, partition, account variable, and
   the per-GPU CPU and memory limits. Verified from `scontrol`: RTX 8 GPUs per node, H200 and H100 4
   each, 16 CPUs per GPU on `kempner_h200` and `kempner_rtx`, 24 on `kempner_h100`, 2-day max time.
4. **Environment build.** The full command sequence inline, including the rationale for
   non-obvious choices such as why H200 needs the cu129 release wheel rather than the default CUDA 13
   PyPI wheel, with the exact wheel URL.
5. **Launch.** Canonical sbatch, then direct SSH, both complete, and explicitly the submit directory,
   because Slurm stages the batch script and `$0` does not point at the repo. See section 9.
6. **Verify.** Exact curl commands, including that a keyless request must return 401, and for
   thinking models that `max_tokens` under roughly 400 returns empty `content` because the budget went
   to the `reasoning` field. Omitted for recipes with no reasoning parser, since pasting it there
   would be wrong.
7. **Connect a client.** For vLLM: the full Anthropic block, including that `ANTHROPIC_AUTH_TOKEN`
   must be used rather than `ANTHROPIC_API_KEY`, and that `ANTHROPIC_SMALL_FAST_MODEL` must be set.
   For SGLang: an OpenAI-compatible block plus the note that no Anthropic API exists.
8. **Tunable inputs.** Every environment variable this recipe's scripts honor, with default and
   effect. The current README documents about 19 of these and v1 had nowhere to put them.
9. **Web search.** The HTTP 400 on Anthropic hosted tools, plus the `search.sh` invocation and the two
   install commands. Required inline because it is needed to use the endpoint.
10. **Measured performance.** Rate, protocol, and what was tried that did not help.
11. **Parallelism and quantization.** Why this TP, PP, and quantization, including divisibility
    constraints against the quantization block size.
12. **Gotchas.** Every issue from the matrix, in full.
13. **Stop the endpoint.** How to tear it down.
14. **Expected startup time.** Environment build, weight load, and first-time JIT, cold and warm, so
    nobody kills a healthy job. Phase 7 measures at least one honestly.

Every flag in `env/env.sh` and in section 12 carries a provenance tag, either
`# verified: <symptom>, <date>` or `# inherited from lib_env.sh, untested for this model`. The audit
rejects an export whose comment matches neither. This is how the DeepGEMM situation stops being
folklore: GLM-5.2 gets `verified`, GLM-4.6 gets `inherited`, and DeepSeek carries an explicit
counter-warning rather than an optimistic flip.

`toolchain.md` from v1 is deleted. It was a second copy of the build instructions with no defined
delta, which is both a drift generator and exactly the second file section 0 forbids.

## 6. Environments and checkpoints

Strict isolation: one venv per recipe at `$ENV_ROOT/<Checkpoint-Name>/<hardware>/venv`, plus a
per-recipe `cuda13/` for RTX recipes. Measured sizes:

| Environment | Size | Files |
| --- | --- | --- |
| H200 cu129, vLLM 0.25.1 plus Ray | 13 GB | 75,703 |
| RTX cu130 | 9.0 GB | 72,068 |
| SGLang 0.5.11.dev | 11 GB | 64,063 |
| conda CUDA 13.0 toolkit, RTX recipes only | 2.9 GB | 6,305 |

So an RTX recipe costs about 11.9 GB and a Hopper recipe about 13 GB. Across the eleven recipes that
build environments, with the Gemma models split three ways per section 3 and `Blocked` recipes
building nothing, the total is roughly 190 GB. netscratch has 515 TB available, so capacity is not a
constraint.

Environments live on netscratch because it is measurably faster: importing torch and vLLM took about
14 minutes from Lustre and 9.2 seconds from VAST, measured on the same node today.

**Existing contents must not be destroyed.** `ENV_ROOT` currently holds a flat `venv-cu130/` and
`cuda13/` that the live RTX endpoints are running from via `VENV_DIR`. The migration creates the new
per-recipe layout alongside and removes the flat copies only after both RTX recipes are validated.

### Reproducibility, corrected

v1 claimed a "fully pinned requirements.txt" per recipe. Both audits showed that is unachievable for
five recipes, and the 90-day rebuild story depended on it:

- The four RTX recipes were built from `https://wheels.vllm.ai/nightly/cu130` with
  `--prerelease=allow --index-strategy unsafe-best-match`. Nightly wheels rotate and are deleted. The
  installed metadata reads `vllm 0.25.1` with no local version tag, so a naive pin `vllm==0.25.1`
  resolves to the PyPI CUDA 13 wheel, a silently different environment from the one tested.
- `flashinfer_python==0.6.15` was installed `--no-deps` and coexists with `flashinfer_cubin==0.6.13`.
  A requirements file cannot express `--no-deps` for one package, and that exact skew is why
  `FLASHINFER_DISABLE_VERSION_CHECK=1` exists.
- H200 recipes need a GitHub release wheel URL plus `--torch-backend=cu129`, which are uv flags, not
  requirements lines.
- SGLang is `0.5.11.dev20260420+g3063d640d`, a dated dev build from a git hash that `sglang>=0.5.10`
  will never resolve to again.

So `env/` carries: `build.sh` as the only supported build path, since it can express `--no-deps`,
`--torch-backend`, index strategy, and the mamba toolkit step; `requirements.lock` from
`uv pip compile --generate-hashes` as the authoritative pin; and `WHEELS` listing exact resolved URLs
with sha256 for every non-PyPI artifact. First build populates a vendored wheel cache under
`$ENV_ROOT/wheels/` so a rebuild still works after a nightly index rotates. Phase 1 archives the
wheels for the three live environments before scratch expiry. The audit rejects any `>=` in a pin file
and any non-PyPI artifact without a URL and hash.

### Checkpoint paths

Documented default, tracked, correct for any cluster user:

```
/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/<Checkpoint-Name>
```

Used for our validation runs, as a launch-time override only:

```
/n/netscratch/kempner_dev/Lab/mmsh/models/<Checkpoint-Name>
```

Every recipe carries the note that testbed works out of the box and copying to personal netscratch
loads faster, that scratch is a 90-day cache rather than the system of record, and that folder names
are identical across both locations. The environment speedup is described as measured; the checkpoint
speedup is described as expected until phase 7 produces a number, since Kimi's 64 shards took roughly
35 seconds each from Lustre and the netscratch comparison has not been run.

## 7. Configuration: a fresh clone must work

v1 put `ENV_ROOT`, `LOG_DIR`, and hostnames in a gitignored `site.conf`, which both audits flagged as
a straight regression. Today `scripts/config.sh` is tracked with working defaults, so
`bash scripts/serve_glm46_ssh.sh` runs on a fresh clone. v1 would have broken every recipe until a
user created a file no recipe mentioned, and would also have broken the phase 5 fresh-clone check,
every `client.env`, and four tools.

Corrected design:

- `common/defaults.sh` is **tracked** and carries cluster-correct values for `MODELS_DIR`,
  `ENV_ROOT`, `LOG_DIR`, `API_PORT`, and the partition names. A fresh clone works with no user action.
- `common/site.conf` is optional and gitignored, for hostnames and `ACCOUNT` only.
  `common/site.conf.example` documents it.
- Loading order is defaults, then site.conf if present, then environment variables.
- `audit_recipes.sh` asserts every variable any recipe script reads appears in `defaults.sh` or
  `site.conf.example`, and in that recipe's Configure once section.

`ACCOUNT` matters for public release: all nine current sbatch files hardcode
`#SBATCH --account=kempner_dev`, a team-only allocation that fails for any external user. The
`#SBATCH --account` line is removed and the account passed at submit time or from `site.conf`.

## 8. The issue matrix

v1 listed 6 issues in prose and claimed a table that did not exist. The real set is about 19, and
most were recorded only in script comments that the migration deletes. Applicability is explicit,
including where an issue must **not** appear, because a mandatory paste that is wrong in a recipe is
worse than a cross-link.

Recipe keys: R1 `GLM-5.2-NVFP4/rtx-8`, R2 `GLM-5.2-FP8/h200-4-nodes2`, R3 the SGLang sibling,
R4 `GLM-4.6-FP8/h200-4`, R5 `Kimi/rtx-8`, R6 `Kimi/h200-4-nodes2`, R7 `Qwen3-235B/rtx-8`,
R8 `Coder-480B-FP8/rtx-8`, R8h `Coder-480B-FP8/h200-4`, R9 Gemma-26B, R10 Gemma-31B,
R11 `DeepSeek-V4-Pro/rtx-8-nodes2`, R11h the H200 sibling.

| Issue | Scope | Applies to | Must not appear in |
| --- | --- | --- | --- |
| Lustre stall plus NCCL watchdog self-kill, `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` | multi-rank | R1-R8, R11 | R9/R10 single-GPU, no NCCL group |
| Node-local `LOG_DIR` | all | all | none |
| Node-local JIT caches, stale NFS handles under concurrent multi-node compiles | multi-node | R2, R3, R6, R11 | none |
| `VLLM_ENGINE_READY_TIMEOUT_S=3600` | vLLM | all vLLM | R3 |
| Anthropic hosted tools return 400, mitigation is `search.sh` plus the skill | vLLM | all vLLM | R3 needs the variant text |
| `ANTHROPIC_AUTH_TOKEN` not `ANTHROPIC_API_KEY`, plus `ANTHROPIC_SMALL_FAST_MODEL` | client | all vLLM | R3, which has no Anthropic API |
| RTX has no NVLink, `NCCL_P2P_DISABLE=1` or NCCL init hangs | RTX multi-GPU | R1, R5, R7, R8, R11 | inert-at-TP1 note only for RTX Gemma |
| RTX is comms-bound: no expert parallelism, TP8 all-reduce over PCIe | RTX | R1, R5, R7, R8 | none |
| FlashInfer 0.6.15 with cubin 0.6.13 needs `FLASHINFER_DISABLE_VERSION_CHECK=1`; conda CUDA 13 toolkit; never add it to `LD_LIBRARY_PATH` | RTX toolchain | all RTX recipes | none |
| Driver 575 and CUDA 12.9 cannot run the default CUDA 13 wheel, use the cu129 release wheel | Hopper | R2, R4, R6, R8h, H200/H100 Gemma | none |
| CUDA graphs crash on H200, `--enforce-eager` | hardware plus model | R2, R4 | none |
| DeepGEMM crashes on GLM-5.2 sparse attention, and forcing it on for Coder-480B on H200 reproduces the crash | H200 | R2 verified, R4 inherited, R8h, R11h counter-warning | none |
| Pipeline parallelism forbids speculative decoding, so no MTP on any PP recipe | topology | R2, R6, R8, R11 | none |
| Pure cross-node TP hangs at NCCL init, keep TP intra-node and PP across | topology | R2, R6, R11 | none |
| Cross-node multimodal profiling deadlock, `--skip-mm-profiling --mm-processor-cache-gb 0` | topology and modality | R6 only | R5 single node |
| Coder-480B FP8 cannot run TP8, 2560/8 is 320 and not a multiple of 128 | model | R8, R8h, and the bf16 README stating it does not apply | none |
| Coder-480B FP8 fails on H200 with CUDA graphs, CUTLASS w8a8, four attempts | model and hardware | R8h, which exists to hold it | none |
| Gemma 4 MTP broken in 0.25.1 and 0.26.0, drafter checkpoints unusable | model family | R9, R10 all variants | none |
| SGLang endpoint is not key-gated and has never loaded weights | recipe bug | R3 | none |

## 9. Repo bugs to fix during the restructure

Found by the audits and verified directly. These are defects in `main`, not restructure artifacts.

**The SGLang endpoint is ungated.** `lib_env_sglang.sh` contains zero API key references and
`sglang_glm.sh` passes no `--api-key`, while `README.md:5` and `:133` claim every endpoint is
key-gated. Not currently exposed, since only the two vLLM RTX endpoints are running. Fix the claim
and add key loading.

**SGLang has never worked.** `logs/sglang-glm.log` ends in four
`assert param.size() == loaded_weight.size()` failures inside `deepseek_weight_loader.do_load_weights`
and three sigquits. No measured SGLang rate exists anywhere. R3 becomes `Blocked` with that traceback
recorded, and the Measured performance requirement is waived for it. v1 treated SGLang as a working
alternative engine.

**`bench.sh` does not implement the slope method.** Both audits caught this. It sends N requests at a
single length, which is the single-generation method the README says understates decode by up to 40
percent. v1 told contributors to measure with it. The slope logic exists only in untracked netscratch
probes. Phase 1 writes `common/tools/bench.sh` properly: time the same greedy request at 128 and 1152
output tokens, report `(1152-128)/(t2-t1)`, repeat three times, print the median plus the literal
protocol string `slope(128,1152)`. Keep the old behavior behind `--single` with a warning.

**API key resolution breaks in every moved file.** All six tools compute `REPO_DIR` as exactly one
level up, and all eight client envs do the same, then read `$REPO_DIR/secrets/vllm_api_key`. At
`common/tools/` that becomes `common/secrets/`, and at `recipes/<Name>/<hw>/` it becomes
`recipes/<Name>/secrets/`. Two tools fall back to an empty key and omit the header, so the symptom is
a bare 401 and the user debugs auth instead of a path. `common/lib/repo_root.sh` and
`common/lib/api_key.sh` own this, and every moved file is a rewrite, not a move.

**`ray_head.sh` and `ray_worker.sh` are not generic.** Both source `lib_env.sh` as a sibling, which
the split dissolves, and both hardcode `--num-gpus=4`, wrong for 8-GPU RTX nodes. With
`set -euo pipefail` they die instantly, so no multi-node recipe would get a Ray cluster. They become
per-recipe copies parameterized by `GPUS_PER_NODE`.

**`serve.sbatch` cannot find its own directory.** Slurm stages the batch script into the job spool, so
`$0` and `BASH_SOURCE` do not point at the repo. Every current sbatch handles this with
`cd "${SLURM_SUBMIT_DIR:-$(pwd)}"`. Recipe sbatch files keep that and use repo-root-relative paths,
and every recipe states that submission is from the repo root. This is a genuine tension with the
standalone principle and is documented as such rather than hidden.

**`#SBATCH --output=logs/...` is relative to the submit directory.** A user who does the natural
`cd recipes/<Name>/<hw> && sbatch serve.sbatch` has no `logs/` there, so Slurm fails the job before
the script runs, with no log to read. Also, keeping Slurm logs on Lustre contradicts the node-local
`LOG_DIR` rationale. Resolution: `--output` uses an absolute path under `$HOME`, and the audit checks
it is not a bare relative `logs/`.

**`clients/claude-code.env` is the GLM-4.6 client**, not a generic example. It maps to
`recipes/GLM-4.6-FP8/h200-4/client.env`, and a fresh generic example is written for
`docs/quickstart.md`. Client envs are also missing or ambiguous for four recipes: Kimi h200, the two
GLM-5.2 variants, DeepSeek, and the Gemma pair share one node variable.

**`SKILL.md` is a migration target, not a bystander.** It names `scripts/search.sh` at lines 3 and 12,
and the README references `scripts/` 33 times.

**`stop_ssh.sh` is not generic.** It only tears down the GLM-5.2 H200 pair and sources `lib_env.sh`
over SSH. It becomes `stop.sh <host> [host...]`.

**DeepSeek-V4-Pro targeting was wrong in v1.** `config.json` has `expert_dtype: fp4` with
`quant_method: fp8`, and `plan-new-models.md` recommends 2 RTX nodes because Blackwell executes FP4
experts natively, warning that Hopper would fall back to emulation or a Marlin path with correctness
and speed risk. v1 made `h200x2` the only DeepSeek recipe and a phase 7 candidate, and cited
in-checkpoint MTP as a benefit while PP disables speculative decoding. Corrected: `rtx-8-nodes2` is
primary, `h200-4-nodes2` is a documented risky sibling, and DeepSeek leaves the phase 7 H200 list.
Note the RTX environment has no Ray installed, so an RTX multi-node recipe needs a new pin.

**Stale node default.** `config.sh` sets `KIMI_HEAD=holygpu8a10202`, which is not in the current
reservation, so a Kimi h200 validation would target a node outside it unless overridden.

**Kimi-K3 claims need sourcing.** v1 asserted absence from vLLM main and SGLang 0.5.16 and a
requirement for an unreleased 0.27.0, while `plan-new-models.md` says 0.25.1, 0.5.11, and 0.26.0.
Both are right about different things: the older text records what was checked earlier, and the newer
claims come from this session's direct verification of vLLM `main`'s registry, PyPI showing 0.26.0 as
latest, and the official recipe page stating 0.27.0 or newer. The recipe records the verification
method and date so the distinction is visible.

## 10. Migration map corrections

All 50 tracked files have a destination and every mapped file exists. v1's defects were fan-out
omissions: five scripts each serve two recipes and appeared once.

| Script | Recipes it feeds |
| --- | --- |
| `vllm_gemma4.sh`, `serve_gemma4_ssh.sh` | both Gemma models, times three hardware variants each |
| `vllm_qwen3.sh`, `serve_qwen3_ssh.sh` | Qwen3-235B and Coder-480B |
| `vllm_kimi.sh` | Kimi RTX and Kimi H200 |

Also corrected: `clients/claude-code.env` to R4 rather than a docs example; `stop_ssh.sh` to
`stop.sh` with the rename declared; `SKILL.md` edited rather than unchanged; `audit_recipes.sh`,
`rebuild_envs_scratch_space.sh`, `new_recipe.sh`, and `docs/roadmap.md` added to the tree, since v1 referenced them
without creating them.

`env/env.sh` is a complete runtime preamble, not "runtime variables only". v1's description would have
dropped venv activation, lmod module loads, `CUDAHOSTCXX`, the CUDA toolkit `PATH`, `CPATH`, and
`LIBRARY_PATH` wiring, `PATH="$HOME/.local/bin:$PATH"`, `HF_HUB_OFFLINE`, `PYTHONUNBUFFERED`, three
JIT cache directories and their `mkdir -p`, `VLLM_ENGINE_READY_TIMEOUT_S`, `NCCL_SOCKET_IFNAME`,
`GLOO_SOCKET_IFNAME`, `NCCL_DEBUG`, the eight-line InfiniBand HCA autodetection loop, and API key
loading.

Git history: recipes are created with `git mv` where a file maps to exactly one recipe, and by copy
where a script fans out to two. The fan-out copies cannot preserve history regardless, and the repo
has no rename history today, so this is stated rather than implied.

## 11. Adding a model

`docs/adding-a-model.md` covers the full sequence, not v1's eight steps. v1 also claimed nothing
outside the new directory needs editing except two index entries, which is false: it is four to six.

Before writing anything: identify the HF repo id and confirm the quantized variant exists; download
with `download_model.sh`, on a compute node, with `HF_HUB_DISABLE_XET=1`; and **verify the
architecture is in your engine's registry by grepping it rather than trusting a model card**. That
last technique is the most valuable contributor trick in the repo and had no home in v1.

Then: `new_recipe.sh <checkpoint> <hardware> --from <recipe>` to scaffold from the nearest existing
recipe so the copy passes the audit by construction, with a documented decision tree for which recipe
is nearest (GPU type picks the toolchain, node count picks mp versus Ray, dense versus MoE, quantized
versus bf16). Then pin, write `env.sh` from a checklist of candidate flags, write `serve.sh` covering
served name uniqueness, tool and reasoning parser discovery, `--enable-auto-tool-choice`, the
`${VAR-default}` versus `${VAR:-default}` distinction needed to omit a parser for non-thinking
models, TP and PP divisibility against the quantization block, and context length against KV
per token. Then launch and verify, then benchmark, then write the README from the template, then run
`audit_recipes.sh --fix` and the static checks, then update every index.

## 12. audit_recipes.sh

The script is the load-bearing mechanism for section 0, so it is specified rather than assumed:
structure, exact required headings in order, status format, issue propagation in both directions,
cross-reference ban, flag and provenance-tag agreement, mandatory flags per the matrix, numbers
agreement between each recipe and the comparison table, sbatch versus README agreement including no
hardcoded account and no reservation, client env agreement including `ANTHROPIC_MODEL` equal to
`--served-model-name` and no `ANTHROPIC_API_KEY`, served-name uniqueness, isolation invariants, pin
strictness, config completeness, hygiene, and index coverage in both directions.

The client-env check alone would have caught the auth bug fixed earlier in this repo's history, and
the numbers check protects the one thing that stays centralized.

## 13. Public release hygiene

No hostnames, no absolute user paths, no keys, no hardcoded account, no reservation flags, partitions
documented as `kempner_h200`, `kempner_rtx`, `kempner_h100`, measured numbers labeled with protocol,
`secrets/` gitignored, no em dashes, US spelling.

## 14. Phasing

Branch `repo-restructure`, in a separate worktree so `main` stays live for the currently serving
endpoints. Commit 1 is already in place.

1. Write `bench.sh` with the slope method; archive the wheels for the three live environments;
   generate `requirements.lock` and `WHEELS`.
2. Scaffold `common/` and `docs/`; write `defaults.sh`, `repo_root.sh`, `api_key.sh`; populate
   `common/issues/` and `common/fragments/` and `matrix.tsv`.
3. Write `audit_recipes.sh` before the recipes, so recipes are validated as they land.
4. Migrate recipes, running the audit after each.
5. Rewrite the README as an index and populate `docs/`, including the content both audits found
   homeless: the model-choice prose, the engines comparison, the tunables, access control and key
   rotation, the three-regime performance analysis, OpenAI-compatible clients, finding the host, and
   the first-launch-is-slow warning.
6. Static checks and a fresh-clone dry run that must work with no `site.conf`.
7. Open the PR.
8. Validate on the four idle H200 nodes over direct SSH, with `MODELS_DIR` pointed at netscratch:
   `gemma-4-31B-it/h200-1` for single GPU, `GLM-4.6-FP8/h200-4` for single-node TP4, and one of
   `GLM-5.2-FP8/h200-4-nodes2` or `Kimi-K2.7-Code/h200-4-nodes2` for multi-node Ray. Record weight
   load time so the checkpoint storage note cites a measurement. DeepSeek is not in this list.
9. Before merge, validate at least one RTX recipe, because `GLM-5.2-NVFP4/rtx-8` and
   `Kimi-K2.7-Code/rtx-8` are the endpoints users are actually on, and merging with both unexercised
   leaves the two fastest endpoints without a tested restart path. Both audits raised this; v1
   deferred all RTX work.

## 15. Open items

1. Directory case: exact checkpoint names as proposed, or normalize to lowercase.
2. Whether the SGLang recipe is worth carrying at all, given it has never loaded weights.
3. Whether the Gemma `SPEC_DRAFT` plumbing and the two drafter checkpoints survive into recipes, or
   are dropped until vLLM newer than 0.26.0 makes them usable.
4. Whether `Kimi-K3` and the bf16 Coder get placeholder directories or only roadmap entries.
5. Recipe tree named `recipes/`, since `models/` is unavailable because `.gitignore` uses it for
   weights.
