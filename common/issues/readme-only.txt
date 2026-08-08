# Recipes that are documentation only, so only README.md is required of them.
#
# One path per line, as <Checkpoint-Name>/<hardware>, which is
# <Checkpoint-Name>/<hardware> for a variant directory. Matched with grep -qxF, so the line must be
# the whole path with no leading "recipes/" and no trailing slash. Lines starting with # are ignored
# because they cannot match any recipe path.
#
# A recipe belongs here when it will never be launched as written: a Blocked recipe holding a negative
# result, or a model with no hardware variant authored yet. Everything else must ship the full set of
# scripts.

# Blocked: four H200 configurations all failed at CUDA graph capture. Serve this model on RTX instead.
Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4

# Blocked: every rank fails a shape assertion in SGLang's DeepSeek weight loader, so this engine has
# The two entries below name model directories that hold a README.md and no hardware subdirectory.
# The checks glob recipes/*/*/ and therefore do not enumerate them at all, so these lines
# are documentation of intent rather than something the audit consults today. They are listed so that
# the intent survives if the audit is later taught to walk model-level READMEs.
#
# Blocked: KimiK3ForConditionalGeneration is in no released engine, so nothing can load the weights.
# Untested: the bf16 twin of the FP8 Coder checkpoint, staged and supported but never launched.
Qwen3-Coder-480B-A35B-Instruct
