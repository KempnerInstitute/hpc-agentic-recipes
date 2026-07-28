# Central deployment configuration for the serve scripts.
#
# Every value can be overridden with an environment variable at launch, e.g.:
#   RTX_NODE=holygpu7c9999 bash scripts/serve_glm52_nvfp4_ssh.sh
#   MODEL=/scratch/GLM-5.2-NVFP4 bash scripts/serve_glm52_nvfp4_ssh.sh
#
# The node names below are EXAMPLES from the current na-kempner reservation. All H200 nodes share
# one hardware spec and all RTX PRO 6000 nodes share another, so only the hostnames differ between
# reservations. Resources required per model:
#   GLM-4.6 FP8    : 1 H200 node         (4 GPUs, TP4)
#   GLM-5.2 FP8    : 2 H200 nodes        (4 GPUs each, TP4 x PP2)
#   GLM-5.2 NVFP4  : 1 RTX PRO 6000 node (8 GPUs, TP8)
#   Kimi-K2.7-Code : 1 RTX PRO 6000 node (8 GPUs, TP8)  or  2 H200 nodes (TP4 x PP2)
#   Gemma-4-26B    : 1 GPU (TP1) on any node type
#   Gemma-4-31B    : 1 GPU (TP1) on any node type, FP8 weights
#   Qwen3-235B     : 1 RTX PRO 6000 node (8 GPUs, TP8)

: "${API_PORT:=8000}"

# Target nodes (hostnames only; override per reservation).
: "${GLM46_NODE:=holygpu8a10201}"     # GLM-4.6 FP8    -> 1 H200 node
: "${GLM52_HEAD:=holygpu8a10101}"     # GLM-5.2 FP8    -> H200 head
: "${GLM52_WORKER:=holygpu8a10102}"   # GLM-5.2 FP8    -> H200 worker
: "${RTX_NODE:=holygpu7c1713}"        # GLM-5.2 NVFP4  -> 1 RTX PRO 6000 node
: "${KIMI_NODE:=holygpu7c1734}"       # Kimi-K2.7-Code -> 1 RTX PRO 6000 node (primary)
: "${KIMI_HEAD:=holygpu8a10202}"      # Kimi-K2.7-Code -> H200 head   (2-node alternative)
: "${KIMI_WORKER:=holygpu8a10301}"    # Kimi-K2.7-Code -> H200 worker (2-node alternative)
: "${GEMMA4_NODE:=holygpu7c2313}"     # Gemma-4-26B/31B -> 1 GPU on any GPU node
: "${QWEN3_NODE:=holygpu7c1913}"      # Qwen3-235B      -> 1 full RTX PRO 6000 node

# Checkpoint paths. Override MODEL at launch to serve a copy elsewhere (e.g. VAST scratch).
: "${MODELS_DIR:=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models}"
: "${GLM46_MODEL:=$MODELS_DIR/GLM-4.6-FP8}"
: "${GLM52_MODEL:=$MODELS_DIR/GLM-5.2-FP8}"
: "${GLM52_NVFP4_MODEL:=$MODELS_DIR/GLM-5.2-NVFP4}"
: "${KIMI_MODEL:=$MODELS_DIR/Kimi-K2.7-Code}"
: "${GEMMA4_MODEL:=$MODELS_DIR/gemma-4-26B-A4B-it}"
: "${GEMMA4_DRAFT:=$MODELS_DIR/gemma-4-26B-A4B-it-assistant}"
: "${GEMMA31_MODEL:=$MODELS_DIR/gemma-4-31B-it}"
: "${GEMMA31_DRAFT:=$MODELS_DIR/gemma-4-31B-it-assistant}"
: "${QWEN3_MODEL:=$MODELS_DIR/Qwen3-235B-A22B}"
