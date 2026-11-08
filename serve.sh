#!/usr/bin/env bash
#
# serve.sh: serve Qwen3.8-27B with image input on TensorFold native (CUDA) on one RTX 5090. Foreground; the systemd
# unit runs it this way.
#
# Knobs (env): PORT (30000), HOST (0.0.0.0), CONTEXT (262144), PARALLEL (auto: three streams at 262144 with the EXL3
#   pack, four with the MLX 4-bit), NAME (qwen38-27b), ALIAS (qwen38-nvfp4), KV_CACHE (fp8: e4m3 attention caches with
#   one scale a row, 262k with vision; bf16: half the window, set CONTEXT=131072 with it), MODELS (./models),
#   TF_SRC (./engine), CHECKPOINT (Qwen3.8-27B-exl3-4.00bpw: the folder under MODELS to serve; Qwen3.8-27B-MLX-4bit
#   after `CHECKPOINT=mlx_4bit ./setup.sh`)
#   PROMPT_PRECISION (unset: the engine's bf16 prompt path; int8, with the MLX 4-bit checkpoint only: prompt matmuls on
#   int8 activations, a 64-input group a scale, 20-26% shorter prefill)
#   PROMPT_CACHE_GIB (8: host memory for kept prompt states, at most half of MemAvailable; 0 turns the cache off)
#   MAX_TOKENS (65536: the output cap when a request sends no max_tokens; it must hold a reply's thinking and its text)
#   THINKING_BUDGET (49152: a cap on reasoning tokens a reply; 8192 cut GSM8K with thinking to 222/250, this gives 242-246)
#   INSTRUCT_SAMPLING (unset: requests with thinking off start from the model card's instruct set, temperature 0.7, top_p
#   0.8, top_k 20, presence_penalty 1.5; name=value pairs change it, `none` serves the thinking set to every request)
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
CHECKPOINT="${CHECKPOINT:-Qwen3.8-27B-exl3-4.00bpw}"
# the engine's own CUDA kernels (no captured glue kernels with this engine)
export TENSORFOLD_CUDA_KERNELS="${TENSORFOLD_CUDA_KERNELS:-native}"
# streams beside the first take 256-position pages of one pool (0: one stream at the full window, a window each)
export TF_QWEN_KV_PAGES="${TF_QWEN_KV_PAGES:-256}"
# structured output needs libtfgrammar beside the binary (setup.sh builds it); the engine loads it on the first
# structured request and keeps a miss until restart, so refuse to start without it
[ -f "$TF_SRC/zig-out/native/lib/libtfgrammar.so" ] || { echo "no $TF_SRC/zig-out/native/lib/libtfgrammar.so: rerun setup.sh" >&2; exit 1; }
[ -f "$MODELS/$CHECKPOINT/config.json" ] || { echo "no checkpoint at $MODELS/$CHECKPOINT: run setup.sh (CHECKPOINT=mlx_4bit for the MLX 4-bit)" >&2; exit 1; }
exec "$TF_SRC/zig-out/native/bin/tensorfold-native" serve "$MODELS/$CHECKPOINT" \
  --drafter "$MODELS/Qwen3.8-27B-DFlash2" \
  --name "${NAME:-qwen38-27b}" --alias "${ALIAS:-qwen38-nvfp4}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-30000}" \
  --context "${CONTEXT:-262144}" --parallel "${PARALLEL:-auto}" \
  --kv-cache "${KV_CACHE:-fp8}" ${PROMPT_PRECISION:+--prompt-precision "$PROMPT_PRECISION"} \
  --max-tokens "${MAX_TOKENS:-65536}" --thinking-budget "${THINKING_BUDGET:-49152}" \
  ${INSTRUCT_SAMPLING:+--instruct-sampling "$INSTRUCT_SAMPLING"} \
  --vision ${PROMPT_CACHE_GIB:+--prompt-cache-gib "$PROMPT_CACHE_GIB"}
