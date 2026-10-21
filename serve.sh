#!/usr/bin/env bash
#
# serve.sh: serve Qwen3.8-27B with image input on TensorFold native (CUDA) on one RTX 5090. Foreground; the systemd
# unit runs it this way.
#
# Knobs (env): PORT (30000), HOST (0.0.0.0), CONTEXT (262144), PARALLEL (auto), NAME (qwen38-27b), ALIAS
#   (qwen38-nvfp4), KV_CACHE (fp8: e4m3 attention caches with one scale a row, 262k with vision in one stream;
#   bf16: half the window, set CONTEXT=131072 with it), MODELS (./models), TF_SRC (./engine)
#   PROMPT_CACHE_GIB (8: host memory for kept prompt states, at most half of MemAvailable; 0 turns the cache off)
#   MAX_TOKENS (the engine's default: the output cap when a request sends no max_tokens)
#   THINKING_BUDGET (the engine's default: a cap on reasoning tokens a reply)
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
# the engine's own CUDA kernels (no captured glue kernels with this engine)
export TENSORFOLD_CUDA_KERNELS="${TENSORFOLD_CUDA_KERNELS:-native}"
# structured output needs libtfgrammar beside the binary (setup.sh builds it); the engine loads it on the first
# structured request and keeps a miss until restart, so refuse to start without it
[ -f "$TF_SRC/zig-out/native/lib/libtfgrammar.so" ] || { echo "no $TF_SRC/zig-out/native/lib/libtfgrammar.so: rerun setup.sh" >&2; exit 1; }
exec "$TF_SRC/zig-out/native/bin/tensorfold-native" serve "$MODELS/Qwen3.8-27B-MLX-4bit" \
  --drafter "$MODELS/Qwen3.8-27B-DFlash2" \
  --name "${NAME:-qwen38-27b}" --alias "${ALIAS:-qwen38-nvfp4}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-30000}" \
  --context "${CONTEXT:-262144}" --parallel "${PARALLEL:-auto}" \
  --kv-cache "${KV_CACHE:-fp8}" \
  ${MAX_TOKENS:+--max-tokens "$MAX_TOKENS"} ${THINKING_BUDGET:+--thinking-budget "$THINKING_BUDGET"} \
  --vision ${PROMPT_CACHE_GIB:+--prompt-cache-gib "$PROMPT_CACHE_GIB"}
