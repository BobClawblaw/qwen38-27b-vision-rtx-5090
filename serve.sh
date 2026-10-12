#!/usr/bin/env bash
#
# serve.sh: serve Qwen3.8-27B with image input on TensorFold native (CUDA) on one RTX 5090. Foreground; the systemd
# unit runs it this way.
#
# Knobs (env): PORT (30000), HOST (0.0.0.0), CONTEXT (262144), PARALLEL (auto), NAME (qwen38-27b), ALIAS
#   (qwen38-nvfp4), KV_CACHE (fp8: e4m3 attention caches with one scale a row, 262k with vision in one stream;
#   bf16: bit-exact against TensorFold 0.6.6, set CONTEXT=122880 with it), MODELS (./models), TF_SRC (./engine)
#   PROMPT_CACHE_GIB (8: host memory for kept prompt states, at most half of MemAvailable; 0 turns the cache off
#   and keeps 0.6.6's even prefill chunks)
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
export TENSORFOLD_CUDA_KERNELS="${TENSORFOLD_CUDA_KERNELS:-$PWD/kernels/sm120}"
exec "$TF_SRC/zig-out/native/bin/tensorfold-native" serve "$MODELS/Qwen3.8-27B-MLX-4bit" \
  --drafter "$MODELS/Qwen3.8-27B-DFlash2" \
  --name "${NAME:-qwen38-27b}" --alias "${ALIAS:-qwen38-nvfp4}" \
  --host "${HOST:-0.0.0.0}" --port "${PORT:-30000}" \
  --context "${CONTEXT:-262144}" --parallel "${PARALLEL:-auto}" \
  --kv-cache "${KV_CACHE:-fp8}" \
  --vision ${PROMPT_CACHE_GIB:+--prompt-cache-gib "$PROMPT_CACHE_GIB"}
