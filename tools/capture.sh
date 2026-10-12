#!/usr/bin/env bash
#
# tools/capture.sh: capture the Triton glue kernels the native qwen27 CUDA engine
# needs (TENSORFOLD_CUDA_KERNELS) from TensorFold 0.6.6 (a Python venv with it installed at .venv), plus the
# 0.6.6 reference outputs the native engine is checked against.
#
# Needs the GPU to itself: stop the server first.
#   captures/<run>/          each run's launches, manifest, reference outputs
#   captures/<run>/triton/   that run's TRITON_CACHE_DIR
#   kernels/sm120/           aot.json + cubins/ (the packed kernel set)
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${TF_SRC:-$SCRIPT_DIR/engine}"
PY="$SCRIPT_DIR/.venv/bin/python"
MODEL="$SCRIPT_DIR/models/Qwen3.8-27B-MLX-4bit"
DRAFT="$SCRIPT_DIR/models/Qwen3.8-27B-DFlash2"
CAP="$SCRIPT_DIR/captures"
T="$SRC/zig/tests/cuda/qwen27"
TOOLS="$SRC/tools/zig"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"

run() { # name, script, args...
  local name=$1 script=$2; shift 2
  mkdir -p "$CAP/$name/triton"
  echo "== $name"
  TRITON_CACHE_DIR="$CAP/$name/triton" "$PY" "$T/$script" --model "$MODEL" --out "$CAP/$name" --tools "$TOOLS" "$@" \
    2>&1 | tee "$CAP/$name/run.log"
  "$PY" "$TOOLS/triton_aot_manifest.py" --cache "$CAP/$name/triton" --launches "$CAP/$name/launches.json" \
    --out "$CAP/$name/manifest.json"
}

run serial  capture.py
run windows capture_windows.py --wide --sweep
run multi   capture_multi.py --draft "$DRAFT"
run draft   capture_draft.py --draft "$DRAFT" --prompts "$CAP/serial/prompts.json"

# --kv-cache fp8's attention kernels (attention_kv8.py checks them and records their launches; no model needed)
mkdir -p "$CAP/kv8/triton"
echo "== kv8"
TRITON_CACHE_DIR="$CAP/kv8/triton" "$PY" "$T/attention_kv8.py" --out "$CAP/kv8" --tools "$TOOLS" 2>&1 | tee "$CAP/kv8/run.log"
"$PY" "$TOOLS/triton_aot_manifest.py" --cache "$CAP/kv8/triton" --launches "$CAP/kv8/launches.json" \
  --out "$CAP/kv8/manifest.json"

"$PY" "$SCRIPT_DIR/tools/jit_all.py" "$T" "$CAP/jit.json"
"$PY" -c 'import json, sys; a = json.load(open(sys.argv[1])); a.update(json.load(open(sys.argv[2]))); json.dump(a, open(sys.argv[1], "w"), indent=1, sort_keys=True)' "$CAP/jit.json" "$CAP/kv8/jit.json"
rm -rf "$SCRIPT_DIR/kernels/sm120"
args=()
for n in serial windows multi draft kv8; do args+=(--manifest "$CAP/$n/manifest.json" --cache "$CAP/$n/triton"); done
"$PY" "$SRC/zig/tests/cuda/nemotron/aot_pack.py" "${args[@]}" --jit "$CAP/jit.json" --out "$SCRIPT_DIR/kernels/sm120"
