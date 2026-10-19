#!/usr/bin/env bash
#
# setup.sh: download the checkpoint and drafter at their pinned revisions, check out the engine at its pinned commit
# and build it for sm_120. Idempotent; rerun after changing recipe.toml.
#
# Env: MODELS (./models), TF_SRC (./engine), ZIG (zig on PATH, 0.17.0), NVCC (/usr/local/cuda/bin/nvcc)
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
pin() { python3 -c "import tomllib, sys; print(tomllib.load(open('recipe.toml', 'rb'))[sys.argv[1]][sys.argv[2]])" "$@"; }

MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
ZIG="${ZIG:-zig}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

"$ZIG" version | grep -qx "$(pin engine zig)" || { echo "need Zig $(pin engine zig) (ZIG=...)" >&2; exit 1; }

hf download "$(pin model repo)" --revision "$(pin model revision)" --local-dir "$MODELS/Qwen3.8-27B-MLX-4bit"
hf download "$(pin model drafter)" --revision "$(pin model drafter_revision)" --local-dir "$MODELS/Qwen3.8-27B-DFlash2"

[ -d "$TF_SRC/.git" ] || git clone --branch "$(pin engine branch)" "$(pin engine repo)" "$TF_SRC"
git -C "$TF_SRC" fetch -q origin "$(pin engine branch)"
git -C "$TF_SRC" checkout -q "$(pin engine sha)"
SM="$(pin engine sm)"   # read here: recipe.toml is not in $TF_SRC, and an empty -Dsm builds for nvcc's default arch
(cd "$TF_SRC" && "$ZIG" build install native -Dnvcc="$NVCC" -Dsm="$SM" -Dversion=1.0.11)
echo "built $TF_SRC/zig-out/native/bin/tensorfold-native at $(git -C "$TF_SRC" rev-parse --short HEAD)"
