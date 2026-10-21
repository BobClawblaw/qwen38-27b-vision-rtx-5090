#!/usr/bin/env bash
#
# setup.sh: download the checkpoint and drafter at their pinned revisions, check out the engine at its pinned commit
# and build it for sm_120, with libtfgrammar (structured output) and the pinned JPEG and WebP decoders beside it.
# Idempotent; rerun after changing recipe.toml.
#
# Env: MODELS (./models), TF_SRC (./engine), XGRAMMAR (./xgrammar), DECODERS (./decoders: libjpeg-turbo and libwebp
#   sources), ZIG (zig on PATH, 0.17.0), NVCC (/usr/local/cuda/bin/nvcc). Needs cmake and nasm for the decoders.
#
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
pin() { python3 -c "import tomllib, sys; print(tomllib.load(open('recipe.toml', 'rb'))[sys.argv[1]][sys.argv[2]])" "$@"; }

MODELS="${MODELS:-$PWD/models}"
TF_SRC="${TF_SRC:-$PWD/engine}"
XGRAMMAR="${XGRAMMAR:-$PWD/xgrammar}"
DECODERS="${DECODERS:-$PWD/decoders}"
ZIG="${ZIG:-zig}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

"$ZIG" version | grep -qx "$(pin engine zig)" || { echo "need Zig $(pin engine zig) (ZIG=...)" >&2; exit 1; }

hf download "$(pin model repo)" --revision "$(pin model revision)" --local-dir "$MODELS/Qwen3.8-27B-MLX-4bit"
hf download "$(pin model drafter)" --revision "$(pin model drafter_revision)" --local-dir "$MODELS/Qwen3.8-27B-DFlash2"

[ -d "$TF_SRC/.git" ] || git clone --branch "$(pin engine branch)" "$(pin engine repo)" "$TF_SRC"
git -C "$TF_SRC" fetch -q origin "$(pin engine branch)"
git -C "$TF_SRC" checkout -q "$(pin engine sha)"
# read here: recipe.toml is not in $TF_SRC, and an empty -Dsm builds for nvcc's default arch
SM="$(pin engine sm)"
VERSION="$(pin engine version)"
(cd "$TF_SRC" && "$ZIG" build native -Dnvcc="$NVCC" -Dsm="$SM" -Dversion="$VERSION")

# libtfgrammar: without it the server refuses response_format, the guided_* fields and tool_choice required or a
# named function. The binary looks in ../lib from zig-out/native/bin; on Linux `zig build grammar` installs into
# zig-out/lib, so the library is copied across.
[ -d "$XGRAMMAR/.git" ] || git clone -q --branch "$(pin grammar tag)" --recurse-submodules "$(pin grammar repo)" "$XGRAMMAR"
(cd "$TF_SRC" && "$ZIG" build grammar -Dxgrammar="$XGRAMMAR" -Dsm="$SM")
mkdir -p "$TF_SRC/zig-out/native/lib"
cp "$TF_SRC/zig-out/lib/libtfgrammar.so" "$TF_SRC/zig-out/native/lib/libtfgrammar.so"

# Image decoders: the engine decodes JPEG with libjpeg-turbo's libturbojpeg and WebP with libwebp's libwebpdecoder,
# at the releases its pixels are pinned to. It checks the version and refuses JPEG or WebP images with any other
# (distributions ship older ones); PNG needs neither. Both are built from source into the same lib folder.
JPEG_TAG="$(pin decoders jpeg_tag)"
[ -d "$DECODERS/libjpeg-turbo/.git" ] || git clone -q --depth 1 --branch "$JPEG_TAG" "$(pin decoders jpeg_repo)" "$DECODERS/libjpeg-turbo"
cmake -S "$DECODERS/libjpeg-turbo" -B "$DECODERS/libjpeg-turbo/build" -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=OFF -DWITH_JAVA=OFF > /dev/null
cmake --build "$DECODERS/libjpeg-turbo/build" -j"$(nproc)" --target turbojpeg > /dev/null
cp -L "$DECODERS/libjpeg-turbo/build/libturbojpeg.so.0" "$TF_SRC/zig-out/native/lib/libturbojpeg.so.0"
[ -d "$DECODERS/libwebp/.git" ] || git clone -q --depth 1 --branch "$(pin decoders webp_tag)" "$(pin decoders webp_repo)" "$DECODERS/libwebp"
cmake -S "$DECODERS/libwebp" -B "$DECODERS/libwebp/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF \
  -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF \
  -DWEBP_BUILD_EXTRAS=OFF > /dev/null
cmake --build "$DECODERS/libwebp/build" -j"$(nproc)" --target webpdecoder > /dev/null
cp -L "$DECODERS/libwebp/build/libwebpdecoder.so.3" "$TF_SRC/zig-out/native/lib/libwebpdecoder.so.3"

echo "built $TF_SRC/zig-out/native/bin/tensorfold-native at $(git -C "$TF_SRC" rev-parse --short HEAD), with" \
  "libtfgrammar, libturbojpeg $JPEG_TAG and libwebpdecoder $(pin decoders webp_tag) in zig-out/native/lib"
