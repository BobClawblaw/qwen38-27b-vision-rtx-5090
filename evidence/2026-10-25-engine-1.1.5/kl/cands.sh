#!/usr/bin/env bash
# Each candidate checkpoint's logits for the truth's sequences (tf-qwen35-cuda step, bf16 caches), scored against the
# bf16 truth. Run under the 5090 lock; production stops for the run and starts again at the end.
set -u
S=$(dirname "$0")   # the folder holding score.py
R=/storage/qwen38-kl/run1
T=/storage/TensorFold-v1.1.5/zig-out/bin/tf-qwen35-cuda
export TENSORFOLD_CUDA_KERNELS=native
wait_free() { for _ in $(seq 1 60); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ] && return; sleep 2; done; }
trap 'wait_free; sudo -n systemctl start tensorfold-qwen38' EXIT
sudo -n systemctl stop tensorfold-qwen38
wait_free
for c in "ours /storage/tensorfold-qwen38-27b-5090/models/Qwen3.8-27B-MLX-4bit" \
         "lmstudio /storage/qwen38-kl/cands/lmstudio-community_Qwen3.8-27B-MLX-4bit" \
         "nvidia-nvfp4 /storage/qwen38-kl/cands/nvidia_Qwen3.8-27B-NVFP4" \
         "unsloth-nvfp4 /storage/qwen38-kl/cands/unsloth_Qwen3.8-27B-NVFP4"; do
  set -- $c
  for seq in chat code prose; do
    out=$R/$1.$seq.bin
    if ! $T "$2" step "$(cat $R/$seq.ids)" "$out" > $R/$1.$seq.log 2>&1; then
      echo "$1 $seq: FAILED: $(tail -2 $R/$1.$seq.log | tr '\n' ' ')"; break
    fi
    echo "$1 $seq: $(/storage/qwen38-kl/venv/bin/python -I $S/score.py $R/$seq.truth.npy $out $R/$seq.ids $R/$1.$seq.json)"
    rm -f "$out"
  done
done
