"""Reference next-token logits from the bf16 Qwen3.8-27B on CPU, for KL against quantized checkpoints.

Builds three sequences (a chat turn in the model's own template, a Python source file, technical prose), writes their
ids as comma lists (tf-qwen35-cuda's input) and each sequence's fp32 logits as .npy (rows x vocab).
usage: truth.py <bf16 model dir> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROWS = 700  # tokens a sequence

model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_dir)

alone = json.load(open(sys.argv[3]))["details"]["alone"] if len(sys.argv) > 3 else []
chat = []
for a in alone:
    chat += [{"role": "user", "content": a["prompt"]}, {"role": "assistant", "content": a["text"]}]
texts = {
    "chat": tok.apply_chat_template(chat, tokenize=False, enable_thinking=False),
    "code": Path("/usr/lib/python3.12/json/decoder.py").read_text(),
    "prose": Path("/storage/TensorFold-v1.1.5/docs/kernel-contract.md").read_text(),
}
seqs = {}
for name, text in texts.items():
    ids = tok(text, add_special_tokens=False)["input_ids"][:ROWS]
    if len(ids) < ROWS:
        raise SystemExit(f"{name}: only {len(ids)} tokens")
    seqs[name] = ids
    (out / f"{name}.ids").write_text(",".join(map(str, ids)))

torch.set_num_threads(32)
model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cpu")
model.eval()
for name, ids in seqs.items():
    with torch.inference_mode():
        h = model.model(input_ids=torch.tensor([ids])).last_hidden_state[0]
        # the head in fp32 over the bf16 hidden rows: the truth's own rounding stays below the candidates'
        logits = h.float() @ model.lm_head.weight.float().T
    np.save(out / f"{name}.truth.npy", logits.numpy().astype(np.float32))
    print(name, len(ids), logits.shape, flush=True)
