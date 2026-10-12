"""Prompts for tf-qwen27-cuda kvcompare: the serial captures' short ones and long ones built from source text."""
import os, json, sys, glob
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = json.load(open(sys.argv[2]))
files = sorted(glob.glob(os.path.join(os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")), "zig/src/**/*.zig"), recursive=True))
text = ""
for f in files:
    text += f"\n// ==== {f} ====\n" + open(f, errors="ignore").read()
def chat(body, q):
    msgs = [{"role": "user", "content": body + "\n\n" + q}]
    r = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return r["input_ids"] if hasattr(r, "keys") else r
for n in [int(x) for x in sys.argv[4].split(",")]:
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    body = tok.decode(ids)
    out[f"src{n // 1000}k"] = list(chat(body, "Explain in detail what the last file above does, function by function."))
json.dump(out, open(sys.argv[3], "w"))
print({k: len(v) for k, v in out.items()})
