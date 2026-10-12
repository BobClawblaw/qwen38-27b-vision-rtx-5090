"""At a long prompt: prefill (time to first streamed token) and decode rate (streamed tokens after the first), greedy."""
import os, glob, json, sys, time, urllib.request
from transformers import AutoTokenizer
url, model_dir, total, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model_dir)
text = ""
for f in sorted(glob.glob(os.path.join(os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")), "zig/src/**/*.zig"), recursive=True)):
    text += f"\n// ==== {f} ====\n" + open(f, errors="ignore").read()
body = tok.decode(tok(text, add_special_tokens=False)["input_ids"][:total])
req = {"model": "qwen38-27b", "messages": [{"role": "user", "content": body + "\n\nWrite a long, detailed walkthrough of the last file above."}],
       "max_tokens": n, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
       "chat_template_kwargs": {"enable_thinking": False}}
t0 = time.time()
first = last = None
chunks = 0
usage = {}
with urllib.request.urlopen(urllib.request.Request(url, json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("choices") and ev["choices"][0].get("delta", {}).get("content"):
            now = time.time()
            first = first or now
            last = now
toks = usage.get("completion_tokens", 0)
print(f"prompt {usage.get('prompt_tokens')}: prefill {first - t0:.1f}s ({usage.get('prompt_tokens', 0) / (first - t0):.0f} tok/s); "
      f"decode {toks} tokens at {(toks - 1) / (last - first):.1f} tok/s")
