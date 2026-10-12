"""Needle in a haystack against an OpenAI-compatible server: a passphrase hidden at several depths of source text."""
import os, glob, json, sys, time, urllib.request
from transformers import AutoTokenizer
url, model_dir, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
depths = [float(x) for x in sys.argv[4].split(",")]
tok = AutoTokenizer.from_pretrained(model_dir)
text = ""
for f in sorted(glob.glob(os.path.join(os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")), "zig/src/**/*.zig"), recursive=True)):
    text += f"\n// ==== {f} ====\n" + open(f, errors="ignore").read()
ids = tok(text, add_special_tokens=False)["input_ids"][:total]
for depth in depths:
    at = int(len(ids) * depth)
    needle = f"\n// NOTE: the deployment passphrase is TANGERINE-{4471 + int(depth * 100)}. Remember it.\n"
    body = tok.decode(ids[:at]) + needle + tok.decode(ids[at:])
    q = "What is the deployment passphrase mentioned in a NOTE comment somewhere in the code above? Reply with the passphrase only."
    req = {"model": "qwen38-27b", "messages": [{"role": "user", "content": body + "\n\n" + q}], "max_tokens": 64,
           "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=3600))
    want = f"TANGERINE-{4471 + int(depth * 100)}"
    got = r["choices"][0]["message"]["content"].strip()
    print(f"depth {depth:.2f}: prompt {r['usage']['prompt_tokens']} tokens, {time.time() - t0:.1f}s -> {got!r} {'OK' if want in got else 'MISS (want ' + want + ')'}", flush=True)
