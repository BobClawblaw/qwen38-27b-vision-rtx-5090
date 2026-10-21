"""Fixed source-code prompts (no nonce) of about 35k and 110k tokens, 512 greedy tokens each, thinking off.

Unlike longbench.py these prompts are the same every run, so two engines answer them alike and their times compare.
Run it against servers with the prompt cache off (PROMPT_CACHE_GIB=0) so each request prefills in full.
argv: URL MODEL OUT; TF_SRC names the TensorFold checkout whose zig/src is the prompt text (default ./engine).
"""
import glob, json, os, sys, time, urllib.request

url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
src = os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
text = ""
for f in sorted(glob.glob(os.path.join(src, "zig/src/**/*.zig"), recursive=True)):
    text += f"\n// ==== {os.path.basename(f)} ====\n" + open(f, errors="ignore").read()
if not text:
    sys.exit(f"no .zig sources under {src}/zig/src: set TF_SRC to a TensorFold checkout")
res = []
for chars in (108000, 335000):
    req = {"model": model, "messages": [{"role": "user", "content": text[:chars] + "\n\nExplain what the code above does, file by file."}],
           "max_tokens": 512, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                                                                {"Content-Type": "application/json"}), timeout=1800))
    res.append({"prompt_tokens": r["usage"]["prompt_tokens"], "completion_tokens": r["usage"]["completion_tokens"],
                "seconds": round(time.time() - t, 2), "text": r["choices"][0]["message"]["content"]})
    print(res[-1]["prompt_tokens"], res[-1]["completion_tokens"], res[-1]["seconds"], flush=True)
json.dump(res, open(out, "w"), indent=1)
