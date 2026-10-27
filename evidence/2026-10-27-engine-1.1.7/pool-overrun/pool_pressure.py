"""Four concurrent long greedy requests that overrun the shared page pool, each compared with the same request alone."""
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"
SRC = pathlib.Path(os.environ.get("TF_SRC", "engine")) / "zig/src"  # the v1.1.7 checkout
CHARS = 150_000  # about 45k tokens of source each


def corpus(i: int) -> str:
    files = sorted(p for p in SRC.rglob("*.zig"))[i * 120:]
    out, n = [], 0
    for p in files:
        t = p.read_text(errors="replace")
        out.append(f"// {p.name}\n{t}")
        n += len(t)
        if n >= CHARS:
            break
    return "".join(out)[:CHARS]


def ask(prompt: str) -> dict:
    body = {"model": "qwen38-27b", "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt + "\n\nName the first three Zig files above, one per line."}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        text = d["choices"][0]["message"]["content"]
        return {"ok": True, "sha": hashlib.sha256(text.encode()).hexdigest()[:12], "prompt_tokens": d["usage"]["prompt_tokens"], "s": round(time.time() - t, 1), "text": text[:80]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "body": e.read()[:300].decode(errors="replace"), "s": round(time.time() - t, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)[:300], "s": round(time.time() - t, 1)}


prompts = [corpus(i) for i in range(4)]
print("together:")
with cf.ThreadPoolExecutor(4) as ex:
    together = list(ex.map(ask, prompts))
for i, r in enumerate(together):
    print(i, r)
print("alone:")
alone = [ask(p) for p in prompts]
for i, r in enumerate(alone):
    print(i, r)
same = sum(1 for a, b in zip(together, alone) if a.get("ok") and b.get("ok") and a["sha"] == b["sha"])
print(f"together == alone: {same}/4; errors together: {sum(not r['ok'] for r in together)}")
