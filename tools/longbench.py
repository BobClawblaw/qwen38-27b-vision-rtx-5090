"""Long-prompt prefill and decode speed of an OpenAI-compatible server, plus a needle check.

Each prompt starts with a fresh nonce so no prefix cache can serve it. The prompt is source
code with one passphrase line at mid-depth; the reply (greedy, thinking off, ignore_eos) must
contain the passphrase. Prefill = time to first token; decode = later tokens / later time.
"""
import argparse, glob, json, os, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("model")
ap.add_argument("--sizes", default="32000,100000,200000", help="approximate prompt tokens")
ap.add_argument("--tokens", type=int, default=512)
ap.add_argument("--label", default="")
ap.add_argument("--output")
args = ap.parse_args()

files = sorted(glob.glob(os.path.join(os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")), "zig/src/**/*.zig"), recursive=True))
corpus = ""
for f in files:
    corpus += f"\n// ===== {os.path.basename(f)} =====\n" + open(f).read()


def run(size):
    word = uuid.uuid4().hex[:10]
    chars = int(size * 3.2)
    body = corpus
    while len(body) < chars:
        body += corpus
    body = body[:chars]
    mid = body.rfind("\n", 0, len(body) // 2)
    body = body[:mid] + f"\n// The secret passphrase is {word}.\n" + body[mid:]
    msg = (f"[run {uuid.uuid4()}]\nHere is a large source dump.\n\n{body}\n\n"
           "Somewhere in the dump is a comment giving a secret passphrase. Reply with the passphrase, "
           "then summarize what the code does.")
    req = {"model": args.model, "messages": [{"role": "user", "content": msg}],
           "max_tokens": args.tokens, "temperature": 0, "stream": True, "ignore_eos": True,
           "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(args.base + "/v1/chat/completions", json.dumps(req).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = None; n_chunks = 0
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    if first is None:
                        first = time.time()
                    text += piece; n_chunks += 1
    end = time.time()
    out_toks = usage["completion_tokens"] if usage else n_chunks
    res = {"label": args.label, "target": size, "prompt_tokens": usage and usage["prompt_tokens"],
           "completion_tokens": out_toks, "prefill_s": round(first - t0, 2),
           "prefill_tps": round((usage["prompt_tokens"] if usage else 0) / (first - t0), 1),
           "decode_tps": round((out_toks - 1) / (end - first), 1), "needle_found": word in text,
           "reply_head": text[:120]}
    print(json.dumps(res), flush=True)
    return res


results = [run(int(s)) for s in args.sizes.split(",")]
if args.output:
    json.dump(results, open(args.output, "w"), indent=1)
