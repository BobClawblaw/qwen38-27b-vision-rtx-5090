"""Greedy 256-token replies, each asked twice, for prompts across the 4,096-row grid, a repeated prompt and a 3-turn chat.
argv: URL OUT.json. Prompts are deterministic, so two servers' outputs compare directly."""
import glob, json, os, sys, urllib.request

url, out = sys.argv[1], sys.argv[2]
corpus = ""
src = os.environ.get("TF066_SRC", "TensorFold-v0.6.6")  # a TensorFold 0.6.6 checkout: its sources make the prompts
for f in sorted(glob.glob(os.path.join(src, "src/tensorfold/**/*.py"), recursive=True)):
    corpus += f"\n# ===== {f.split('/src/')[1]} =====\n" + open(f).read()
while len(corpus) < 300000:
    corpus += corpus

def ask(messages, logprobs):
    req = {"model": "qwen38-27b", "messages": messages, "max_tokens": 256, "temperature": 0,
           "chat_template_kwargs": {"enable_thinking": False}}
    if os.environ.get("LOGPROBS"):
        req |= {"logprobs": True, "top_logprobs": 5}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=3600) as resp:
        d = json.load(resp)
    ch = d["choices"][0]
    lp = [(t["token"], t["logprob"], [(a["token"], a["logprob"]) for a in t["top_logprobs"]])
          for t in (ch.get("logprobs") or {}).get("content") or []]
    return {"text": ch["message"]["content"], "logprobs": lp, "usage": d.get("usage")}

res = {}
def run(name, messages):
    for lp in (True, False):
        k = f"{name}{'' if lp else ' (again)'}"
        res[k] = ask(messages, lp)
        u = res[k]["usage"] or {}
        print(k, u.get("prompt_tokens"), (u.get("prompt_tokens_details") or {}).get("cached_tokens"), repr(res[k]["text"][:60]), flush=True)

for chars in (6000, 15000, 28000, 45000, 100000, 200000):
    run(f"single {chars}", [{"role": "user", "content": f"[case {chars}]\n{corpus[:chars]}\n\nExplain in detail, function by function, what the last file above does."}])
run("repeat 200000", [{"role": "user", "content": f"[case 200000]\n{corpus[:200000]}\n\nExplain in detail, function by function, what the last file above does."}])
msgs = [{"role": "user", "content": f"[chat]\n{corpus[:90000]}\n\nWalk through the first file above in detail."}]
for t, q in enumerate(["Now walk through the second file in detail.", "Compare the two files' styles in detail.", None]):
    run(f"chat turn {t + 1}", msgs)
    if q:
        msgs = msgs + [{"role": "assistant", "content": res[f"chat turn {t + 1}"]["text"]}, {"role": "user", "content": q}]
json.dump(res, open(out, "w"), indent=1)
