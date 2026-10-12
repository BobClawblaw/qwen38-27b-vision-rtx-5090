"""A long conversation's next turns against an OpenAI-compatible server: first-token time and cached tokens a turn.

Turn 1 sends a large source dump with a question; each later turn sends the whole history back plus a new question,
as a chat client does. With a prompt cache a later turn resumes the kept state and reads only what is new.
argv: URL MODEL [approximate prompt tokens, default 100000] [turns, default 3]
"""
import glob, json, os, sys, time, urllib.request, uuid

url, model = sys.argv[1], sys.argv[2]
size = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
turns = int(sys.argv[4]) if len(sys.argv) > 4 else 3
src = os.environ.get("TF_SRC", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine"))
corpus = ""
for f in sorted(glob.glob(os.path.join(src, "zig/src/**/*.zig"), recursive=True)):
    corpus += f"\n// ===== {os.path.basename(f)} =====\n" + open(f).read()
while len(corpus) < size * 3.2:
    corpus += corpus
questions = ["Which file defines the HTTP server's routes? One sentence.",
             "Name one CUDA kernel file the code loads. One sentence.",
             "What does the prompt cache key its states by? One sentence.",
             "Which allocator does the engine use for host memory? One sentence."]
messages = [{"role": "user", "content": f"[run {uuid.uuid4()}]\n{corpus[:int(size * 3.2)]}\n\n{questions[0]}"}]
for t in range(turns):
    req = {"model": model, "messages": messages, "max_tokens": 48, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = {}
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            c = json.loads(line[5:])
            usage = c.get("usage") or usage
            for ch in c.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece and first is None:
                    first = time.time()
                text += piece
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(json.dumps({"turn": t + 1, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
                      "first_token_s": round(first - t0, 2), "reply": text[:80]}), flush=True)
    messages += [{"role": "assistant", "content": text}, {"role": "user", "content": questions[(t + 1) % len(questions)]}]
