"""Image chat requests against a server: greedy, thinking off; prints and saves each reply (argv: port, out.json)."""
import os, base64, json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
D = os.path.dirname(os.path.abspath(__file__)) + "/images/"
def url(p):
    mime = "image/jpeg" if p.endswith(".jpg") else "image/png"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(open(D + p, "rb").read()).decode()}}
def low(p):
    u = url(p); u["image_url"]["detail"] = "low"; return u
cases = {
  "shapes-read": [{"role": "user", "content": [url("vision-test.png"), {"type": "text", "text": "Describe the shapes and their colors, and read the text in the image exactly."}]}],
  "photo-jpg": [{"role": "user", "content": [url("photo.jpg"), {"type": "text", "text": "What kind of image is this? Describe its colors and patterns in two sentences."}]}],
  "two-images": [{"role": "user", "content": [{"type": "text", "text": "Image A:"}, url("logo.png"), {"type": "text", "text": "Image B:"}, url("rotated.jpg"), {"type": "text", "text": "What is in image A and in image B? One sentence each."}]}],
  "low-detail": [{"role": "user", "content": [low("large.png"), {"type": "text", "text": "Describe this image briefly."}]}],
  "followup": [{"role": "user", "content": [url("vision-test.png"), {"type": "text", "text": "What number is written in the image?"}]},
               {"role": "assistant", "content": "The number written in the image is 7351."},
               {"role": "user", "content": "Multiply that number by 2 and give only the result."}],
}
res = {}
for name, msgs in cases.items():
    body = json.dumps({"model": "qwen38-27b", "messages": msgs, "max_tokens": 160, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600))
    dt = time.time() - t
    if "error" in r: print(name, "ERROR", r["error"]); res[name] = {"error": r["error"]}; continue
    text = r["choices"][0]["message"]["content"]; u = r.get("usage", {})
    res[name] = {"text": text, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"), "seconds": round(dt, 2)}
    print(f"--- {name}: prompt {u.get('prompt_tokens')} tokens, {u.get('completion_tokens')} out, {dt:.1f}s\n{text}\n", flush=True)
json.dump(res, open(out, "w"), indent=1)
