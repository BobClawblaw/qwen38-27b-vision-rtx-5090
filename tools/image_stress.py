import os, base64, json, sys, time, threading, urllib.request
D=os.path.dirname(os.path.abspath(__file__))+"/images/"; port=sys.argv[1]
def img(p, mime="image/png"): return {"type":"image_url","image_url":{"url":f"data:{mime};base64,"+base64.b64encode(open(D+p,"rb").read()).decode()}}
def ask(msgs, max_tokens=80, think=False):
    body=json.dumps({"model":"qwen38-27b","messages":msgs,"max_tokens":max_tokens,"temperature":0,"chat_template_kwargs":{"enable_thinking":think}}).encode()
    t=time.time()
    try: r=json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=900))
    except urllib.error.HTTPError as e: return {"http": e.code, "body": e.read().decode()[:300]}, time.time()-t
    return r, time.time()-t
def show(name, r, dt):
    if "choices" not in r: print(f"{name}: REFUSED/ERROR {r}"); return
    m=r["choices"][0]["message"]; print(f"{name}: prompt {r['usage']['prompt_tokens']} {dt:.1f}s -> {m['content'][:200]!r}" + (f" [reasoning {len(m.get('reasoning_content') or '')} chars]" if m.get('reasoning_content') else ""))
filler = " ".join(f"item{i}" for i in range(1400))
show("chunk-boundary", *ask([{"role":"user","content":[{"type":"text","text":"Ignore this list: "+filler},img("large.png"),{"type":"text","text":"Describe the image in one sentence."}]}]))
show("max-image", *ask([{"role":"user","content":[img("max.png"),{"type":"text","text":"How many colored vertical bars are there, and what text is in the middle?"}]}]))
show("thinking-on", *ask([{"role":"user","content":[img("vision-test.png"),{"type":"text","text":"What number is in the image?"}]}], max_tokens=600, think=True))
show("four-images", *ask([{"role":"user","content":[img("logo.png"),img("tiny.png"),img("gray.png"),img("vision-test.png"),{"type":"text","text":"Describe each of the four images in a few words, numbered."}]}], max_tokens=120))
show("five-images", *ask([{"role":"user","content":[img("logo.png")]*5+[{"type":"text","text":"hi"}]}]))
# concurrency: text replies must equal their solo runs while an image request shares the rounds
texts=["Write a haiku about rivers.","List three prime numbers above 100.","Explain what a hash map is in one sentence."]
solo=[ask([{"role":"user","content":t}])[0]["choices"][0]["message"]["content"] for t in texts]
res={}
def run(k, msgs): res[k]=ask(msgs)[0]
ths=[threading.Thread(target=run,args=(i,[{"role":"user","content":t}])) for i,t in enumerate(texts)]
ths.append(threading.Thread(target=run,args=("img",[{"role":"user","content":[img("photo.jpg","image/jpeg"),{"type":"text","text":"Describe this image in one sentence."}]}])))
for t in ths: t.start()
for t in ths: t.join()
for i,t in enumerate(texts): print(f"concurrent text {i}: {'EQUALS solo' if res[i]['choices'][0]['message']['content']==solo[i] else 'DIFFERS from solo'}")
show("concurrent image", res["img"], 0)
