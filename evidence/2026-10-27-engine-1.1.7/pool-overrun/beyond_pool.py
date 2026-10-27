import concurrent.futures as cf, json, os, pathlib, time, urllib.request, urllib.error
URL="http://127.0.0.1:30000/v1/chat/completions"
files=sorted(pathlib.Path(os.environ.get("TF_SRC", "engine")) / "zig/src".rglob("*.zig"))
def text(start, chars):
    out=[];n=0
    for p in files[start:]:
        t=p.read_text(errors="replace"); out.append(f"// {p.name}\n{t}"); n+=len(t)
        if n>=chars: break
    return "".join(out)[:chars]
def ask(prompt, mt, delay=0):
    time.sleep(delay)
    body={"model":"qwen38-27b","temperature":0,"max_tokens":mt,"chat_template_kwargs":{"enable_thinking":False},"messages":[{"role":"user","content":prompt+"\n\nSummarize the files above."}]}
    t=time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(URL,json.dumps(body).encode(),{"content-type":"application/json"}),timeout=600) as r: d=json.load(r)
        return {"ok":True,"prompt_tokens":d["usage"]["prompt_tokens"],"out":d["usage"]["completion_tokens"],"s":round(time.time()-t,1)}
    except urllib.error.HTTPError as e: return {"ok":False,"status":e.code,"body":e.read()[:300].decode(errors="replace"),"s":round(time.time()-t,1)}
    except Exception as e: return {"ok":False,"error":repr(e)[:200],"s":round(time.time()-t,1)}
a=text(0,150_000); b=text(300,400_000)
with cf.ThreadPoolExecutor(2) as ex:
    fa=ex.submit(ask,a,1500); fb=ex.submit(ask,b,32,3)
    print("A (first, long decode):",fa.result()); print("B (beyond the pool):",fb.result())
