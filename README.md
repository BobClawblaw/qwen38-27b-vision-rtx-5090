# Qwen3.8-27B with image input and a 262k window on one RTX 5090

Serve [TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter on a single RTX 5090 (sm_120,
32 GB), through [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine on CUDA:
- text and images (PNG, JPEG, WebP) in one OpenAI-compatible API (`image_url` parts);
- the model's full 262,144-token window, with the vision tower loaded, in one stream;
- DFlash2 drafts on every reply;
- structured output (`response_format`) and forced tool calls (`tool_choice: "required"` or a named function);
- a prompt cache that resumes a chat's kept state on the next turn;
- a systemd unit.

## Engine

The recipe builds TensorFold 1.1.1 with three upstream PRs: [#1079](https://github.com/ashhart/TensorFold/pull/1079)
(the qwen3_5 CUDA engine's decode), [#1080](https://github.com/ashhart/TensorFold/pull/1080) (its prefill) and
[#1168](https://github.com/ashhart/TensorFold/pull/1168) (its fp8 decode attention on tensor cores). That is the
[`q35-decode-attn-tc`](https://github.com/BobClawblaw/TensorFold/tree/q35-decode-attn-tc) branch at `50c05658`.
- Every kernel is built from the engine's own CUDA sources (`TENSORFOLD_CUDA_KERNELS=native`), with no captured Triton kernels.
- `setup.sh` also builds three libraries the server loads from `zig-out/native/lib`:
  - `libtfgrammar.so` (`zig build grammar` on [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8), for structured output and forced tool calls;
  - `libturbojpeg.so.0` from [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) 3.1.4.1, for JPEG images;
  - `libwebpdecoder.so.3` from [libwebp](https://chromium.googlesource.com/webm/libwebp) 1.6.0, for WebP images.

  The engine checks the decoders' versions and refuses JPEG or WebP images with any other release, and distributions
  ship older ones, so both are built from source. `serve.sh` refuses to start without `libtfgrammar.so`: the server
  loads it on the first structured request and would keep a failed load until restart.

### Against the previous version of this recipe

Until 2026-10-21 the recipe built the qwen27 CUDA engine on TensorFold 1.0.11 (fork branch `qwen27-cuda-1.0.7`,
upstream PR [#726](https://github.com/ashhart/TensorFold/pull/726)). That version is at
[`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52), with its evidence.

Same card, same day, fresh servers, fp8 caches, 262,144-token window
([`evidence/2026-10-21-attn-tensor-cores/`](evidence/2026-10-21-attn-tensor-cores/), `vs-qwen27/` and
`vs-qwen27-fixed/`):

| | this engine | previous engine |
|---|---:|---:|
| Short prompts, decode tok/s (code sampled / chat sampled / code greedy / chat greedy) | **326 / 234 / 344 / 300** | 305 / 219 / 307 / 283 |
| Fixed 35,554-token prompt, 512 greedy tokens, whole request ([`tools/fixedlen.py`](tools/fixedlen.py), two runs) | 12.8-13.1 s | 12.7 s |
| Fixed 109,537-token prompt, the same | 46.1-46.4 s | 46.6 s |
| `longbench` 33,750-token prompt: prefill / decode | 10.3 s / **216.2** tok/s | 10.1 s / 202.7 tok/s |
| `longbench` 104,610-token prompt: prefill / decode | 41.5 s / 172.2 tok/s | 40.9 s / **202.3** tok/s |
| Structured output, forced tool calls | **yes** | refused with a 400 |
| WebP images | **yes** | no |

- **Short prompts:** this engine decodes 6-12% faster.
- **Long prompts:** the two are about even. On the fixed prompts (the same text every run, prompt cache off) a whole
  request takes 1-3% longer at 35k and up to 1% less at 110k; this engine reports 219-230 tok/s decode at 35k and 181-184 at 110k, while
  the previous engine reports no decode time.
- **`longbench`** puts a fresh nonce in each prompt, so its replies, and with them the drafts' acceptance and the
  decode speed, change from run to run. In an earlier run on the same day the previous engine decoded 153.7 tok/s at
  105k; in this one it decoded 202.3.

The previous version is still a working recipe if you need its exact replies.

## Against vLLM on the same card

The vLLM side was measured on 2026-10-12 ([`evidence/2026-10-12-vs-vllm/`](evidence/2026-10-12-vs-vllm/)) and this
engine on 2026-10-21 ([`evidence/2026-10-21-attn-tensor-cores/`](evidence/2026-10-21-attn-tensor-cores/)): same RTX 5090, one
request at a time on both, greedy unless noted. The vLLM side is vLLM 0.27.1 serving
[RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) with these settings:
- `--max-model-len 262144`
- `--kv-cache-dtype turboquant_4bit_nc`
- `--max-num-seqs 1`
- MTP speculation with 1 token
- `--max-num-batched-tokens 512`

This is one vLLM configuration, chosen to fit the full window, not tuned for speed. It drafts 1 MTP token; the
[MiaAI-Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090) uses the same settings with 3 and reports
about 160 tok/s single-stream, which we have not reproduced. vLLM with more draft tokens, a larger
`--max-num-batched-tokens` or a shorter window would likely narrow the gaps below. We have not compared against
SGLang, llama.cpp, ExLlamaV3 or SparkInfer.

**Decode, short prompts** (`bench_openai.py --tokens 64 --reps 5`, tok/s, median):

| Cell | this recipe | vLLM NVFP4 |
|---|---:|---:|
| Code, sampled (T=1) | **326.6** | 113.2 |
| Chat, sampled (T=1) | **235.3** | 110.2 |
| Code, greedy | **344.6** | 119.2 |
| Chat, greedy | **300.7** | 114.1 |
| First token (chat) | 0.10 s | **0.06 s** |

**Long prompts** ([`tools/longbench.py`](tools/longbench.py)):
- Source code, with a fresh nonce at the top so no prefix cache can answer.
- A passphrase line hidden at mid-depth.
- A 512-token greedy reply, thinking off.

| Prompt (this recipe / vLLM) | this recipe: prefill | vLLM: prefill | this recipe: decode | vLLM: decode | needle |
|---:|---:|---:|---:|---:|---|
| 33,757 / 33,916 | 10.2 s (3,309 tok/s) | **6.5 s** (5,192 tok/s) | **215.0** tok/s | 69.9 tok/s | both found |
| 104,613 / 105,334 | 41.3 s (2,531 tok/s) | **33.1 s** (3,187 tok/s) | **158.7** tok/s | 36.0 tok/s | both found |
| 214,897 / 214,767 | 117.1 s (1,835 tok/s) | **114.8 s** (1,871 tok/s) | **128.9** tok/s | 21.3 tok/s | both found |

**Images:** one image chat (a 640x400 PNG, 200-token cap, greedy) takes 0.45-0.68 s end to end on this recipe (the first of five runs is the slowest)
against 0.76-0.95 s on vLLM. Both read the image's text correctly.

**Quality** (the [qwen quality suite](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks/tree/main/quality),
through the [recipe harness](https://github.com/BobClawblaw/recipe-harness), thinking off). This table was measured on
engine `2bd805a4`, before #1168, whose attention rounds its weights to bf16 and so changes replies slightly; the
quality suite has not been rerun on `50c05658` ([`evidence/2026-10-21-q35-engine/harness/`](evidence/2026-10-21-q35-engine/harness/)).
The exactness rows (drafts, concurrent, soak) were rerun on `50c05658` and are the same (below).

| Check | this recipe | vLLM NVFP4 |
|---|---:|---:|
| drafts (drafted equals undrafted) | 12/12 | 12/12 |
| concurrent (4 together equal alone) | 12/12 | 12/12 |
| behaviour: streaming, stop, copy, max-tokens | 12/12 | 12/12 |
| behaviour: multiturn recall | 7/8 | **8/8** |
| behaviour: thinking (`reasoning_content`, no tag leaks) | **10/10** | 0/10 |
| tools | **59/60** | 58/60 |
| json | 30/30 | 30/30 |
| IFEval (loose / strict) | 126 / 119 of 150 | **129 / 122** of 150 |
| GSM8K | 238/250 | **240/250** |
| MGSM (8 languages) | 214/240 | **217/240** |
| MMLU (57 subjects x 6) | **270/342** | 263/342 |
| HumanEval pass@1 | 151/164 | **153/164** |
| repetition | 8/8 | 8/8 |
| long context (needles to 224k, facts to 225k) | 55/55 | 55/55 |
| vision probes | 12/12 | 12/12 |
| soak: replies equal | 60/60 | 60/60 |
| soak: turns resuming a kept state | **60/60** | not reported |

GSM8K's 12 misses include 10 replies cut at the suite's 1,024-token cap. Thinking-on quality (`--plan thinking`) was
not run on this engine.

The harness's `quick` plan on `50c05658` ([`harness-quick/`](evidence/2026-10-21-attn-tensor-cores/harness-quick/)):
- **drafts** 12/12, **concurrent** 12/12 and **soak** 60/60 (all 60 turns resumed a kept state);
- **structured:** 18/18 replies obey their format, drafted equals plain 12/12, and a bad regex is refused with a 400;
- **tools:** 5/5 tool calls right: `auto`, `required`, and a named function plain, with thinking on and streamed;
- **sampling:** 9/9 seeded replies equal on repeat and undrafted;
- **decode ruler** (prose / code / structured / list): 254.5 / 486.0 / 327.4 / 306.6 tok/s fresh, 250.9 / 483.9 /
  326.1 / 305.6 after the checks.

The two engines score within a few answers of each other: different 4-bit quantizations of the same model.

In short, against this vLLM configuration only:
- **Decode:** this recipe measured 2-3x faster at short prompts and 3-6x at long ones. Against vLLM with MTP-3, as
  reported elsewhere, the gap would be smaller; we have not measured how much.
- **Prefill:** vLLM reads prompts faster up to about 100k tokens; at 215k the two are close. First token is faster on vLLM.
- **Quality:** the same within a few answers either way. This recipe returns reasoning in `reasoning_content`; vLLM, as configured, does not.
- **Prompt cache:** here a 105k-token chat's next turn starts in 1.35 s instead of 42 s. vLLM does not report cached
  tokens, so we did not measure its prefix caching.
- **Concurrency:** both ran one request at a time. This recipe serves one stream at the full window; batching
  servers reach higher total throughput under concurrent load.

## Prompt cache

The engine keeps a chat's prompt state in pinned host memory, so the next turn reads only what is new. Prompts are
prefilled in passes of up to 4,096 rows, and states are kept at those marks. The cache takes 8 GiB by default, never
more than half of `MemAvailable`; set it with `PROMPT_CACHE_GIB` (`--prompt-cache-gib`), and `0` turns it off.

Evidence in [`evidence/2026-10-21-attn-tensor-cores/`](evidence/2026-10-21-attn-tensor-cores/):
- **Multi-turn**, from `tools/turns.py`: a 104,571-token source dump, then two short follow-ups.

  | Turn | prompt tokens | resumed | first token |
  |---:|---:|---:|---:|
  | 1 | 104,571 | 0 | 42.02 s |
  | 2 | 104,630 | 102,400 | 1.35 s |
  | 3 | 104,703 | 102,400 | 1.36 s |

- **Cache on against off:** a fixed 3-turn chat (21,456 to 21,603 tokens, greedy, 128 tokens a turn) on a server
  with the cache and on one with `PROMPT_CACHE_GIB=0`. Turns 2 and 3 resumed 20,480 kept tokens with the cache, and
  all 3 replies are byte-identical.
- **Soak:** 3 chats of 20 turns at once beside a decoding stream. All 60 replies equal the same request alone, and
  all 60 turns resumed a kept state.

## Measured on this card

RTX 5090 (32 GB), CUDA 13.3, engine `50c05658`, 262,144-token window, vision loaded. At startup the engine reports
the weights at 17.89 GiB and one stream at 9.02 GiB of the 12.0 GiB left after a 1.0 GiB reserve. `nvidia-smi`
shows 28,054 MiB in use when idle.

- **Needle in 255,054 tokens** of source code at depths 0.10, 0.33, 0.50 and 0.90: found 4/4, 151 s a prompt.
- **Prefill and decode** ([`tools/longspeed.py`](tools/longspeed.py), 512 tokens): 100,024 tokens in 39.2 s, then
  216.6 tok/s; 250,024 tokens in 110.7 s, then 151.1 tok/s.
- **Images** (`tools/image_chats.py`, `tools/image_stress.py`, and one picture sent as PNG, JPEG and WebP): all
  three formats read the picture's text and shape, and the chats describe their images correctly under fp8 and
  bf16, except that one reads the vertical text in an EXIF-rotated JPEG as "ORANGE7561" (it says "ORANGE 7351").
  The stress set covers an image across a 4,096-row prompt pass, a 16,384-patch image, thinking on, 4 images in one
  request, a 5th image refused, and three text requests beside an image request, whose replies equal their solo
  replies.

**KV cache, fp8 against bf16:**

| | fp8 (default) | bf16 |
|---|---:|---:|
| Window with vision, one stream | 262,144 (9.02 GiB) | 131,072 measured (8.90 GiB) |
| 100k-token prompt (`longspeed.py`): prefill / decode | 39.2 s / 216.6 tok/s | 37.5 s / 82.1 tok/s |
| Short prompts (code sampled / chat sampled / code greedy / chat greedy) | 327 / 235 / 345 / 301 | 340 / 245 / 341 / 298 |

The fp8 cache's decode attention runs on tensor cores (#1168); the bf16 cache's still takes one key at a time, so
bf16 decodes much slower after a long prompt.

Not rerun on this engine: the quality suite (above), fp8 against bf16 teacher-forced KL, the vision tower against fp32
references, and token exactness against TensorFold 0.6.6. The previous engine's results are in the evidence folders at
[`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52).

**Decode on a long-running server.** On engine `2bd805a4`, after the full quality suite (about 40 minutes of mixed
load), the harness's decode ruler read 234 / 372 / 288 / 292 tok/s against 251 / 478 / 322 / 308 on a fresh server,
and restarting the server restored it. On `50c05658` the `quick` plan's ruler after about four minutes of checks is
within 1.4% of fresh (above); the longer run has not been repeated.

## Setup

You need:
- an RTX 5090 (the engine is built for sm_120);
- the CUDA 13 toolkit (`nvcc`) and Zig 0.17.0;
- `cmake` and `nasm`, for the JPEG and WebP decoders;
- `hf` (huggingface_hub).

```bash
git clone https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090.git && cd qwen38-27b-vision-rtx-5090
NVCC=/usr/local/cuda/bin/nvcc ./setup.sh     # ~19 GB of weights, the engine at the pinned commit, built for sm_120
./serve.sh                                   # foreground; :30000, model qwen38-27b (alias qwen38-nvfp4)
```

To run it as a service:
1. Copy `systemd/tensorfold-qwen38.service` to `/etc/systemd/system/`.
2. Set its `User`, `Group` and paths.
3. Run `sudo systemctl enable --now tensorfold-qwen38`.

**Knobs** (environment, or a systemd drop-in via `sudo systemctl edit tensorfold-qwen38`):

| Variable | Default | |
|---|---|---|
| `KV_CACHE` | `fp8` | `bf16` holds half the window: set `CONTEXT=131072` with it |
| `PROMPT_CACHE_GIB` | `8` | host memory for kept prompt states, at most half of `MemAvailable`; `0` turns the cache off |
| `CONTEXT` | `262144` | the window |
| `PARALLEL` | `auto` | streams at once, as many as the window allows: one at 262,144 (more streams were not measured on this engine) |
| `MAX_TOKENS` | the engine's | the output cap for a request that sends no `max_tokens` |
| `THINKING_BUDGET` | the engine's | a cap on a reply's reasoning tokens |
| `PORT` / `HOST` | `30000` / `0.0.0.0` | |
| `NAME` / `ALIAS` | `qwen38-27b` / `qwen38-nvfp4` | |
| `MODELS` / `TF_SRC` | `./models` / `./engine` | |

**Requests:**
- There is no API key.
- Thinking is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- Images go in `image_url` parts as data URLs (PNG, JPEG, WebP).
- A request can hold up to 4 images, sharing 4,096 visual tokens. `detail: low` caps one image at 256.
- Structured output: `response_format` (`json_object`, `json_schema`), and `tool_choice` `"auto"`, `"required"` or a
  named function.

## Tools

| Script | What |
|---|---|
| `tools/longbench.py URL MODEL` | long-prompt prefill and decode with a mid-depth needle (the table above) |
| `tools/needle.py URL MODEL_DIR TOKENS DEPTHS` | needle in a haystack at chosen depths, token-exact lengths |
| `tools/longspeed.py` | prefill and decode at one long prompt |
| `tools/turns.py URL MODEL [TOKENS] [TURNS]` | a long chat's next turns: first-token time and resumed tokens |
| `tools/fixedlen.py URL MODEL OUT` | the same 35k- and 110k-token prompts every run, 512 greedy tokens: whole-request times to compare engines |
| `tools/image_chats.py PORT OUT` | 5 image chats, greedy, saved for comparison |
| `tools/image_stress.py PORT` | images at a prompt-pass boundary, the largest image, 4 images, a 5th refused, concurrent text |
| `tools/vision-req.json` | one image chat for `curl -d @tools/vision-req.json` |

The decode cells come from TensorFold's own `tools/bench_openai.py`. The quality suite and the decode ruler come from the
[qwen Spark recipe](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks), run through the
[recipe harness](https://github.com/BobClawblaw/recipe-harness).

## License

MIT for this repository. The model, drafter and TensorFold carry their own licenses.
