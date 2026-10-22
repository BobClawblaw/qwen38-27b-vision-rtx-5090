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

The recipe builds TensorFold 1.1.2 with four upstream PRs: [#1079](https://github.com/ashhart/TensorFold/pull/1079)
(the qwen3_5 CUDA engine's decode), [#1080](https://github.com/ashhart/TensorFold/pull/1080) (its prefill),
[#1168](https://github.com/ashhart/TensorFold/pull/1168) (its decode attention on tensor cores, for fp8 and bf16 caches)
and [#1176](https://github.com/ashhart/TensorFold/pull/1176) (prompt matmuls on int8 activations). That is the
[`q35-prompt-int8`](https://github.com/BobClawblaw/TensorFold/tree/q35-prompt-int8) branch at `18d93451`. Every number
below was measured on a fresh `setup.sh` build of it ([`evidence/2026-10-22-lone-bands/`](evidence/2026-10-22-lone-bands/))
unless it says otherwise.
- Prompts run with int8 activations by default (`PROMPT_PRECISION=int8`, `--prompt-precision int8`): each 4-bit
  projection's inputs are rounded to int8 with one scale a 64-input group, and the matmul runs on the int8 tensor
  cores. Prefill takes 20-27% less time than with bf16 activations. Replies differ slightly from bf16 prompts' (the
  measured divergence is in [#1176](https://github.com/ashhart/TensorFold/pull/1176)); drafted replies still equal
  undrafted ones. `PROMPT_PRECISION=bf16` keeps the exact bf16 path.
- A long prompt no longer slows the short replies after it (#1079's top commit, `f15dad20`; see
  [Decode on a long-running server](#decode-on-a-long-running-server)).
- Every kernel is built from the engine's own CUDA sources (`TENSORFOLD_CUDA_KERNELS=native`), with no captured Triton kernels.
- `setup.sh` also builds three libraries the server loads from `zig-out/native/lib`:
  - `libtfgrammar.so` (`zig build grammar` on [xgrammar](https://github.com/mlc-ai/xgrammar) v0.2.8), for structured output and forced tool calls;
  - `libturbojpeg.so.0` from [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) 3.1.4.1, for JPEG images;
  - `libwebpdecoder.so.3` from [libwebp](https://chromium.googlesource.com/webm/libwebp) 1.6.0, for WebP images.

  The engine checks the decoders' versions and refuses JPEG or WebP images with any other release, and distributions
  ship older ones, so both are built from source. `serve.sh` refuses to start without `libtfgrammar.so`: the server
  loads it on the first structured request and would keep a failed load until restart.

### Against the previous versions of this recipe

Two earlier versions:
- `0d3322a1` (earlier on 2026-10-22) was this stack without the fix for decode after a long prompt. It is at
  [`35bad3f`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/35bad3f), with its evidence; the fork
  tag `q35-prompt-int8-0d3322a1` keeps the commit.
- `d6260feb` (2026-10-21) built TensorFold 1.1.1 with #1079, #1080 and #1168, with bf16 prompt activations. It is at
  [`6eebd01`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/6eebd01).

Same card, fp8 caches, 262,144-token window. The second column is this build with `PROMPT_PRECISION=bf16`, measured
on the same day as the first:

| | this version (`18d93451`) | this build, bf16 prompts | `0d3322a1` | `d6260feb` |
|---|---:|---:|---:|---:|
| Fixed 35,554-token prompt, 512 greedy tokens, whole request ([`tools/fixedlen.py`](tools/fixedlen.py), prompt cache off, two runs) | **10.01-10.02 s** | 12.93-13.07 s | 10.0-10.2 s | 12.8-13.1 s |
| Its first token | **7.90-7.94 s** | 10.66-10.79 s | 7.90-7.96 s | |
| Fixed 109,537-token prompt, the same | **37.85-37.87 s** | 46.55-46.71 s | 37.8-37.9 s | 46.1-46.4 s |
| Its first token | **34.9-35.0 s** | 43.5-43.7 s | 34.9-35.1 s | |
| 100,024-token prompt ([`tools/longspeed.py`](tools/longspeed.py)): prefill / decode | **31.3 s** / 213.7 tok/s | | 31.1 s / 216.0 tok/s | 39.2 s / 216.6 tok/s |
| 250,024-token prompt: prefill / decode | **98.4 s** / 146.2 tok/s | | 98.2 s / 139.5 tok/s | 110.7 s / 151.1 tok/s |
| Harness decode ruler, fresh server (prose / code / structured / list) | 257.3 / 496.8 / 325.5 / 312.0 | | 257.1 / 497.2 / 325.7 / 312.3 | 254.5 / 486.0 / 327.4 / 306.6 |
| The ruler right after a 512-token prose reply to a cold 245k-token prompt (harness `after-long` plan) | **257.7 / 497.5** / 326.4 / 312.7 | | 238.6 / 415.6 / 326.2 / 312.0 | |

- **Prefill:** int8 activations take 20-27% off the time to first token at 35k and 110k tokens; on the same build the
  int8 and bf16 columns differ only in the prompt's arithmetic.
- **Decode:** the ruler is level on a fresh server. On the fixed prompts decode ran 243-247 tok/s at 35k and 173-179
  at 110k with int8 prompts, against 225-226 and 169 with bf16. A long reply's speed depends on how many drafts it
  accepts, which moves with the reply's text (at 250k: 146.2, 139.5 and 151.1).
- **After a long prompt:** on `0d3322a1` the short replies right after one long reply ran 7% slower on prose and 17%
  on code; this version keeps them at fresh speed. The replies to the fixed prompts are byte-identical to `0d3322a1`'s.

Until 2026-10-21 the recipe built the qwen27 CUDA engine on TensorFold 1.0.11 (fork branch `qwen27-cuda-1.0.7`,
upstream PR [#726](https://github.com/ashhart/TensorFold/pull/726)). That version is at
[`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52), with its evidence; the comparison
against it is in [`evidence/2026-10-21-attn-tensor-cores/`](evidence/2026-10-21-attn-tensor-cores/) (`vs-qwen27/` and
`vs-qwen27-fixed/`). It refuses structured output and WebP images.

## Against vLLM on the same card

The vLLM side was measured on 2026-10-12 ([`evidence/2026-10-12-vs-vllm/`](evidence/2026-10-12-vs-vllm/)) and this
engine on 2026-10-22 ([`evidence/2026-10-22-lone-bands/`](evidence/2026-10-22-lone-bands/)): same RTX 5090, one
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
| Code, sampled (T=1) | **318.6** | 113.2 |
| Chat, sampled (T=1) | **246.4** | 110.2 |
| Code, greedy | **319.7** | 119.2 |
| Chat, greedy | **300.0** | 114.1 |
| First token (chat) | **0.05 s** | 0.06 s |

**Long prompts** ([`tools/longbench.py`](tools/longbench.py)):
- Source code, with a fresh nonce at the top so no prefix cache can answer.
- A passphrase line hidden at mid-depth.
- A 512-token greedy reply, thinking off.

| Prompt (this recipe / vLLM) | this recipe: prefill | vLLM: prefill | this recipe: decode | vLLM: decode | needle |
|---:|---:|---:|---:|---:|---|
| 33,756 / 33,916 | 7.6 s (4,427 tok/s) | **6.5 s** (5,192 tok/s) | **205.7** tok/s | 69.9 tok/s | both found |
| 104,608 / 105,334 | 33.3 s (3,140 tok/s) | **33.1 s** (3,187 tok/s) | **152.3** tok/s | 36.0 tok/s | both found |
| 214,893 / 214,767 | **100.6 s** (2,137 tok/s) | 114.8 s (1,871 tok/s) | **112.9** tok/s | 21.3 tok/s | both found |

**Images:** one image chat (a 640x400 PNG, 200-token cap, greedy) takes 0.47-0.67 s end to end on this recipe (the first of five runs is the slowest)
against 0.76-0.95 s on vLLM. Both read the image's text correctly.

**Quality** (the [qwen quality suite](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks/tree/main/quality),
through the [recipe harness](https://github.com/BobClawblaw/recipe-harness), thinking off). The suite was not rerun on
the published build. The int8 column is the int8 engine two revisions before `18d93451`, served on this card
([`harness-quality/`](evidence/2026-10-22-int8-prompts/harness-quality/)). One later revision only stopped reserving
int8 memory when int8 is off. The other (`f15dad20`) changes only how many draft rows a round verifies, and drafted
replies equal undrafted ones. The greedy replies to a 64,811-token prompt and the replies to the short benchmark
prompts are byte-identical across the three, and `18d93451`'s to the fixed 35k and 110k prompts equal `0d3322a1`'s. The bf16 column is the previous version's engine, `50c05658` on 1.1.1
([`evidence/2026-10-21-attn-tensor-cores/harness/`](evidence/2026-10-21-attn-tensor-cores/harness/)), and the last
column `2bd805a4`, before #1168 ([`evidence/2026-10-21-q35-engine/harness/`](evidence/2026-10-21-q35-engine/harness/)).

| Check | int8 prompts (this recipe) | bf16 prompts, `50c05658` | vLLM NVFP4 | engine `2bd805a4` |
|---|---:|---:|---:|---:|
| drafts (drafted equals undrafted) | 12/12 | 12/12 | 12/12 | 12/12 |
| concurrent (4 together equal alone) | 12/12 | 12/12 | 12/12 | 12/12 |
| behaviour: streaming, stop, copy, max-tokens | 12/12 | 12/12 | 12/12 | 12/12 |
| behaviour: multiturn recall | 7/8 | 7/8 | **8/8** | 7/8 |
| behaviour: thinking (`reasoning_content`, no tag leaks) | **10/10** | **10/10** | 0/10 | 10/10 |
| tools | 58/60 | **59/60** | 58/60 | 59/60 |
| json | 30/30 | 30/30 | 30/30 | 30/30 |
| IFEval (loose / strict) | 129 / 121 of 150 | **130** / 122 of 150 | 129 / 122 of 150 | 126 / 119 of 150 |
| GSM8K | **241/250** | 240/250 | 240/250 | 238/250 |
| MGSM (8 languages) | 215/240 | 213/240 | **217/240** | 214/240 |
| MMLU (57 subjects x 6) | 270/342 | **271/342** | 263/342 | 270/342 |
| HumanEval pass@1 | 152/164 | 151/164 | **153/164** | 151/164 |
| repetition | 8/8 | 8/8 | 8/8 | 8/8 |
| long context (needles to 224k, facts to 225k) | 55/55 | 55/55 | 55/55 | 55/55 |
| vision probes | 12/12 | 12/12 | 12/12 | 12/12 |
| soak: replies equal | 60/60 | 60/60 | 60/60 | 60/60 |
| soak: turns resuming a kept state | **60/60** | **60/60** | not reported | 60/60 |

GSM8K's 9 misses with int8 prompts include 8 replies cut at the suite's 1,024-token cap. Thinking-on quality (`--plan
thinking`) was not run on this engine.

The harness's `quick` plan on the fresh build of `18d93451` ([`harness-quick/`](evidence/2026-10-22-lone-bands/harness-quick/)):
- **drafts** 12/12, **concurrent** 12/12 and **soak** 60/60 (all 60 turns resumed a kept state);
- **structured:** 18/18 replies obey their format, drafted equals plain 12/12, and a bad regex is refused with a 400;
- **tools:** 5/5 tool calls right: `auto`, `required`, and a named function plain, with thinking on and streamed;
- **sampling:** 9/9 seeded replies equal on repeat and undrafted;
- **decode ruler** (prose / code / structured / list): 257.3 / 496.8 / 325.5 / 312.0 tok/s fresh, 257.5 / 497.7 /
  326.1 / 312.3 after the checks.

The engines score within a few answers of each other: int8 and bf16 prompts on the same weights, and different
4-bit quantizations of the same model.

In short, against this vLLM configuration only:
- **Decode:** this recipe measured 2-3x faster at short prompts and 3-5x at long ones. Against vLLM with MTP-3, as
  reported elsewhere, the gap would be smaller; we have not measured how much.
- **Prefill:** vLLM reads a 34k-token prompt faster, the two are level at 105k, and this recipe is faster at 215k.
  First token on a short chat is 0.05 s here and 0.06 s on vLLM.
- **Quality:** the same within a few answers either way. This recipe returns reasoning in `reasoning_content`; vLLM, as configured, does not.
- **Prompt cache:** here a 105k-token chat's next turn starts in 1.14 s instead of 34 s. vLLM does not report cached
  tokens, so we did not measure its prefix caching.
- **Concurrency:** both ran one request at a time. This recipe serves one stream at the full window; batching
  servers reach higher total throughput under concurrent load.

## Prompt cache

The engine keeps a chat's prompt state in pinned host memory, so the next turn reads only what is new. Prompts are
prefilled in passes of up to 4,096 rows, and states are kept at those marks. The cache takes 8 GiB by default, never
more than half of `MemAvailable`; set it with `PROMPT_CACHE_GIB` (`--prompt-cache-gib`), and `0` turns it off.

Evidence in [`evidence/2026-10-22-lone-bands/`](evidence/2026-10-22-lone-bands/):
- **Multi-turn**, from `tools/turns.py`: a 104,571-token source dump, then two short follow-ups.

  | Turn | prompt tokens | resumed | first token |
  |---:|---:|---:|---:|
  | 1 | 104,570 | 0 | 34.0 s |
  | 2 | 104,628 | 102,400 | 1.14 s |
  | 3 | 104,703 | 102,400 | 1.14 s |

- **Cache on against off:** a fixed 3-turn chat (21,456 to 21,600 tokens, greedy, 128 tokens a turn) on a server
  with the cache and on one with `PROMPT_CACHE_GIB=0`. Turns 2 and 3 resumed 20,480 kept tokens with the cache, and
  all 3 replies are byte-identical.
- **Soak:** 3 chats of 20 turns at once beside a decoding stream. All 60 replies equal the same request alone, and
  all 60 turns resumed a kept state.

## Measured on this card

RTX 5090 (32 GB), CUDA 13.3, engine `18d93451`, int8 prompts, 262,144-token window, vision loaded. At startup the
engine reports the model and its working memory at 18.13 GiB and one stream at 9.02 GiB of the 11.8 GiB left after a
1.0 GiB reserve. `nvidia-smi` shows 27,984 MiB in use when idle.

- **Needle in 255,054 tokens** of source code at depths 0.10, 0.33, 0.50 and 0.90: found 4/4, 130-131 s a prompt.
- **Prefill and decode** ([`tools/longspeed.py`](tools/longspeed.py), 512 tokens): 100,024 tokens in 31.3 s, then
  213.7 tok/s; 250,024 tokens in 98.4 s, then 146.2 tok/s.
- **Images** (`tools/image_chats.py`, `tools/image_stress.py`, and one picture sent as PNG, JPEG and WebP): all
  three formats read the picture's text and shape under fp8 and bf16 caches, and the chats describe their images
  correctly, except that one reads the vertical text in an EXIF-rotated JPEG as "ORANGE7561" (it says "ORANGE 7351").
  The stress set covers an image across a 4,096-row prompt pass, a 16,384-patch image, thinking on, 4 images in one
  request, a 5th image refused, and three text requests beside an image request, whose replies equal their solo
  replies.

**KV cache, fp8 against bf16:**

| | fp8 (default) | bf16 |
|---|---:|---:|
| Window with vision, one stream | 262,144 (9.02 GiB) | 131,072 measured (8.90 GiB) |
| 100k-token prompt (`longspeed.py`): prefill / decode | 31.3 s / 213.7 tok/s | 29.7 s / 197.6 tok/s |
| Short prompts (code sampled / chat sampled / code greedy / chat greedy) | 319 / 247 / 320 / 300 | 300 / 245 / 325 / 297 |

Not rerun on this engine: fp8 against bf16 teacher-forced KL, the vision tower against fp32
references, and token exactness against TensorFold 0.6.6. The previous engine's results are in the evidence folders at
[`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52).

### Decode on a long-running server

Earlier engines slowed down after long prompts. After the full quality suite (about 40 minutes of mixed load) the
harness's decode ruler read lower than on a fresh server: on the int8 engine 239.1 / 378.3 / 292.8 / 297.2 tok/s
against 253.9 / 498.5 / 326.4 / 312.3, and on `2bd805a4` 234 / 372 / 288 / 292 against 251 / 478 / 322 / 308.
Restarting the server restored it.

The cause was in #1079. Each lone stream started from the previous stream's measured round costs and from its
deadlines for retrying draft-window widths.
- The costs carried no context length. A long prompt's wide rounds cost about twice a short prompt's, so every short
  stream after one verified narrower windows: 8 drafts a round instead of 14-15 on the ruler's prose.
- The retry deadlines doubled from stream to stream without bound, so after a long run no stream re-timed the wider
  windows.

`f15dad20` keeps the costs a context band each (under 8,192 tokens, then one band a doubling) and caps a carried
deadline at 256 rounds. The harness's `after-long` plan runs the ruler, one 512-token prose reply to a cold
245,409-token prompt, and the ruler again. This version reads 257.7 / 497.5 / 326.4 / 312.7 after the long reply
against 256.9 / 497.6 / 326.5 / 313.2 fresh; `0d3322a1` read 238.6 / 415.6 / 326.2 / 312.0 against 257.6 / 498.2 /
326.4 / 312.7 ([`harness-after-long/`](evidence/2026-10-22-lone-bands/harness-after-long/),
[`harness-after-long-0d3322a1/`](evidence/2026-10-22-lone-bands/harness-after-long-0d3322a1/)). The full quality suite
was not rerun on this version.

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
| `PROMPT_PRECISION` | `int8` | the prompt matmuls' activations; `bf16` is exact and takes 24-37% longer to read a prompt |
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
