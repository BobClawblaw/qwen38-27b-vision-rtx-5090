# Qwen3.8-27B with image input and a 262k window on one RTX 5090

Serve [TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter on a single RTX 5090 (sm_120,
32 GB), through [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine on CUDA:
- text and images in one OpenAI-compatible API (`image_url` parts);
- the model's full 262,144-token window, with the vision tower loaded, in one stream;
- DFlash2 drafts on every reply;
- a systemd unit.

Upstream TensorFold v1.0.7 serves only Nemotron on CUDA and has no image path. This recipe builds the
[`qwen27-cuda-1.0.7`](https://github.com/BobClawblaw/TensorFold/tree/qwen27-cuda-1.0.7) branch (upstream PR
[#726](https://github.com/ashhart/TensorFold/pull/726)), which is v1.0.7 plus:

- **The qwen27 CUDA engine** from upstream [#608](https://github.com/ashhart/TensorFold/pull/608), merged onto 1.0.7's chunked admission.
- **Native image input** in the server from upstream [#598](https://github.com/ashhart/TensorFold/pull/598) and [#614](https://github.com/ashhart/TensorFold/pull/614): image parts, limits, placeholder expansion.
- **The Qwen vision tower on CUDA** (`zig/kernels/cuda/qwen_vision.cu`, `families/qwen27/cuda/vision.zig`).
  It is transformers' `Qwen3_5VisionModel`: 27 blocks, axial 2-D RoPE, a bilinear position grid and a 2x2 merger, with the Linears on cuBLASLt.
  Its rows match TensorFold 0.6.6's at a mean cosine of 0.9997-0.99998.
- **The processor's preparation on Linux** (`families/qwen27/cuda/image.zig`):
  - JPEG decoding through libturbojpeg (PIL's decoder), falling back to stb_image;
  - EXIF orientation, alpha composited over white, `smart_resize`, PIL's bicubic, then Qwen patches;
  - bit-identical to 0.6.6 on 9 fixtures.
- **Image rows spliced into the prompt pass**, with multimodal (M-RoPE) positions on the chunks that hold image rows and on decode and verify after them.
- **`--kv-cache fp8`**: attention caches stored as e4m3 rows, each with one fp32 scale (amax / 448) per position and key head.
  - That is half of bf16's bytes, so 262,144 tokens fit in one stream beside the tower.
  - The decode kernel quantizes the in-window rows exactly as their commit will. A key's bits therefore never depend on windows, drafts or batching: drafted replies equal undrafted ones, and a concurrent reply equals the same request alone.
  - bf16 stays available (`KV_CACHE=bf16`) and is exact against TensorFold 0.6.6, with the prompt cache on or off.

- **A prompt cache** (`families/qwen27/cuda/reuse.zig`): a chat's prompt state kept in pinned host memory, so the next turn reads only what is new. Details in [Prompt cache](#prompt-cache).
- **Merged with upstream main** through 1.0.11.

## Against vLLM on the same card

Same RTX 5090, same day, one request at a time on both, greedy unless noted. The vLLM side is vLLM 0.27.1 serving
[RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) with these settings:
- `--max-model-len 262144`
- `--kv-cache-dtype turboquant_4bit_nc`
- `--max-num-seqs 1`
- MTP speculation with 1 token
- `--max-num-batched-tokens 512`

Files: [`evidence/2026-10-12-vs-vllm/`](evidence/2026-10-12-vs-vllm/).

This is one vLLM configuration, chosen to fit the full window, not tuned for speed. It drafts 1 MTP token; the
[MiaAI-Lab recipe](https://github.com/MiaAI-Lab/Qwen3.8-27B-NVFP4-RTX-5090) uses the same settings with 3 and reports
about 160 tok/s single-stream, which we have not reproduced. vLLM with more draft tokens, a larger
`--max-num-batched-tokens` or a shorter window would likely narrow the gaps below. We have not compared against
SGLang, llama.cpp, ExLlamaV3 or SparkInfer.

**Decode, short prompts** (`bench_openai.py --tokens 64 --reps 5`, tok/s, median):

| Cell | this recipe | vLLM NVFP4 |
|---|---:|---:|
| Code, sampled (T=1) | **307.3** | 113.2 |
| Chat, sampled (T=1) | **220.3** | 110.2 |
| Code, greedy | **309.1** | 119.2 |
| Chat, greedy | **285.1** | 114.1 |
| First token | 0.10 s | **0.06 s** |

**Long prompts** ([`tools/longbench.py`](tools/longbench.py)):
- Source code, with a fresh nonce at the top so no prefix cache can answer.
- A passphrase line hidden at mid-depth.
- A 512-token greedy reply, thinking off.

| Prompt | this recipe: prefill | vLLM: prefill | this recipe: decode | vLLM: decode | needle |
|---:|---:|---:|---:|---:|---|
| 33,916 | 10.2 s (3,319 tok/s) | **6.5 s** (5,192 tok/s) | **224.3** tok/s | 69.9 tok/s | both found |
| 105,334 | 40.8 s (2,584 tok/s) | **33.1 s** (3,187 tok/s) | **181.5** tok/s | 36.0 tok/s | both found |
| 214,767 | 114.7 s (1,873 tok/s) | 114.8 s (1,871 tok/s) | **125.7** tok/s | 21.3 tok/s | both found |

**Images:** one image chat (a 640x400 PNG, 200-token cap, greedy) takes 0.70 s end to end on this recipe against
0.76-0.95 s on vLLM. Both read the image's text correctly.

**Quality** (the [qwen quality suite](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks/tree/main/quality),
all 15 checks, thinking off):

| Check | this recipe | vLLM NVFP4 |
|---|---:|---:|
| drafts (drafted equals undrafted) | 12/12 | 12/12 |
| concurrent (4 together equal alone) | 12/12 | 12/12 |
| behaviour: streaming, stop, copy, max-tokens | 12/12 | 12/12 |
| behaviour: multiturn recall | 7/8 | **8/8** |
| behaviour: thinking (`reasoning_content`, no tag leaks) | **10/10** | 0/10 |
| tools | **59/60** | 58/60 |
| json | 30/30 | 30/30 |
| IFEval (loose / strict) | 124 / 116 of 150 | **129 / 122** of 150 |
| GSM8K | **242/250** | 240/250 |
| MGSM (8 languages) | 213/240 | **217/240** |
| MMLU (57 subjects x 6) | **268/342** | 263/342 |
| HumanEval pass@1 | 153/164 | 153/164 |
| repetition | 8/8 | 8/8 |
| long context (needles to 224k, facts to 225k) | 55/55 | 55/55 |
| vision probes | 12/12 | 12/12 |
| soak: replies equal | 60/60 | 60/60 |
| soak: turns resuming a kept state | **60/60** | not reported |
| suite time, all checks | **42 min** | 74 min |

The two engines score within a few answers of each other: different 4-bit quantizations of the same model. Soak's
replies are all equal on both. On this recipe all 60 soak turns also resumed a kept prompt state (see
[Prompt cache](#prompt-cache)); vLLM does not report cached tokens, so its resumes go unchecked. vLLM's thinking 0/10 means its replies carried no `reasoning_content` field.

In short, against this vLLM configuration only:
- **Decode:** this recipe measured 2-3x faster at short prompts and 3-6x at long ones. Against vLLM with MTP-3, as
  reported elsewhere, the gap would be smaller; we have not measured how much.
- **Prefill:** vLLM reads prompts faster up to about 100k tokens; at 215k the two are even. First token is faster on vLLM.
- **Quality:** the same within a few answers either way. This recipe returns reasoning in `reasoning_content`; vLLM, as configured, does not.
- **Prompt cache:** here a 105k-token chat's next turn starts in 1.5 s instead of 40 s. vLLM does not report cached
  tokens, so we did not measure its prefix caching.
- **Concurrency:** both ran one request at a time. This recipe serves one stream at the full window; batching
  servers reach higher total throughput under concurrent load.

## Prompt cache

The engine keeps a chat's prompt state in pinned host memory, so the next turn reads only what is new. A state
holds every DeltaNet layer's recurrence and conv window plus the attention rows up to the mark, about 275 MiB at
4,096 tokens. Prompts are prefilled on a fixed 4,096-row grid, so the marks of one chat's turns line up. The
cache takes 8 GiB by default, never more than half of `MemAvailable`; set it with `PROMPT_CACHE_GIB`
(`--prompt-cache-gib`). Image prompts are neither kept nor resumed.

`PROMPT_CACHE_GIB=0` turns the cache off and goes back to even prefill chunks. The cache does not change replies:
under `KV_CACHE=bf16`, cache on or off, they match TensorFold 0.6.6 token for token (below).

Evidence in [`evidence/2026-10-12-prompt-cache/`](evidence/2026-10-12-prompt-cache/):
- **Multi-turn**, from `tools/turns.py`: a 104,688-token source dump, then two short follow-ups.

  | Turn | prompt tokens | resumed | first token |
  |---:|---:|---:|---:|
  | 1 | 104,688 | 0 | 40.38 s |
  | 2 | 104,746 | 102,400 | 1.49 s |
  | 3 | 104,819 | 102,400 | 1.49 s |

- **Soak:** 3 chats of 20 turns at once beside a decoding stream. All 60 replies equal the same request alone, all
  60 turns resumed, and the check took 118 s instead of 244 s.
- **Everything else** in the full suite scores as it did without the cache.
- **bf16 against TensorFold 0.6.6, cache on** (`tools/exact_cache.py`, in `bf16-cache-vs-0.6.6/`): greedy replies to
  prompts of 1,497 to 50,191 tokens, each asked twice, the same 50k prompt again, and a 3-turn 22k chat. With the cache
  on, the 256-token replies match 0.6.6's in 20/20, 16 of them after resuming 4,096 to 49,152 kept tokens. A first
  run with shorter replies (8-74 tokens) matched 20/20 as well, and the cache-off build matched 20/20. Neither
  server's CUDA path returns logprobs, so this compares tokens, not logits.
- **With the cache off**, the 0.6.6 exactness check still matches: serial 4/4, windows 54/54, drafted 4/4, and no
  reply differs in shared rounds.

## Measured on this card

RTX 5090 (32 GB), CUDA 13.3, engine `5d8bf485` (the prompt cache: `2c222ae4`), `KV_CACHE=fp8`, 262,144-token window, vision loaded. At startup the
engine reports one stream at 8.54 GiB of the 9.4 GiB left after the weights (18.61 GiB).

**fp8 against bf16** ([`evidence/2026-10-12-fp8-kv/`](evidence/2026-10-12-fp8-kv/)):

- **Quality**, from `tf-qwen27-cuda kvcompare`:
  - Both caches teacher-forced on bf16's greedy reply, 128 steps each, prompts of 28 to 60,027 tokens.
  - Top-1 agreement 763/768.
  - Mean KL 0.0002-0.0013 nats a token.
- **Needle in 255,055 tokens** of source code at depths 0.10, 0.33, 0.50 and 0.90: found 4/4.
- **Invariance:**
  - fp8 drafted decode equals fp8 serial decode on all 4 prompts.
  - Three concurrent replies equal their solo replies.
  - Committed keys and in-window keys are bit-identical in the kernel check.
- **Images:** the 5 end-to-end image chats and the stress set are all correct under fp8.
  The stress set covers an image across a 4,096-row chunk boundary, a 16,384-patch image, thinking on, 4 images in one request, a 5th image refused, and text requests beside an image request.

| | fp8 | bf16 |
|---|---:|---:|
| Window with vision, one stream | 262,144 | 126,976 (measured before the partials fix) |
| Prefill, 100k tokens | 38.7 s | 36.6 s |
| Decode at 100k | 206.6 tok/s | 209.2 tok/s |
| Prefill / decode at 250k | 146 s / 130.7 tok/s | does not fit |
| Short prompts (code sampled / chat sampled / code greedy / chat greedy) | 305 / 219 / 307 / 283 | 321 / 244 / 323 / 298 |

fp8 costs short prompts about 5%, because the decode kernel quantizes the window's rows itself.

**bf16 against TensorFold 0.6.6** ([`evidence/2026-10-12-native-vision/`](evidence/2026-10-12-native-vision/)):
- Serial replies token-identical, 4/4, including a 4,722-token prompt.
- 54/54 verify windows identical in logit bytes.
- Drafted replies identical, 4/4.
- Shared rounds: no reply differs.

**After merging upstream 1.0.9** (engine `e3b72103`, [`evidence/2026-10-13-merge-1.0.9/`](evidence/2026-10-13-merge-1.0.9/)):
the same checks give the same results.
- bf16 against 0.6.6: serial 4/4, windows 54/54, drafted 4/4, and no reply differs in shared rounds.
- fp8: serial, drafted and shared-round replies are identical to before the merge.
- Cache on, bf16: 20/20 equal to 0.6.6, 12 of them after a resume.
- Images: the stress set gives the same results.
- A 105k-token chat's next turn starts in 1.35 s.
- Short-prompt decode: 306 / 219 / 308 / 283 tok/s.

**After merging upstream 1.0.10 and 1.0.11** (engine `4e15fae5`, [`evidence/2026-10-14-merge-1.0.11/`](evidence/2026-10-14-merge-1.0.11/)):
the same checks give the same results again.
- bf16 against 0.6.6: serial 4/4, windows 54/54, drafted 4/4, and no reply differs in shared rounds.
- fp8: serial, drafted and shared-round replies are identical to before.
- Cache on, bf16: 20/20 equal to 0.6.6, 12 of them after a resume.
- Images: the stress set gives the same results.
- A 105k-token chat's next turn starts in 1.36 s.
- Short-prompt decode: 305 / 219 / 306 / 283 tok/s.

Against 0.6.6 on short prompts, native is 11-17% faster: 321 / 244 / 323 / 298 tok/s against 288 / 208 / 277 / 262.
Image chats take 0.2-0.7 s against 0.5-1.6 s.

## Setup

You need:
- an RTX 5090 (the kernels in `kernels/sm120` are for sm_120);
- the CUDA 13 toolkit (`nvcc`) and Zig 0.17.0;
- `hf` (huggingface_hub);
- `libturbojpeg` (`apt install libturbojpeg`) for PIL-exact JPEGs. Without it the engine decodes with stb_image.

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
| `KV_CACHE` | `fp8` | `bf16` is exact against 0.6.6 but holds about half the window: set `CONTEXT=122880` with it |
| `PROMPT_CACHE_GIB` | `8` | host memory for kept prompt states, at most half of `MemAvailable`; `0` turns the cache off |
| `CONTEXT` | `262144` | 65,536 gives 4 streams under fp8 (startup allocation checked; throughput not measured) |
| `PARALLEL` | `auto` | streams at once, as many as the window allows |
| `PORT` / `HOST` | `30000` / `0.0.0.0` | |
| `NAME` / `ALIAS` | `qwen38-27b` / `qwen38-nvfp4` | |
| `MODELS` / `TF_SRC` | `./models` / `./engine` | |

**Requests:**
- There is no API key.
- Thinking is on by default. Turn it off per request with `"chat_template_kwargs": {"enable_thinking": false}`.
- Images go in `image_url` parts as data URLs (PNG, JPEG).
- A request can hold up to 4 images, sharing 4,096 visual tokens. `detail: low` caps one image at 256.

## Kernels

`kernels/sm120` holds the Triton "glue" kernels the native qwen27 engine launches: attention, the fp8 decode
attention, and the drafter's.
- They were captured from TensorFold 0.6.6 on an RTX 5090 and packed ahead of time.
- They are not built from source by `setup.sh`.
- [`tools/capture.sh`](tools/capture.sh) recaptures them. It needs a Python venv at `.venv` with TensorFold 0.6.6, torch and Triton, and the GPU to itself.

## Tools

| Script | What |
|---|---|
| `tools/longbench.py URL MODEL` | long-prompt prefill and decode with a mid-depth needle (the table above) |
| `tools/needle.py URL MODEL_DIR TOKENS DEPTHS` | needle in a haystack at chosen depths, token-exact lengths |
| `tools/longspeed.py` | prefill and decode at one long prompt |
| `tools/image_chats.py PORT OUT` | 5 image chats, greedy, saved for comparison |
| `tools/image_stress.py PORT` | images at a chunk boundary, the largest image, 4 images, a 5th refused, concurrent text |
| `tools/vision-req.json` | one image chat for `curl -d @tools/vision-req.json` |

The decode cells come from TensorFold's own `tools/bench_openai.py`. The quality suite and the decode ruler come from the
[qwen Spark recipe](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks).

## License

MIT for this repository. The model, drafter and TensorFold carry their own licenses.
