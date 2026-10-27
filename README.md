# Qwen3.8-27B with image input and a 262k window on one RTX 5090

Serve [TensorFold/Qwen3.8-27B-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-27B-MLX-4bit) with the
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) drafter on a single RTX 5090 (sm_120,
32 GB), through [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine on CUDA:
- text and images (PNG, JPEG, WebP) in one OpenAI-compatible API (`image_url` parts);
- the model's full 262,144-token window, with the vision tower loaded, in one stream, or up to 8 streams at shorter
  windows ([Streams](#streams));
- DFlash2 drafts on every reply;
- structured output (`response_format`) and forced tool calls (`tool_choice: "required"` or a named function);
- a prompt cache that resumes a chat's kept state on the next turn;
- a systemd unit.

## Engine

The recipe builds TensorFold [1.1.7](https://github.com/ashhart/TensorFold/releases/tag/v1.1.7) (`59044397`) from
upstream. The four PRs an earlier version carried on a fork branch, [#1079](https://github.com/ashhart/TensorFold/pull/1079)
(the qwen3_5 CUDA engine's decode), [#1080](https://github.com/ashhart/TensorFold/pull/1080) (its prefill),
[#1168](https://github.com/ashhart/TensorFold/pull/1168) (its decode attention on tensor cores, for fp8 and bf16 caches)
and [#1176](https://github.com/ashhart/TensorFold/pull/1176) (prompt matmuls on int8 activations), landed in 1.1.4, so
the engine is upstream's own.

On 1.1.7 (2026-10-27, a build of the pinned commit with `setup.sh`'s flags,
[`evidence/2026-10-27-engine-1.1.7/`](evidence/2026-10-27-engine-1.1.7/)) only the harness's `quick` plan and the
suite's vision check were run, against 1.1.5's `quick` run ([`harness-quick/`](evidence/2026-10-25-engine-1.1.5/harness-quick/)):

| One stream, fresh server | 1.1.5 | 1.1.7 |
|---|---:|---:|
| prose / code / structured / list, tok/s | 278.3 / 538.1 / 352.6 / 337.3 | 274.1 / 537.7 / 352.3 / 337.7 |
| first token | 0.06 s | 0.06 s |

Every serving check is unchanged: drafted equals undrafted 12/12, 4 streams together equal alone 12/12, soak 60/60,
structured output 18/18, tools 5/5, 9/9 seeded replies equal on repeat and undrafted, a resumed turn equals a cold one;
vision 12/12 ([`harness-quick/`](evidence/2026-10-27-engine-1.1.7/harness-quick/),
[`harness-vision/`](evidence/2026-10-27-engine-1.1.7/harness-vision/)). The quality suite, the long-prompt speeds and
the streams plan were not run on 1.1.7. What 1.1.7 changes for this recipe, from its changelog:
- `presence_penalty` and `frequency_penalty` are applied ([#1346](https://github.com/ashhart/TensorFold/pull/1346));
  1.1.5 ignored them, so requests that send them now get penalized draws (the suite's penalized run is below).
- Paged streams (`TF_QWEN_KV_PAGES`) take the same attention as whole windows
  ([#1373](https://github.com/ashhart/TensorFold/pull/1373)); see [Streams](#streams).
- An int8 prompt chunk that falls back to bf16 is counted and logged once
  ([#1371](https://github.com/ashhart/TensorFold/pull/1371)).

The numbers in [Against the previous versions](#against-the-previous-versions-of-this-recipe), the quality
runs marked 1.1.5 and [Streams](#streams) are 1.1.5's (`84e0d560`), measured on a build of that commit with `setup.sh`'s
flags ([`evidence/2026-10-25-engine-1.1.5/`](evidence/2026-10-25-engine-1.1.5/)). The rest of this document was measured
on an earlier engine, `18d93451` ([`evidence/2026-10-22-lone-bands/`](evidence/2026-10-22-lone-bands/)); the harness's
`quick` plan on 1.1.5 ([`harness-quick/`](evidence/2026-10-25-engine-1.1.5/harness-quick/)) gives the same result on
every serving check: drafted equals undrafted 12/12, 4 streams together equal alone 12/12, soak 60/60 with every turn
resuming a kept state, structured output 18/18, tools 5/5, 9/9 seeded replies equal on repeat and undrafted.
- Prompts run with int8 activations by default (`PROMPT_PRECISION=int8`, `--prompt-precision int8`): each 4-bit
  projection's inputs are rounded to int8 with one scale a 64-input group, and the matmul runs on the int8 tensor
  cores. Prefill takes 20-26% less time than with bf16 activations. Replies differ slightly from bf16 prompts' (the
  measured divergence is in [#1176](https://github.com/ashhart/TensorFold/pull/1176)); drafted replies still equal
  undrafted ones. `PROMPT_PRECISION=bf16` keeps the exact bf16 path.
- A long prompt no longer slows the short replies after it (`f15dad20` in the previous version, `af9dc338` in 1.1.4; see
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

The previous version, `18d93451` (2026-10-22), built TensorFold 1.1.2 with #1079, #1080, #1168 and #1176 from a fork
branch; it is at [`2147ce9`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/2147ce9), with its evidence
and its own comparison against `0d3322a1` and `d6260feb`; the qwen27 engine on TensorFold 1.0.11 that the recipe
served until 2026-10-21 (at [`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52)) is
compared there too. In the table below, `18d93451`'s fixed-prompt and ruler cells are its 2026-10-22 measurements, taken
with the same prompts and settings as 1.1.5's; its long-prompt cells were measured again on 2026-10-26 beside 1.1.5, from
the same binary, cold.

Same card, fp8 caches, int8 prompts, 262,144-token window, vision loaded, one stream, **prompt cache off** for the
prompt rows. The fixed prompts are the first 108,000 and 335,000 characters of a frozen TensorFold source tree
(`corpus-bcd433c0`; [`tools/fixedlen.py`](tools/fixedlen.py) with `TF_SRC` pointed at it), 512 greedy tokens each,
two runs; the long prompts are [`tools/longspeed.py`](tools/longspeed.py)'s over the same tree, one run (two at 250k)
([`speed/`](evidence/2026-10-25-engine-1.1.5/speed/)):

| | 1.1.5 | `18d93451` |
|---|---:|---:|
| Fixed 35,554-token prompt, 512 tokens, whole request | 10.08-10.14 s | **9.94-9.95 s** |
| Its first token | 8.11-8.17 s | **7.85-7.87 s** |
| Fixed 109,537-token prompt, the same | 38.26-38.30 s | **37.47-37.59 s** |
| Its first token | 35.57-35.62 s | **34.63-34.65 s** |
| 100,024-token prompt, cold: prefill / decode | 31.3 s / **222.2-224.9 tok/s** | **30.4 s** / 209.8 tok/s |
| 250,024-token prompt, cold: prefill / decode | 127.2-127.5 s / **147.8-153.6 tok/s** | **125.0 s** / 142.1 tok/s |
| Harness decode ruler, fresh server (prose / code / structured / list) | **278.3 / 538.1 / 352.6 / 337.3** | 257.7 / 497.7 / 326.3 / 312.5 |
| The ruler right after a 512-token prose reply to a cold 245k-token prompt (harness `after-long` plan) | **277.9 / 537.5 / 352.6 / 336.4** | 257.7 / 498.2 / 326.0 / 312.2 |

So 1.1.5 reads a prompt 2-3% slower than `18d93451` and decodes 4-8% faster after a long one and about 8% faster on
short chats; replies are equal on every serving check (above). Most of the prompt gap is the prompt pass itself, not the
int8 path: with `PROMPT_PRECISION=bf16` on both engines the first token comes 1.5-2% later on 1.1.5 (10.77 s against
10.61 s at 35k, 43.76 against 43.13 s at 110k, 38.9 against 38.1 s at 100k), where 1.1.5 runs a prompt a cut at a time
so that decoding streams keep their rounds between the cuts ([#1188](https://github.com/ashhart/TensorFold/pull/1188));
with int8 prompts the gap is about a point wider (8.11 against 7.84 s, 35.54 against 34.65 s, 31.2 against 30.4 s),
consistent with 1.1.5's int8 quantizer now falling back to bf16 for a chunk with a scale overflow, which 1.1.4's did not
([`speed/prefill-cause-bf16-vs-int8-both-engines.txt`](evidence/2026-10-25-engine-1.1.5/speed/prefill-cause-bf16-vs-int8-both-engines.txt)).
The previous README gave 97.5 s for the 250k prompt on `18d93451`: that run followed the 100k one with the prompt cache
on and resumed about 98k tokens from the kept state (the server log of the same procedure on 1.1.5 reads
`prompt=250024 cached=98304`), so it was not a cold number; cold, the engine takes 125.0 s.

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
| Code, sampled (T=1) | **319.3** | 113.2 |
| Chat, sampled (T=1) | **247.1** | 110.2 |
| Code, greedy | **320.0** | 119.2 |
| Chat, greedy | **299.1** | 114.1 |
| First token (chat) | **0.05 s** | 0.06 s |

**Long prompts** ([`tools/longbench.py`](tools/longbench.py)):
- Source code, with a fresh nonce at the top so no prefix cache can answer.
- A passphrase line hidden at mid-depth.
- A 512-token greedy reply, thinking off.

| Prompt (this recipe / vLLM) | this recipe: prefill | vLLM: prefill | this recipe: decode | vLLM: decode | needle |
|---:|---:|---:|---:|---:|---|
| 33,761 / 33,916 | 7.6 s (4,432 tok/s) | **6.5 s** (5,192 tok/s) | **196.2** tok/s | 69.9 tok/s | both found |
| 104,608 / 105,334 | 33.2 s (3,152 tok/s) | **33.1 s** (3,187 tok/s) | **156.1** tok/s | 36.0 tok/s | both found |
| 214,896 / 214,767 | **99.3 s** (2,163 tok/s) | 114.8 s (1,871 tok/s) | **140.0** tok/s | 21.3 tok/s | both found |

**Images:** one image chat (a 640x400 PNG, 200-token cap, greedy) takes 0.46-0.60 s end to end on this recipe (the first of five runs is the slowest)
against 0.76-0.95 s on vLLM. Both read the image's text correctly.

**Quality** (the [qwen quality suite](https://github.com/BobClawblaw/qwen38-flashnext-int4mixed-2x-dgx-sparks/tree/main/quality),
through the [recipe harness](https://github.com/BobClawblaw/recipe-harness), thinking off). The int8 column is this
build's fresh `setup.sh` build ([`harness-quality/`](evidence/2026-10-22-lone-bands/harness-quality/); HumanEval in a
second run, `suite-humaneval/`, because the first could not reach the sandbox). Every score equals the earlier int8
engine's ([`evidence/2026-10-22-int8-prompts/harness-quality/`](evidence/2026-10-22-int8-prompts/harness-quality/)),
and so do the replies, except two of the 12 vision probes worded differently ("Yellow" for "yellow", "showing" for
"containing"). The two older columns' runs say "Yellow" as well. Rerun on this build, these two probes did not give
those words again: 107 replies over 9 server starts (fresh, after each suite check alone, after the whole suite in its
order, and beside other streams, drafted and undrafted) all said "yellow" and "containing". What makes a suite run land
on the other word is not known. The bf16 column is the previous version's engine, `50c05658` on 1.1.1
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

On 1.1.5 the same suite gives the same score on every check ([`harness-quality-off/`](evidence/2026-10-25-engine-1.1.5/harness-quality-off/),
2026-10-25): IFEval 129/150 loose and 121 strict, GSM8K 241/250, MGSM 215/240, MMLU 270/342, tools 58/60, json 30/30,
repetition 8/8, long context 55/55, vision 12/12, soak 60/60, drafts 12/12, concurrent 12/12, and HumanEval 152/164.
Two more runs of the suite on the same build used the Qwen3.8 model card's thinking-off sampling (temperature 0.7, top_p
0.8, top_k 20, seed 42), without and with its `presence_penalty` 1.5: IFEval / GSM8K / MGSM / MMLU / tools read 128 / 242
/ 212 / 276 / 57 without the penalty and 126 / 242 / 210 / 266 / 57 with it, HumanEval 153/164 and 154/164 (sampled, seed 42)
([`harness-quality-sampled/`](evidence/2026-10-25-engine-1.1.5/harness-quality-sampled/),
[`harness-quality-penalized/`](evidence/2026-10-25-engine-1.1.5/harness-quality-penalized/)). The sampling and the
penalty move each score by a few answers either way, within run-to-run noise. TensorFold 1.1.5 itself ignores
`presence_penalty` and `frequency_penalty` (its server reads neither field): a request that sends them gets unpenalized
draws.
The penalized run above used a TensorFold build with the penalties added, which is being prepared for upstream. They
landed in 1.1.7 ([#1346](https://github.com/ashhart/TensorFold/pull/1346)), which this recipe now serves; the suite has
not been run on 1.1.7.

### The checkpoint, against the NVFP4 exports

The MLX 4-bit checkpoint is not the closest 4-bit export to the bf16 model. Teacher-forced on three 700-token sequences
(a chat in the model's template, a Python source file, technical prose), with the bf16 checkpoint's fp32 logits on the
CPU as the truth and each candidate's logits from the 1.1.5 engine (`tf-qwen35-cuda step`; [`kl/`](evidence/2026-10-25-engine-1.1.5/kl/)):

| checkpoint | mean KL(bf16 ‖ this), chat / code / prose | top-1 agreement | weights (language model) |
|---|---:|---:|---:|
| TensorFold/Qwen3.8-27B-MLX-4bit (this recipe) | 0.0285 / 0.109 / 0.0579 | 95.9% / 94.7% / 86.4% | 14.09 GiB |
| lmstudio-community/Qwen3.8-27B-MLX-4bit | 0.0293 / 0.108 / 0.0576 | 95.7% / 94.3% / 86.7% | 14.09 GiB |
| nvidia/Qwen3.8-27B-NVFP4 (ModelOpt: NVFP4 MLPs, FP8 attention and DeltaNet projections) | 0.0174 / 0.0446 / 0.0205 | 97.7% / 96.7% / 92.7% | 18.77 GiB |
| unsloth/Qwen3.8-27B-NVFP4 (compressed-tensors) | 0.0164 / 0.0593 / 0.0158 | 97.4% / 95.7% / 93.0% | 20.16 GiB |

The NVFP4 exports sit about twice as close to bf16, because they keep the attention and DeltaNet projections in FP8.
On this card that costs the window: nvidia's export loads at 22.79 GiB and refuses a 262,144-token stream (9.02 GiB
needed, 7.13 GiB left); it serves 131,072 with vision and the drafter at 28.6 GB in use. Served that way through the
same suite ([`quality-nvfp4/`](evidence/2026-10-25-engine-1.1.5/quality-nvfp4/)) it reads IFEval 126/150, GSM8K 242/250,
MGSM 215/240, MMLU 270/342, tools 60/60, json 30/30, vision 12/12, soak 60/60, HumanEval 153/164, and fails the
long-context check past its window: no gain the suite can see, for half the context. The recipe stays on MLX 4-bit.

The harness's `quick` plan on the same build ([`harness-quick/`](evidence/2026-10-22-lone-bands/harness-quick/)):
- **drafts** 12/12, **concurrent** 12/12 and **soak** 60/60 (all 60 turns resumed a kept state);
- **structured:** 18/18 replies obey their format, drafted equals plain 12/12, and a bad regex is refused with a 400;
- **tools:** 5/5 tool calls right: `auto`, `required`, and a named function plain, with thinking on and streamed;
- **sampling:** 9/9 seeded replies equal on repeat and undrafted;
- **decode ruler** (prose / code / structured / list): 257.7 / 497.7 / 326.3 / 312.5 tok/s fresh, 253.3 / 496.4 /
  325.5 / 312.3 after the checks.

The engines score within a few answers of each other: int8 and bf16 prompts on the same weights, and different
4-bit quantizations of the same model.

In short, against this vLLM configuration only:
- **Decode:** this recipe measured 2-3x faster at short prompts and 3-7x at long ones. Against vLLM with MTP-3, as
  reported elsewhere, the gap would be smaller; we have not measured how much.
- **Prefill:** vLLM reads a 34k-token prompt faster, the two are level at 105k, and this recipe is faster at 215k.
  First token on a short chat is 0.05 s here and 0.06 s on vLLM.
- **Quality:** the same within a few answers either way. This recipe returns reasoning in `reasoning_content`; vLLM, as configured, does not.
- **Prompt cache:** here a 105k-token chat's next turn starts in 1.14 s instead of 33.6 s. vLLM does not report cached
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
  | 1 | 104,571 | 0 | 33.6 s |
  | 2 | 104,629 | 102,400 | 1.14 s |
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

- **Needle in 255,054 tokens** of source code at depths 0.10, 0.33, 0.50 and 0.90: found 4/4, 129-130 s a prompt.
- **Prefill and decode** ([`tools/longspeed.py`](tools/longspeed.py), 512 tokens): 100,024 tokens in 30.9 s, then
  215.5 tok/s; 250,024 tokens in 97.5 s, then 145.0 tok/s. The 250k run resumed about 98k tokens from the prompt cache
  (it followed the 100k run); cold, this engine reads it in 125.0 s (see [Against the previous versions](#against-the-previous-versions-of-this-recipe)).
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
| 100k-token prompt (`longspeed.py`): prefill / decode | 30.9 s / 215.5 tok/s | 29.5 s / 198.0 tok/s |
| Short prompts (code sampled / chat sampled / code greedy / chat greedy) | 319 / 247 / 320 / 299 | 300 / 245 / 325 / 298 |

Not rerun on this engine: fp8 against bf16 teacher-forced KL, the vision tower against fp32
references, and token exactness against TensorFold 0.6.6. The previous engine's results are in the evidence folders at
[`f9caf52`](https://github.com/BobClawblaw/qwen38-27b-vision-rtx-5090/tree/f9caf52).

### Streams

One stream at the full window is the default. With a shorter window the card holds several, each with its own whole KV
window, and the engine shares its rounds between them. Eight streams at 262,144 do not fit, not even on pages: the
engine refuses with 6.38 GiB needed for seven more streams' fixed parts, state runs and a page each, against the 2.75 GiB
one 262k window leaves. The harness's `streams` plan on 1.1.5 (the
decode ruler at 1, 2, 4 and 8 users, 256 tokens a reply, median of 3; then 12 chats four at a time against the same
chats alone):

| | 1 user | 2 users | 4 users | 8 users |
|---|---:|---:|---:|---:|
| `CONTEXT=8192 PARALLEL=8`, prose / code tok/s over all streams ([`harness-streams-8192x8-whole/`](evidence/2026-10-25-engine-1.1.5/harness-streams-8192x8-whole/)) | 274.6 / 537.6 | 448.1 / 739.0 | 573.2 / 970.6 | 719.3 / 1170.3 |
| its first token | 0.06 s | 0.09 s | 0.15 s | 0.26-0.27 s |
| `CONTEXT=32768 PARALLEL=4` ([`harness-streams-32768x4-whole/`](evidence/2026-10-25-engine-1.1.5/harness-streams-32768x4-whole/)) | 278.2 / 537.5 | 448.8 / 741.8 | 572.9 / 966.4 | four lanes: the rest queue |

Both configurations keep the 12 chats four at a time equal to the same chats alone (12/12) and drafted equal to
undrafted (12/12). `nvidia-smi` shows 25,694 MiB in use at 8192 x 8 and 25,760 MiB at 32768 x 4 after loading. The
ruler's aggregate is the sum of the streams' decode rates, so a cell with more users than lanes would overcount the
queued ones and is left out.

1.1.5 can also give the streams beside the first pages of one pool (`TF_QWEN_KV_PAGES=256`), which fits more streams at
long windows (at 65,536 x 8: 1,095 pages, 8.69 GiB). On this card that path does not keep replies exact: the 12 chats
four at a time equal the same chats alone in 6 to 8 of 12 cases, against 12 of 12 with whole windows, for fp8 and
bf16 caches and int8 and bf16 prompts alike, while drafted still equals undrafted
([`harness-together-16384x4-paged/`](evidence/2026-10-25-engine-1.1.5/harness-together-16384x4-paged/) against
[`harness-together-16384x4-whole/`](evidence/2026-10-25-engine-1.1.5/harness-together-16384x4-whole/), and
[`harness-streams-65536x8-paged/`](evidence/2026-10-25-engine-1.1.5/harness-streams-65536x8-paged/)). 1.1.7's changelog
reports this fixed ([#1373](https://github.com/ashhart/TensorFold/pull/1373): paged streams take the whole window's
attention); it has not been measured here on 1.1.7, so the recipe still leaves it unset.

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
245,333-token prompt, and the ruler again. This version reads 257.7 / 498.2 / 326.0 / 312.2 after the long reply
against 257.5 / 496.1 / 326.2 / 312.4 fresh; `0d3322a1` read 233.1 / 371.0 / 326.1 / 312.3 against 256.8 / 496.1 /
324.8 / 312.1 ([`harness-after-long/`](evidence/2026-10-22-lone-bands/harness-after-long/),
[`harness-after-long-0d3322a1/`](evidence/2026-10-22-lone-bands/harness-after-long-0d3322a1/)). After the 37-minute
quality suite this version's ruler read 257.2 / 497.7 / 326.2 / 312.5 against 257.1 / 497.1 / 325.8 / 312.1 before it
([`harness-quality/`](evidence/2026-10-22-lone-bands/harness-quality/)). The fix landed upstream in 1.1.4, and 1.1.5 holds
it: after a cold 245,304-token prompt (124.6 s, 1,969 tok/s) its ruler reads 277.9 / 537.5 / 352.6 / 336.4 against 278.6 /
538.9 / 352.8 / 338.0 fresh, and after the thinking-off quality suite 277.9 / 537.5 / 352.4 / 336.5 against 279.1 / 538.0 /
353.0 / 337.9 before it ([`harness-after-long/`](evidence/2026-10-25-engine-1.1.5/harness-after-long/),
[`harness-quality-off/`](evidence/2026-10-25-engine-1.1.5/harness-quality-off/)).

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
| `PROMPT_PRECISION` | `int8` | the prompt matmuls' activations; `bf16` is exact and takes 25-36% longer to read a prompt |
| `PROMPT_CACHE_GIB` | `8` | host memory for kept prompt states, at most half of `MemAvailable`; `0` turns the cache off |
| `CONTEXT` | `262144` | the window |
| `PARALLEL` | `auto` | streams at once: one at 262,144. `PARALLEL=N` with a shorter `CONTEXT` serves N streams in shared rounds, each with a whole window ([Streams](#streams)); `CONTEXT=32768 PARALLEL=4` and `CONTEXT=8192 PARALLEL=8` are measured. Leave `TF_QWEN_KV_PAGES` unset on this engine |
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
