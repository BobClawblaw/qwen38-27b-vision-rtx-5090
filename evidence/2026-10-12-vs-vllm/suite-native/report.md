| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 31s | drafted 288 tok/s vs undrafted 69 tok/s |
| concurrent | 12/12 | 100.0% | 12s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 7/8 | 87.5% | 4s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 4s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 59/60 | 98.3% | 29s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5-minute timer'})] |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 124/150 | 82.7% | 278s | loose 124/150, strict 116/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 214s | misses at [45, 62, 85, 93, 119, 187, 205, 214] |
| mgsm | 213/240 | 88.8% | 489s | de 27/30, es 29/30, fr 26/30, ja 27/30, zh 27/30, ru 29/30, sw 22/30, bn 26/30 |
| mmlu | 268/342 | 78.4% | 436s | 57 subjects x 6; weakest college_mathematics 1/6, high_school_mathematics 2/6, anatomy 3/6 |
| humaneval | 153/164 | 93.3% | 69s | pass@1 greedy; failed ['HumanEval_108', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_145', 'HumanEval_147'] ... |
| repetition | 8/8 | 100.0% | 35s | repeated 4-grams 0.35% mean, max 0.62%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 791s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1833 tok/s |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 0/60 | 0.0% | 244s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 0/60 concurrent turns resumed a kept state (under 50%: resumes not exercised) |
| snapshot | 12/12 | 100.0% | 6s | replies written to /storage/tensorfold-qwen38-27b-5090/quality-runs/native-full/replies.json |
