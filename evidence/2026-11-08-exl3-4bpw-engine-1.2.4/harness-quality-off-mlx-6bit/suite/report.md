| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 37s | drafted 241 tok/s vs undrafted 58 tok/s |
| concurrent | 12/12 | 100.0% | 14s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 5s | identical |
| multiturn | 8/8 | 100.0% | 3s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 4s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 59/60 | 98.3% | 32s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 6s | all valid |
| ifeval | 131/150 | 87.3% | 323s | loose 131/150, strict 122/150, 150 covered prompts |
| gsm8k | 240/250 | 96.0% | 228s | misses at [45, 62, 85, 93, 119, 125, 159, 201, 241, 244] |
| mgsm | 213/240 | 88.8% | 373s | de 28/30, es 29/30, fr 26/30, ja 28/30, zh 27/30, ru 29/30, sw 22/30, bn 24/30 |
| mmlu | 277/342 | 81.0% | 529s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, business_ethics 3/6 |
| humaneval | 154/164 | 93.9% | 83s | pass@1 greedy; failed ['HumanEval_116', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_145', 'HumanEval_147'] ... |
| repetition | 8/8 | 100.0% | 42s | repeated 4-grams 0.51% mean, max 1.07%; 0 replies hit max_tokens |
| long-context | 0/0 | 0.0% | 239s | failed: HTTP Error 400: Bad Request |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 60/60 | 100.0% | 141s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 7s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090-mlx6/20261108-054556-quality-off/suite/replies.json |
