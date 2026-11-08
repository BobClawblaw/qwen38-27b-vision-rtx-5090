| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 30s | drafted 295 tok/s vs undrafted 74 tok/s |
| concurrent | 12/12 | 100.0% | 10s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 7/8 | 87.5% | 4s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 59/60 | 98.3% | 42s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 3s | all valid |
| ifeval | 132/150 | 88.0% | 143s | loose 132/150, strict 129/150, 150 covered prompts |
| gsm8k | 245/250 | 98.0% | 138s | misses at [45, 62, 85, 93, 119] |
| mgsm | 217/240 | 90.4% | 184s | de 28/30, es 29/30, fr 26/30, ja 27/30, zh 27/30, ru 29/30, sw 24/30, bn 27/30 |
| mmlu | 281/342 | 82.2% | 263s | 57 subjects x 6; weakest high_school_mathematics 2/6, high_school_physics 2/6, business_ethics 3/6 |
| humaneval | 154/164 | 93.9% | 66s | pass@1 greedy; failed ['HumanEval_115', 'HumanEval_116', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_140'] ... |
| repetition | 8/8 | 100.0% | 18s | repeated 4-grams 0.25% mean, max 0.95%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 1154s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1315 tok/s |
| vision | 12/12 | 100.0% | 1s | all probes answered |
| soak | 60/60 | 100.0% | 160s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 6s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090-exl3-4/20261108-031856-quality-off/suite/replies.json |
