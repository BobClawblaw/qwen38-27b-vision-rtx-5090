| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 33s | drafted 269 tok/s vs undrafted 65 tok/s |
| concurrent | 12/12 | 100.0% | 13s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 7/8 | 87.5% | 4s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 59/60 | 98.3% | 32s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 130/150 | 86.7% | 284s | loose 130/150, strict 123/150, 150 covered prompts |
| gsm8k | 243/250 | 97.2% | 193s | misses at [12, 62, 85, 93, 119, 205, 241] |
| mgsm | 210/240 | 87.5% | 345s | de 28/30, es 29/30, fr 26/30, ja 26/30, zh 27/30, ru 29/30, sw 21/30, bn 24/30 |
| mmlu | 277/342 | 81.0% | 460s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, professional_law 2/6 |
| humaneval | 152/164 | 92.7% | 77s | pass@1 greedy; failed ['HumanEval_10', 'HumanEval_115', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_134'] ... |
| repetition | 8/8 | 100.0% | 38s | repeated 4-grams 0.30% mean, max 0.89%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 843s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1734 tok/s |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 60/60 | 100.0% | 134s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 6s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090-mlx5/20261108-042804-quality-off/suite/replies.json |
