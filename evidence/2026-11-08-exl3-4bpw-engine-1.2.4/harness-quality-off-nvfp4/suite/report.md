| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 34s | drafted 259 tok/s vs undrafted 65 tok/s |
| concurrent | 12/12 | 100.0% | 13s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 8/8 | 100.0% | 4s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 1s | 3/3 exact |
| tools | 60/60 | 100.0% | 30s | all exact |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 126/150 | 84.0% | 273s | loose 126/150, strict 118/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 201s | misses at [62, 85, 87, 93, 119, 157, 162, 241] |
| mgsm | 215/240 | 89.6% | 331s | de 26/30, es 29/30, fr 27/30, ja 27/30, zh 26/30, ru 30/30, sw 23/30, bn 27/30 |
| mmlu | 270/342 | 78.9% | 466s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, professional_law 2/6 |
| humaneval | 153/164 | 93.3% | 65s | pass@1 greedy; failed ['HumanEval_103', 'HumanEval_108', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_145'] ... |
| repetition | 8/8 | 100.0% | 36s | repeated 4-grams 0.23% mean, max 0.63%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 769s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1877 tok/s |
| vision | 0/12 | 0.0% | 0s | red-circle-blue-square: 'HTTP Error 400: Bad Request' |
| soak | 60/60 | 100.0% | 122s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 7s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090-nvfp4/20261108-065235-quality-off/suite/replies.json |
