| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 28s | drafted 308 tok/s vs undrafted 75 tok/s |
| concurrent | 12/12 | 100.0% | 11s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 7/8 | 87.5% | 3s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 4s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 59/60 | 98.3% | 29s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5-minute timer'})] |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 130/150 | 86.7% | 253s | loose 130/150, strict 122/150, 150 covered prompts |
| gsm8k | 240/250 | 96.0% | 197s | misses at [12, 40, 45, 62, 85, 93, 119, 187, 205, 241] |
| mgsm | 213/240 | 88.8% | 306s | de 28/30, es 29/30, fr 25/30, ja 28/30, zh 25/30, ru 29/30, sw 22/30, bn 27/30 |
| mmlu | 271/342 | 79.2% | 403s | 57 subjects x 6; weakest high_school_mathematics 2/6, business_ethics 3/6, college_chemistry 3/6 |
| humaneval | 151/164 | 92.1% | 72s | pass@1 greedy; failed ['HumanEval_108', 'HumanEval_116', 'HumanEval_127', 'HumanEval_129', 'HumanEval_132', 'HumanEval_140'] ... |
| repetition | 8/8 | 100.0% | 32s | repeated 4-grams 0.37% mean, max 0.92%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 806s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1786 tok/s |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 60/60 | 100.0% | 115s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 6s | replies written to <harness>/runs/qwen38-27b-5090/20261021-172825-quality-off/suite/replies.json |
