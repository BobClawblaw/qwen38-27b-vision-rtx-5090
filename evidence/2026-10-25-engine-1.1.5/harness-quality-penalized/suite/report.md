| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 27s | drafted 326 tok/s vs undrafted 84 tok/s |
| concurrent | 12/12 | 100.0% | 11s | 4 streams together vs alone |
| streaming | 0/4 | 0.0% | 3s | 'Explain how a bloom filter wor' differs |
| multiturn | 7/8 | 87.5% | 4s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 57/60 | 95.0% | 24s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 4s | all valid |
| ifeval | 126/150 | 84.0% | 257s | loose 126/150, strict 119/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 227s | misses at [62, 85, 87, 93, 119, 157, 164, 184] |
| mgsm | 210/240 | 87.5% | 366s | de 27/30, es 29/30, fr 27/30, ja 27/30, zh 26/30, ru 29/30, sw 24/30, bn 21/30 |
| mmlu | 266/342 | 77.8% | 441s | 57 subjects x 6; weakest college_mathematics 1/6, high_school_mathematics 2/6, professional_law 2/6 |
| humaneval | 0/0 | 0.0% | 56s | failed: sandbox failed: Unable to find image 'tf-qwen38-int4mixed:native-v1.0.12' locally
docker: Error resp |
| repetition | 8/8 | 100.0% | 34s | repeated 4-grams 0.19% mean, max 1.16%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 679s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 2065 tok/s |
| vision | 12/12 | 100.0% | 1s | all probes answered |
| soak | 60/60 | 100.0% | 100s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 5s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090/20261026-000635-quality-penalized/suite/replies.json |
