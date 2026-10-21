| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 29s | drafted 305 tok/s vs undrafted 74 tok/s |
| concurrent | 12/12 | 100.0% | 11s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 7/8 | 87.5% | 3s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 4s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 59/60 | 98.3% | 30s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5-minute timer'})] |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 126/150 | 84.0% | 263s | loose 126/150, strict 119/150, 150 covered prompts |
| gsm8k | 238/250 | 95.2% | 201s | misses at [12, 45, 62, 85, 87, 93, 119, 157, 187, 205] |
| mgsm | 214/240 | 89.2% | 318s | de 28/30, es 29/30, fr 28/30, ja 27/30, zh 25/30, ru 29/30, sw 22/30, bn 26/30 |
| mmlu | 270/342 | 78.9% | 415s | 57 subjects x 6; weakest high_school_mathematics 2/6, professional_law 2/6, business_ethics 3/6 |
| humaneval | 0/0 | 0.0% | 66s | failed: sandbox failed: permission denied while trying to connect to the docker API at unix:///var/run/docke |
| repetition | 8/8 | 100.0% | 33s | repeated 4-grams 0.55% mean, max 1.13%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 809s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1783 tok/s |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 60/60 | 100.0% | 120s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 6s | replies written to <harness>/runs/qwen38-27b-5090/20261021-073551-full/suite/replies.json |
