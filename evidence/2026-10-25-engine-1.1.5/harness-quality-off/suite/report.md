| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 26s | drafted 343 tok/s vs undrafted 84 tok/s |
| concurrent | 12/12 | 100.0% | 10s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 3s | identical |
| multiturn | 7/8 | 87.5% | 3s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 58/60 | 96.7% | 23s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 3s | all valid |
| ifeval | 129/150 | 86.0% | 238s | loose 129/150, strict 121/150, 150 covered prompts |
| gsm8k | 241/250 | 96.4% | 203s | misses at [12, 45, 62, 85, 87, 93, 119, 159, 214] |
| mgsm | 215/240 | 89.6% | 310s | de 29/30, es 29/30, fr 27/30, ja 28/30, zh 26/30, ru 29/30, sw 23/30, bn 24/30 |
| mmlu | 270/342 | 78.9% | 399s | 57 subjects x 6; weakest college_mathematics 1/6, high_school_mathematics 2/6, abstract_algebra 3/6 |
| humaneval | 0/0 | 0.0% | 52s | failed: sandbox failed: permission denied while trying to connect to the docker API at unix:///var/run/docke |
| repetition | 8/8 | 100.0% | 31s | repeated 4-grams 0.46% mean, max 1.12%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 683s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 2057 tok/s |
| vision | 12/12 | 100.0% | 1s | all probes answered |
| soak | 60/60 | 100.0% | 96s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 5s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090/20261025-224232-quality-off/suite/replies.json |
