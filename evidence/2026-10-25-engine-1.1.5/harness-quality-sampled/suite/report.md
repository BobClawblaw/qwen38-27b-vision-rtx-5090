| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 27s | drafted 335 tok/s vs undrafted 84 tok/s |
| concurrent | 12/12 | 100.0% | 11s | 4 streams together vs alone |
| streaming | 1/4 | 25.0% | 3s | 'Explain how a bloom filter wor' differs |
| multiturn | 7/8 | 87.5% | 3s | recalled 7/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 57/60 | 95.0% | 24s | 'Set a timer for 5 minutes.': got [('set_timer', {'seconds': 300, 'label': '5 minute timer'})] |
| json | 30/30 | 100.0% | 4s | all valid |
| ifeval | 128/150 | 85.3% | 253s | loose 128/150, strict 120/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 209s | misses at [12, 45, 85, 93, 119, 162, 184, 214] |
| mgsm | 212/240 | 88.3% | 332s | de 28/30, es 29/30, fr 28/30, ja 28/30, zh 26/30, ru 28/30, sw 22/30, bn 23/30 |
| mmlu | 276/342 | 80.7% | 401s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, business_ethics 3/6 |
| humaneval | 0/0 | 0.0% | 58s | failed: sandbox failed: permission denied while trying to connect to the docker API at unix:///var/run/docke |
| repetition | 8/8 | 100.0% | 32s | repeated 4-grams 0.44% mean, max 1.14%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 681s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 2064 tok/s |
| vision | 12/12 | 100.0% | 1s | all probes answered |
| soak | 60/60 | 100.0% | 97s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 5s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090/20261025-232444-quality-sampled/suite/replies.json |
