| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 34s | drafted 267 tok/s vs undrafted 66 tok/s |
| concurrent | 12/12 | 100.0% | 13s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 8/8 | 100.0% | 4s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 60/60 | 100.0% | 31s | all exact |
| json | 30/30 | 100.0% | 5s | all valid |
| ifeval | 126/150 | 84.0% | 271s | loose 126/150, strict 118/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 218s | misses at [62, 85, 87, 93, 119, 157, 162, 241] |
| mgsm | 215/240 | 89.6% | 360s | de 26/30, es 29/30, fr 27/30, ja 27/30, zh 26/30, ru 30/30, sw 23/30, bn 27/30 |
| mmlu | 270/342 | 78.9% | 478s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, professional_law 2/6 |
| humaneval | 0/0 | 0.0% | 59s | failed: sandbox failed: permission denied while trying to connect to the docker API at unix:///var/run/docke |
| repetition | 8/8 | 100.0% | 36s | repeated 4-grams 0.23% mean, max 0.63%; 0 replies hit max_tokens |
| long-context | 0/0 | 0.0% | 230s | failed: HTTP Error 400: Bad Request |
| vision | 12/12 | 100.0% | 2s | all probes answered |
| soak | 60/60 | 100.0% | 122s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 7s | replies written to /storage/qwen38-kl/quality-nvfp4/replies.json |
