| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 33s | drafted 300 tok/s vs undrafted 65 tok/s |
| concurrent | 12/12 | 100.0% | 12s | 4 streams together vs alone |
| streaming | 4/4 | 100.0% | 4s | identical |
| multiturn | 8/8 | 100.0% | 4s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 0s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 3s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 0s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 0s | 3/3 exact |
| tools | 57/60 | 95.0% | 39s | "How do you say 'where is the station' in": got [] |
| json | 30/30 | 100.0% | 4s | all valid |
| ifeval | 131/150 | 87.3% | 305s | loose 131/150, strict 123/150, 150 covered prompts |
| gsm8k | 243/250 | 97.2% | 207s | misses at [12, 45, 62, 85, 93, 119, 205] |
| mgsm | 213/240 | 88.8% | 366s | de 27/30, es 29/30, fr 26/30, ja 28/30, zh 25/30, ru 30/30, sw 22/30, bn 26/30 |
| mmlu | 276/342 | 80.7% | 503s | 57 subjects x 6; weakest college_mathematics 2/6, high_school_mathematics 2/6, professional_law 2/6 |
| humaneval | 0/0 | 0.0% | 63s | failed: sandbox failed: Unable to find image 'tf-qwen38-int4mixed:native-v1.0.12' locally
docker: Error resp |
| repetition | 8/8 | 100.0% | 40s | repeated 4-grams 0.21% mean, max 0.38%; 0 replies hit max_tokens |
| long-context | 55/55 | 100.0% | 1029s | needles 15/15 to 224k, facts 10/10@28k, 10/10@57k, 10/10@114k, 10/10@225k, slowest prefill 1458 tok/s |
| vision | 12/12 | 100.0% | 1s | all probes answered |
| soak | 60/60 | 100.0% | 152s | 3 chats x 20 turns at once beside a decoding stream (5.2k-5.9k-token prompts): 60/60 equal to the same request alone, 0 errors, server answers after; 60/60 concurrent turns resumed a kept state |
| snapshot | 12/12 | 100.0% | 6s | replies written to /storage/recipe-harness/runs/qwen38-27b-5090-exl3-5/20261108-021217-quality-off/suite/replies.json |
