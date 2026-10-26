"""KL(truth || candidate) per row, top-1 agreement and both perplexities on the next id.

The truth is truth.py's fp32 .npy (rows x vocab); the candidate is tf-qwen35-cuda step's raw bf16 rows (its padded
vocabulary cut to the truth's width).
usage: score.py <truth.npy> <candidate.bin> <ids file> [json out]
"""
import json
import sys

import numpy as np


def bf16_rows(path: str, vocab_pad: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(-1, vocab_pad)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


truth = np.load(sys.argv[1], mmap_mode="r")
ids = [int(t) for t in open(sys.argv[3]).read().split(",")]
rows, vocab = truth.shape
raw_len = np.fromfile(sys.argv[2], dtype=np.uint16).size
pad = raw_len // rows
if pad * rows != raw_len or pad < vocab:
    raise SystemExit(f"candidate has {raw_len} words: not {rows} rows of at least {vocab}")
cand = bf16_rows(sys.argv[2], pad)[:, :vocab]

kl, agree, nll_t, nll_c = [], 0, [], []
for r0 in range(0, rows, 64):
    lt = log_softmax(np.asarray(truth[r0:r0 + 64]))
    lc = log_softmax(cand[r0:r0 + 64])
    kl += list((np.exp(lt) * (lt - lc)).sum(axis=-1))
    agree += int((lt.argmax(-1) == lc.argmax(-1)).sum())
    for i in range(lt.shape[0]):
        r = r0 + i
        if r + 1 < len(ids):
            nll_t.append(-lt[i, ids[r + 1]])
            nll_c.append(-lc[i, ids[r + 1]])
kl = np.array(kl)
out = {"rows": rows, "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
       "top1": agree / rows, "ppl_truth": float(np.exp(np.mean(nll_t))), "ppl_cand": float(np.exp(np.mean(nll_c)))}
print(json.dumps(out))
if len(sys.argv) > 4:
    json.dump(out, open(sys.argv[4], "w"), indent=1)
