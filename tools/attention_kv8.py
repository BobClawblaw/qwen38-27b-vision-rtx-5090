#!/usr/bin/env python3
"""attention.attention over an fp8 (e4m3) key and value cache, for the native engine's --kv-cache fp8.

The cache holds each (position, key head) row of 256 as e4m3 times one fp32 scale (the row's absolute max over 448).
_shared8 and _tail8 are attention.py's _shared and _tail with those loads: e4m3 widens to bf16 exactly, the key's
scale multiplies its scores and the value's scale its probabilities, so the dots stay bf16 tensor-core dots. The
window's own rows (KN, VN) are quantized in the kernel exactly as their commit will quantize them, so each key's
contribution is the same committed or not. Partials keep attention.py's layout, so its _merge folds them unchanged.

``--out DIR`` checks both against attention.attention over the dequantized cache and records their launches for
triton_aot_manifest.py (one specialization each for W of 1, of a multiple of 16, and of anything else).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0


@triton.jit
def _tile8(q, k, v, ks, vs, m, l, o, valid, scale: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * ks[None, :] * scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot((p * vs[None, :]).to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _fold8(m, l, o, cm, cl, co):
    active = cl > 0.0
    next_m = tl.where(active, tl.maximum(m, cm), m)
    a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    b = tl.where(active, tl.exp(cm - next_m), 0.0)
    return next_m, l * a + cl * b, o * a[:, None] + co * b[:, None]


@triton.jit
def _q8(x):
    """Rows of bf16 as the cache keeps them: e4m3 of x * (448 / amax) widened back to bf16, and s = amax / 448
    (kv_quant's rule: two IEEE divisions a row, one IEEE product an element)."""

    x = x.to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    s = tl.where(amax > 0, tl.math.div_rn(amax, 448.0), 1.0)
    inv = tl.where(amax > 0, tl.math.div_rn(448.0, amax), 1.0)
    return (x * inv[:, None]).to(tl.float8e4nv).to(tl.bfloat16), s


@triton.jit
def _shared8(Q, KC, VC, KS, VS, OFF, STREAM, ITEMS, PO, PM, PL, W, H: tl.constexpr, HK: tl.constexpr,
             D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, SCALE: tl.constexpr, GR: tl.constexpr):
    """_shared over the fp8 cache; OFF holds four offsets a stream: keys and values (bytes), their scales (fp32)."""

    item = tl.program_id(0) // HK
    hk = tl.program_id(0) % HK
    s = tl.load(ITEMS + item * 3)
    first = tl.load(ITEMS + item * 3 + 1)
    code = tl.load(ITEMS + item * 3 + 2)
    start = tl.load(STREAM + s * 4)
    rows = tl.load(STREAM + s * 4 + 1)
    p = tl.load(STREAM + s * 4 + 2)
    groups = ((p + rows + CH - 1) // CH - tl.load(STREAM + s * 4 + 3)) // (GR - 1)
    whole = code >= 0
    chunk = tl.where(whole, code * GR, groups * GR - 1 - code)
    n = tl.where(whole, GR, 1)
    slot = tl.where(whole, code, chunk - groups * (GR - 1))
    if ((chunk + n) * CH <= p) & (code < groups):
        koff = tl.multiple_of(tl.load(OFF + s * 4), 16)
        voff = tl.multiple_of(tl.load(OFF + s * 4 + 1), 16)
        ksoff = tl.load(OFF + s * 4 + 2)
        vsoff = tl.load(OFF + s * 4 + 3)
        rr = first + tl.arange(0, 16)
        ok = rr < rows * G
        node = start + rr // G
        head = hk * G + rr % G
        d = tl.arange(0, D)
        q = tl.load(Q + (node[:, None] * H + head[:, None]) * D + d[None, :], mask=ok[:, None],
                    other=0).to(tl.bfloat16)
        gm = tl.full((16,), float("-inf"), tl.float32)
        gl = tl.zeros((16,), tl.float32)
        go = tl.zeros((16, D), tl.float32)
        m, l, o = gm, gl, go
        for c in range(n):
            key = (chunk + c) * CH + tl.arange(0, 64)
            m = tl.full((16,), float("-inf"), tl.float32)
            l = tl.zeros((16,), tl.float32)
            o = tl.zeros((16, D), tl.float32)
            for t in range(CH // 64):
                ki = key + t * 64
                kk = tl.load(KC + koff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                vv = tl.load(VC + voff + (ki[:, None] * HK + hk) * D + d[None, :]).to(tl.bfloat16)
                ks = tl.load(KS + ksoff + ki * HK + hk)
                vs = tl.load(VS + vsoff + ki * HK + hk)
                m, l, o = _tile8(q, kk, vv, ks, vs, m, l, o, ki < p, SCALE)
            gm, gl, go = _fold8(gm, gl, go, m, l, o)
        if not whole:
            gm, gl, go = m, l, o
        base = (slot * W + node) * H + head
        tl.store(PO + base[:, None] * D + d[None, :], go, mask=ok[:, None])
        tl.store(PM + base, gm, mask=ok)
        tl.store(PL + base, gl, mask=ok)


@triton.jit
def _tail8(Q, KN, VN, KC, VC, KS, VS, OFF, STREAM, ROWS, PATHS, DEPTHS, PO, PM, PL, W, H: tl.constexpr,
           HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr, MAXD: tl.constexpr,
           SCALE: tl.constexpr, GR: tl.constexpr):
    """_tail over the fp8 cache; the row's own path quantized as its commit will, so a key's bits never depend on
    whether it is committed or in the window (nor on drafts, or how streams share a round)."""

    node = tl.program_id(0)
    hk = tl.program_id(1)
    s = tl.load(ROWS + node)
    p = tl.load(STREAM + s * 4 + 2)
    nch = (p + tl.load(STREAM + s * 4 + 1) + CH - 1) // CH
    groups = (nch - tl.load(STREAM + s * 4 + 3)) // (GR - 1)
    chunk = p // CH + tl.program_id(2)
    if chunk < nch:
        koff = tl.multiple_of(tl.load(OFF + s * 4), 16)
        voff = tl.multiple_of(tl.load(OFF + s * 4 + 1), 16)
        ksoff = tl.load(OFF + s * 4 + 2)
        vsoff = tl.load(OFF + s * 4 + 3)
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (node * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0).to(tl.bfloat16)
        depth = tl.load(DEPTHS + node)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        key = chunk * CH + tl.arange(0, 64)
        for t in range(CH // 64):
            logical = key + t * 64
            committed = logical < p
            path_slot = logical - p
            on_path = (path_slot >= 0) & (path_slot < depth)
            path_node = tl.load(PATHS + node * MAXD + path_slot, mask=on_path, other=0)
            kc = tl.load(KC + koff + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0.0)
            vc = tl.load(VC + voff + (logical[:, None] * HK + hk) * D + d[None, :], mask=committed[:, None], other=0.0)
            ks = tl.load(KS + ksoff + logical * HK + hk, mask=committed, other=1.0)
            vs = tl.load(VS + vsoff + logical * HK + hk, mask=committed, other=1.0)
            kn = tl.zeros((64, D), tl.bfloat16)
            vn = tl.zeros((64, D), tl.bfloat16)
            kns = tl.full((64,), 1.0, tl.float32)
            vns = tl.full((64,), 1.0, tl.float32)
            if tl.max(on_path.to(tl.int32), 0) > 0:      # a tile of committed keys only skips the quantizing
                kn, kns = _q8(tl.load(KN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0))
                vn, vns = _q8(tl.load(VN + (path_node[:, None] * HK + hk) * D + d[None, :], mask=on_path[:, None], other=0))
            kk = tl.where(committed[:, None], kc.to(tl.bfloat16), kn)
            vv = tl.where(committed[:, None], vc.to(tl.bfloat16), vn)
            ks = tl.where(committed, ks, kns)
            vs = tl.where(committed, vs, vns)
            m, l, o = _tile8(q, kk, vv, ks, vs, m, l, o, committed | on_path, SCALE)
        base = ((chunk - groups * (GR - 1)) * W + node) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, HK, D) bf16 rows as e4m3 and one fp32 scale a row: the native kv_quant kernel's rule."""

    amax = x.float().abs().amax(-1)
    # tensor by tensor: IEEE division, as kv_quant's __fdiv_rn (torch divides by a Python float as a reciprocal)
    full = torch.full_like(amax, FP8_MAX)
    scale = torch.where(amax > 0, amax / full, torch.ones_like(amax))
    inv = torch.where(amax > 0, full / amax, torch.ones_like(amax))
    q = (x.float() * inv[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


def attention8(q, kn, vn, origin, kc8, vc8, ks, vs, p: "A.Plan", scale: float):
    """attention.attention with the fp8 cache of one stream (offsets from ``origin``, as the engine passes them)."""

    import tensorfold.cuda.kernels.attention as A

    w, h, d = q.shape
    hk = kn.shape[1]
    g = h // hk
    o8 = origin.view(torch.float8_e4m3fn)
    of = origin.view(torch.float32)
    b = origin.data_ptr()
    offs = torch.tensor([[kc8.data_ptr() - b, vc8.data_ptr() - b, (ks.data_ptr() - b) // 4, (vs.data_ptr() - b) // 4]],
                        dtype=torch.int64, device=q.device)
    po = torch.empty((p.chunks, w, h, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((p.chunks, w, h), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    if p.items.shape[0]:
        _shared8[(p.items.shape[0] * hk,)](q, o8, o8, of, of, offs, p.streams, p.items, po, pm, pl, w, H=h, HK=hk, D=d,
                                           G=g, CH=A.CHUNK, SCALE=scale, GR=A.GROUP, num_warps=4, num_stages=1)
    tails = 1 + -(-A.MAX_NODES // A.CHUNK)
    _tail8[(w, hk, tails)](q, kn, vn, o8, o8, of, of, offs, p.streams, p.rows, p.paths, p.depths, po, pm, pl, w, H=h,
                           HK=hk, D=d, G=g, CH=A.CHUNK, MAXD=A.MAX_NODES, SCALE=scale, GR=A.GROUP, num_warps=4,
                           num_stages=1)
    out = torch.empty_like(q)
    A._merge[(w, hk, d // A.MERGE_COLUMNS)](po, pm, pl, out, p.streams, p.rows, w, H=h, D=d, G=g, DS=A.MERGE_COLUMNS,
                                            CH=A.CHUNK, GR=A.GROUP, num_warps=4)
    return out


def check(p_len: int, parents: list[int], dev) -> float:
    """attention8 against attention.attention over the dequantized cache: the worst row's cosine."""

    import tensorfold.cuda.kernels.attention as A

    torch.manual_seed(p_len * 131 + len(parents))
    h, hk, d = 24, 4, 256
    w = len(parents)
    cap = p_len + 64
    k = torch.randn((cap, hk, d), device=dev).mul(3).to(torch.bfloat16)
    v = torch.randn((cap, hk, d), device=dev).to(torch.bfloat16)
    k8, ks = quantize(k)
    v8, vs = quantize(v)
    kd = (k8.float() * ks[..., None]).to(torch.bfloat16)
    vd = (v8.float() * vs[..., None]).to(torch.bfloat16)
    q = torch.randn((w, h, d), device=dev).to(torch.bfloat16)
    kn = torch.randn((w, hk, d), device=dev).mul(3).to(torch.bfloat16)
    vn = torch.randn((w, hk, d), device=dev).to(torch.bfloat16)
    plan = A.plan([parents], [p_len], h // hk, dev)
    scale = d ** -0.5
    kq, kqs = quantize(kn)
    vq, vqs = quantize(vn)
    knd = (kq.float() * kqs[..., None]).to(torch.bfloat16)
    vnd = (vq.float() * vqs[..., None]).to(torch.bfloat16)
    ref = A.attention(q, knd, vnd, torch.tensor([A.offsets([(kd, vd)], dev)], dtype=torch.int64, device=dev), plan,
                      scale=scale)
    got = attention8(q, kn, vn, A.base(dev).view(torch.uint8), k8, v8, ks, vs, plan, scale)
    cos = torch.nn.functional.cosine_similarity(ref.float().view(w * h, d), got.float().view(w * h, d), dim=1)
    if parents == list(range(-1, w - 1)) and w > 1:
        # a chain's first row committed instead: the other rows' outputs must keep their bits
        k8b = torch.cat([k8[:p_len], kq[:1], k8[p_len + 1:]])
        ksb = torch.cat([ks[:p_len], kqs[:1], ks[p_len + 1:]])
        v8b = torch.cat([v8[:p_len], vq[:1], v8[p_len + 1:]])
        vsb = torch.cat([vs[:p_len], vqs[:1], vs[p_len + 1:]])
        plan_b = A.plan([list(range(-1, w - 2))], [p_len + 1], h // hk, dev)
        got_b = attention8(q[1:].contiguous(), kn[1:].contiguous(), vn[1:].contiguous(), A.base(dev).view(torch.uint8),
                           k8b, v8b, ksb, vsb, plan_b, scale)
        same = torch.equal(got[1:], got_b)
        print(f"  committed vs in-window keys: {'bit-identical' if same else 'DIFFER'}", flush=True)
        if not same:
            return 0.0
    return float(cos.min())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", required=True, help="folder of triton_aot_manifest.py")
    a = ap.parse_args()
    sys.path.insert(0, a.tools)
    import triton_aot_manifest as aot

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = aot.Recorder().install()
    rec.detail = True
    dev = torch.device("cuda:0")
    worst = 1.0
    # W of 1 (decode), 16 (a divisible verify window), 7 and 33 (neither); committed keys under and over a group
    cases = [(0, [-1]), (700, [-1]), (5000, [-1]), (9000, list(range(-1, 15))), (3000, [-1, 0, 1, 1, 3, 4, 2]),
             (20000, [-1] + list(range(32))), (300, list(range(-1, 15)))]
    for p_len, parents in cases:
        c = check(p_len, parents, dev)
        worst = min(worst, c)
        print(f"p={p_len} w={len(parents)}: worst row cosine {c:.6f}", flush=True)
    rec.dump(out / "launches.json")
    from triton.runtime.jit import JITFunction
    import gc

    spec = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            spec[f"{fn.fn.__module__}.{fn.fn.__qualname__}"] = {
                "params": [x.name for x in fn.params],
                "do_not_specialize": [x.name for x in fn.params if x.do_not_specialize],
                "no_align": [x.name for x in fn.params if x.do_not_specialize_on_alignment]}
    (out / "jit.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print(f"worst {worst:.6f}")
    return 0 if worst > 0.995 else 1


if __name__ == "__main__":
    raise SystemExit(main())
