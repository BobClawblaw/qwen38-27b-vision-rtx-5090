#!/usr/bin/env python3
"""Fixtures for the native Qwen vision path, from TensorFold 0.6.6's --vision frontend and tower on CUDA.

For a fixed image set: each image's file, its decoded RGB (images.py), its patches and grid (QwenImageProcessor's
processor at 0.6.6's per-image cap), and the tower's merged rows (QwenCudaVision, bf16, flash attention). With
--fp32 also the tower's rows in fp32 (math attention), the reference both towers are measured against: copy
NAME.ref32 to OUT/NAME.ref.f32 and run ``tf-qwen35-cuda MODEL vision OUT TOKENS NAME.f32@HxW...`` on the patches.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np


def images():
    """(name, encoded bytes) for the set: PNG layouts, sizes from tiny to large, an EXIF-rotated JPEG."""
    from PIL import Image, ImageDraw

    out = []
    shapes = Image.new("RGB", (517, 389), "white")
    d = ImageDraw.Draw(shapes)
    d.ellipse((30, 40, 230, 240), fill=(220, 20, 20))
    d.rectangle((280, 60, 490, 260), fill=(20, 60, 220))
    d.text((60, 300), "ORANGE 7351", fill="black")
    rng = np.random.default_rng(7)
    y, x = np.mgrid[0:768, 0:1024]
    photo = np.stack([(x / 4) % 256, (y / 3) % 256, ((x + y) / 7) % 256], -1) + rng.normal(0, 12, (768, 1024, 3))
    photo = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8))
    large = photo.resize((2000, 1500))
    tiny = shapes.resize((20, 15))
    logo = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
    ImageDraw.Draw(logo).ellipse((8, 8, 88, 88), fill=(30, 160, 90, 200))
    palette = shapes.convert("P", palette=Image.Palette.ADAPTIVE, colors=16)
    palette.info["transparency"] = 0
    gray = photo.convert("L")

    def png(img, **kw):
        b = io.BytesIO()
        img.save(b, "PNG", **kw)
        return b.getvalue()

    def jpeg(img, exif=None):
        b = io.BytesIO()
        img.save(b, "JPEG", quality=90, **({"exif": exif} if exif else {}))
        return b.getvalue()

    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 on display
    out += [("shapes.png", png(shapes)), ("photo.png", png(photo)), ("large.png", png(large)),
            ("tiny.png", png(tiny)), ("logo.png", png(logo)), ("palette.png", png(palette)), ("gray.png", png(gray)),
            ("photo.jpg", jpeg(photo)), ("rotated.jpg", jpeg(shapes, exif.tobytes()))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp32", action="store_true",
                    help="also write the tower's rows in fp32 (math attention): the reference tf-qwen35-cuda vision bounds both towers by")
    a = ap.parse_args()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tensorfold.vision.images import _decode, DEFAULT_LIMITS
    from tensorfold.vision.qwen_cuda import QwenCudaVision, TOKENS_PER_IMAGE

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vision = QwenCudaVision(a.model, torch.device("cuda:0"))
    proc = vision.frontend.processor
    factor = vision.config["patch_size"] * vision.config["spatial_merge_size"]
    cap = min(int(getattr(proc, "max_pixels", TOKENS_PER_IMAGE * factor**2)), TOKENS_PER_IMAGE * factor**2)
    floor = min(int(getattr(proc, "min_pixels", factor**2)), cap)
    entries = []
    for name, data in images():
        stem = name.replace(".", "_")
        (out / name).write_bytes(data)
        decoded = _decode(data, "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_pixels)
        pil = decoded.to_pil()
        rgb = np.asarray(pil, dtype=np.uint8)
        (out / f"{stem}.rgb").write_bytes(rgb.tobytes())
        result = proc(images=[pil], max_pixels=cap, min_pixels=floor)
        pixels = np.asarray(result["pixel_values"], dtype=np.float32)
        t, h, w = (int(v) for v in np.asarray(result["image_grid_thw"])[0])
        (out / f"{stem}.f32").write_bytes(pixels.tobytes())
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            feats = vision.tower(torch.tensor(pixels, dtype=torch.bfloat16, device="cuda:0"),
                                 grid_thw=torch.tensor([[t, h, w]], device="cuda:0"), return_dict=True).pooler_output
        feats = feats.to(torch.bfloat16).contiguous().cpu()
        (out / f"{stem}.bf16").write_bytes(feats.view(torch.int16).numpy().tobytes())
        entry = {"name": name, "file": name, "rgb": f"{stem}.rgb", "size": [rgb.shape[1], rgb.shape[0]],
                 "grid": [h, w], "pixels": f"{stem}.f32", "features": f"{stem}.bf16", "tokens": h * w // 4}
        if a.fp32:
            tower32 = vision.tower.float()
            with torch.inference_mode(), sdpa_kernel([SDPBackend.MATH]):
                ref = tower32(torch.tensor(pixels, dtype=torch.float32, device="cuda:0"),
                              grid_thw=torch.tensor([[t, h, w]], device="cuda:0"), return_dict=True).pooler_output
            (out / f"{stem}.ref32").write_bytes(ref.float().contiguous().cpu().numpy().tobytes())
            vision.tower.to(torch.bfloat16)
            entry["reference"] = f"{stem}.ref32"
        entries.append(entry)
        print(f"{name}: {rgb.shape[1]}x{rgb.shape[0]} -> grid {h}x{w}, {h * w // 4} tokens", flush=True)
    (out / "manifest.json").write_text(json.dumps({"images": entries, "cap_pixels": cap, "min_pixels": floor},
                                                  indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
