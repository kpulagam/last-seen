"""Benchmark DINOv3 / DINOv2 crop embeddings on this machine (MPS if available).

    python scripts/bench_embed.py                      # tries dinov3 ViT-S/16, falls back to dinov2-small
    python scripts/bench_embed.py --model facebook/dinov2-small

DINOv3 weights are gated on Hugging Face: accept the license on the model page and run
`huggingface-cli login` once. If that fails this script falls back to DINOv2-small (Apache-2.0).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen.reid import Embedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    from PIL import Image
    import numpy as np

    emb = Embedder(model_id=args.model)
    print("model:", emb.model_id, "device:", emb.device, "dim:", emb.dim)
    imgs = [Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8")) for _ in range(args.n)]
    emb.embed_images(imgs[:4])  # warm-up
    t0 = time.time()
    vecs = emb.embed_images(imgs)
    dt = time.time() - t0
    print(f"{args.n} crops in {dt:.2f}s -> {dt / args.n * 1000:.1f} ms/crop; vec shape {vecs.shape}")


if __name__ == "__main__":
    main()
