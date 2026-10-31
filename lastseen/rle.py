"""COCO-style RLE helpers without pycocotools.

We store masks as *uncompressed* COCO RLE ({"size": [h, w], "counts": [int, ...]}) in
column-major order, which is what pycocotools/`decode_mask_to_rle` produce, so both
interoperate. Compressed (string) counts from pycocotools are also accepted by `decode`.
"""
from __future__ import annotations

import numpy as np


def encode(mask: np.ndarray) -> dict:
    """Binary HxW array -> {"size": [h, w], "counts": [..]} (column-major, starts with zeros run)."""
    mask = np.asarray(mask, dtype=np.uint8)
    h, w = mask.shape
    flat = mask.T.flatten()  # column-major
    counts: list[int] = []
    if flat.size == 0:
        return {"size": [h, w], "counts": [0]}
    # positions where value changes
    diff = np.diff(flat)
    idx = np.flatnonzero(diff) + 1
    starts = np.concatenate(([0], idx))
    ends = np.concatenate((idx, [flat.size]))
    runs = (ends - starts).tolist()
    if flat[0] == 1:
        counts.append(0)
    counts.extend(runs)
    return {"size": [h, w], "counts": counts}


def _decompress_counts(s: str) -> list[int]:
    """LEB128-like COCO compressed RLE string -> list of ints."""
    counts: list[int] = []
    i = 0
    m = 0
    n = len(s)
    while i < n:
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if m > 2:
            x += counts[m - 2]
        counts.append(x)
        m += 1
    return counts


def decode(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    if isinstance(counts, (bytes, str)):
        counts = _decompress_counts(counts.decode() if isinstance(counts, bytes) else counts)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for c in counts:
        c = int(c)
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val ^= 1
    return flat.reshape((w, h)).T  # column-major -> HxW


def area(rle: dict) -> int:
    counts = rle["counts"]
    if isinstance(counts, (bytes, str)):
        counts = _decompress_counts(counts.decode() if isinstance(counts, bytes) else counts)
    return int(sum(int(c) for c in counts[1::2]))


def iou_boxes(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two (x1, y1, x2, y2) boxes with half-open x2/y2."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua else 0.0
