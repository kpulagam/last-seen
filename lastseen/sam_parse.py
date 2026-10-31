"""Parse SAM 3.1 special-token output into frame/object records.

Two layers:
  * `parse_lines()` — our own tolerant parser of the documented grammar. It extracts
    frame index, object id, pixel box, frame size, confidence, and the *raw* mask token.
    It needs no third-party code and is what the pipeline keys on.
  * `decode_mask()` — decodes a raw mask token to a binary raster using the official
    `meta-sam-parser` package (pip install meta-sam-parser). The lossless encoding is an
    arithmetic coder, so there is no sensible way to re-implement it here.

Observed/documented geometry: the mask raster is box-local — `data=H,W,...` has
H = y2-y1+1 and W = x2-x1+1 (inclusive wire coordinates), so the raster is pasted at
(x1, y1) in the full frame. `full_frame_mask()` does that.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

FRAME_RE = re.compile(r"<(\d+)f>")
# tokens: <|box;...|> and <|mask;...|>; the mask payload may contain almost any printable char
TOKEN_RE = re.compile(r"<\|(box|mask);([^|]*?)\|>", re.DOTALL)
# a record = object id followed by its tokens; records separated by commas outside tokens
RECORD_START_RE = re.compile(r"(?:^|,)\s*(\d+)\s*(?=<\|)")


@dataclass
class SamRecord:
    frame: int
    object_id: int
    box: tuple[int, int, int, int]      # x1, y1, x2, y2  (half-open x2/y2, i.e. wire x2+1)
    frame_w: int
    frame_h: int
    confidence: float | None = None
    mask_token: str = ""                # full "<|mask;...|>" token for decoding
    mask_hw: tuple[int, int] | None = None

    @property
    def area_box(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass
class SamFrame:
    frame: int
    records: list[SamRecord] = field(default_factory=list)


def _parse_kv(s: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in s.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v
    return out


def parse_lines(text: str) -> list[SamFrame]:
    """Parse complete SAM output text (image or video) into SamFrame list in output order."""
    frames: list[SamFrame] = []
    if not text or not text.strip():
        return frames
    # Split on frame markers; keep the index.
    positions = [(m.start(), m.end(), int(m.group(1))) for m in FRAME_RE.finditer(text)]
    for i, (start, end, fidx) in enumerate(positions):
        seg_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        seg = text[end:seg_end].strip().strip("\n")
        fr = SamFrame(frame=fidx)
        # Walk tokens; an object id precedes its box token.
        pos = 0
        current: SamRecord | None = None
        for tm in TOKEN_RE.finditer(seg):
            kind, body = tm.group(1), tm.group(2)
            # find the object id in the gap before this token (only when a box starts a record)
            gap = seg[pos:tm.start()]
            pos = tm.end()
            if kind == "box":
                m = re.search(r"(\d+)\s*$", gap.strip().lstrip(","))
                oid = int(m.group(1)) if m else (len(fr.records))
                kv = _parse_kv(body)
                try:
                    x1, y1, x2, y2 = int(kv["x1"]), int(kv["y1"]), int(kv["x2"]), int(kv["y2"])
                    w, h = int(kv["w"]), int(kv["h"])
                except (KeyError, ValueError):
                    current = None
                    continue
                conf = float(kv["c"]) if "c" in kv else None
                current = SamRecord(frame=fidx, object_id=oid, box=(x1, y1, x2 + 1, y2 + 1),
                                    frame_w=w, frame_h=h, confidence=conf)
                fr.records.append(current)
            elif kind == "mask" and current is not None:
                current.mask_token = tm.group(0)
                # data=H,W,<enc>payload  -> parse H,W
                dm = re.search(r"data=(\d+),(\d+),", body)
                if dm:
                    current.mask_hw = (int(dm.group(1)), int(dm.group(2)))
                if current.confidence is None:
                    cm = re.search(r"(?:^|;)c=([0-9.]+)", body)
                    if cm:
                        current.confidence = float(cm.group(1))
                current = None
        frames.append(fr)
    return frames


# ------------------------------------------------------------------ decoding
_decoder = None


def _get_decoder():
    """Lazily import meta_sam_parser and return a function(mask_token)->np.ndarray(H,W) uint8."""
    global _decoder
    if _decoder is not None:
        return _decoder
    try:
        import meta_sam_parser as msp  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install meta-sam-parser  (needed to decode SAM masks)") from e

    def decode(mask_token: str, hw: tuple[int, int] | None) -> np.ndarray:
        # The package parses full lines; feed it a minimal synthetic line containing the token.
        fmt = msp.image_segmentation_format()
        parser = fmt.create_parser()
        line = f"<0f>0<|box;x1=0;y1=0;x2={(hw[1] - 1) if hw else 0};y2={(hw[0] - 1) if hw else 0};w=10000;h=10000|>{mask_token}"
        parser.push(line)
        fin = parser.finish(msp.CompletedOutcome())
        result = getattr(fin, "result", fin)
        records = getattr(result, "records", [])
        mask_rec = None
        for r in records:
            if r.__class__.__name__.lower().startswith("segmentationmask") or hasattr(r, "mask"):
                mask_rec = r
        if mask_rec is None:
            raise ValueError("meta_sam_parser produced no mask record for token")
        raster = msp.decode_mask_to_raster(mask_rec.mask)
        H, W = _mask_dims(mask_rec, hw)
        arr = np.frombuffer(bytes(raster), dtype=np.uint8).reshape(H, W).copy()
        return arr

    _decoder = decode
    return _decoder


def _mask_dims(mask_rec, hw: tuple[int, int] | None) -> tuple[int, int]:
    m = getattr(mask_rec, "mask", None)
    for attr in ("height", "rows"):
        if m is not None and hasattr(m, attr):
            H = int(getattr(m, attr))
            W = int(getattr(m, "width", getattr(m, "cols", 0)))
            if H and W:
                return H, W
    b = getattr(mask_rec, "bounds", None)
    if b is not None and all(hasattr(b, a) for a in ("left", "top", "right", "bottom")):
        return int(b.bottom - b.top), int(b.right - b.left)
    if hw:
        return hw
    raise ValueError("cannot determine mask dimensions")


def decode_mask(rec: SamRecord) -> np.ndarray:
    """Box-local binary mask (H x W) for a record."""
    if not rec.mask_token:
        x1, y1, x2, y2 = rec.box
        return np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
    return _get_decoder()(rec.mask_token, rec.mask_hw)


def full_frame_mask(rec: SamRecord, local: np.ndarray | None = None) -> np.ndarray:
    """Paste the box-local mask into a frame-sized (frame_h x frame_w) raster."""
    if local is None:
        local = decode_mask(rec)
    H, W = rec.frame_h, rec.frame_w
    out = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = rec.box
    h, w = local.shape
    # If the raster isn't exactly the box size, resize nearest-neighbour to fit the box.
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    if (h, w) != (bh, bw):
        try:
            from PIL import Image
            local = np.array(Image.fromarray(local * 255).resize((bw, bh), Image.NEAREST)) // 255
            h, w = local.shape
        except Exception:
            pass
    y_end, x_end = min(H, y1 + h), min(W, x1 + w)
    out[y1:y_end, x1:x_end] = local[: y_end - y1, : x_end - x1]
    return out


# ----------------------------------------------------------- prompt building
def build_box_prompt(frame_w: int, frame_h: int, objects: list[dict]) -> str:
    """Build a box/point prompt in SAM 3.1 grammar.

    objects: [{"id": 1, "prompts": [{"frame": 30, "box": (x1, y1, x2, y2)}, {"frame": 120, "points": [(x, y, +1), (x, y, -1)]}]}]
    Boxes are half-open (x2, y2 exclusive) and converted to inclusive wire coordinates.
    Output groups prompts by ascending frame, objects in input order within each frame.
    """
    by_frame: dict[int, list[str]] = {}
    for obj in objects:
        oid = int(obj["id"])
        for p in obj["prompts"]:
            f = int(p.get("frame", 0))
            toks: list[str] = []
            if "box" in p and p["box"] is not None:
                x1, y1, x2, y2 = p["box"]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(frame_w - 1, int(x2) - 1), min(frame_h - 1, int(y2) - 1)
                if x2i <= x1 or y2i <= y1:
                    continue
                toks.append(f"<|box;x1={x1};y1={y1};x2={x2i};y2={y2i};w={frame_w};h={frame_h}|>")
            for pt in p.get("points", []) or []:
                x, y = int(pt[0]), int(pt[1])
                x, y = max(0, min(frame_w - 1, x)), max(0, min(frame_h - 1, y))
                label = pt[2] if len(pt) > 2 else 1
                tok = f"<|point;x={x};y={y};w={frame_w};h={frame_h}|>"
                toks.append(("-" if label < 0 else "") + tok)
            if toks:
                by_frame.setdefault(f, []).append(f"{oid}" + "".join(toks))
    return "".join(f"<{f}f>" + ",".join(by_frame[f]) for f in sorted(by_frame))
