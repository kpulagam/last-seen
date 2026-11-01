"""Rendering: mask overlays on frames and on tracking windows."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config, db, rle
from .video import vsync_args

TINT = (255, 61, 143)        # Muse pink: the one colour the UI, the masks and the tracked clips share


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)       # Pillow >= 10.1 ships a scalable default
    except TypeError:
        return ImageFont.load_default()


def tint_mask(img: Image.Image, mask: np.ndarray, box: tuple[int, int, int, int] | None = None,
              alpha: int = 105, outline: bool = True, label: str | None = None) -> Image.Image:
    base = img.convert("RGBA")
    if mask is not None and mask.shape[:2] != (base.height, base.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(base.size, Image.NEAREST)) // 255
    if mask is not None:
        colour = Image.new("RGBA", base.size, TINT + (0,))
        colour.putalpha(Image.fromarray((mask.astype(np.uint8) * alpha)))
        base = Image.alpha_composite(base, colour)
    d = ImageDraw.Draw(base)
    if box and outline:
        x1, y1, x2, y2 = box
        w = max(2, base.width // 480)
        d.rounded_rectangle([x1, y1, x2 - 1, y2 - 1], radius=max(4, w * 3), outline=TINT + (255,), width=w)
        if label:
            size = max(13, base.width // 70)
            f = _font(size)
            tw = d.textlength(label, font=f)
            pad = size // 2
            top = y1 - size - pad * 2 - 4
            if top < 0:                        # no room above: put the pill inside the box
                top = y1 + 4
            d.rounded_rectangle([x1, top, x1 + tw + pad * 2, top + size + pad * 2], radius=(size + pad * 2) // 2, fill=TINT + (235,))
            d.text((x1 + pad, top + pad - 1), label, font=f, fill=(255, 255, 255, 255))
    return base.convert("RGB")


def overlay_png(con, sighting_id: int, out: str | Path | None = None) -> Path:
    s = db.one(con, "SELECT * FROM sightings WHERE id=?", (sighting_id,))
    kf = db.one(con, "SELECT jpg_path FROM keyframes WHERE id=?", (s["keyframe_id"],))
    img = Image.open(config.local_path(kf["jpg_path"])).convert("RGB")
    mask = rle.decode(db.jload(s["mask_rle"])) if s["mask_rle"] else None
    out = Path(out) if out else config.FRAMES / "overlays" / f"s_{sighting_id}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    obj = db.one(con, "SELECT user_name, canonical_label FROM objects WHERE id=?", (s["object_id"],)) if s["object_id"] else None
    label = (obj or {}).get("user_name") or (obj or {}).get("canonical_label") or s["label"]
    tint_mask(img, mask, (s["x1"], s["y1"], s["x2"], s["y2"]), label=label).save(out, quality=90)
    return out


def overlay_video(con, window_id: int, out: str | Path | None = None, fps: float | None = None) -> Path | None:
    """Render the tracked object onto every frame of a window and encode to MP4."""
    w = db.one(con, "SELECT * FROM windows WHERE id=?", (window_id,))
    if not w or not w["path"] or not Path(w["path"]).exists():
        return None
    out = Path(out) if out else config.CLIPS / "windows" / f"w_{w['sighting_id']}_overlay.mp4"
    if out.exists():
        return out
    tracks = {t["frame_idx"]: t for t in db.rows(con, "SELECT * FROM tracks WHERE window_id=?", (window_id,))}
    clip = db.one(con, "SELECT fps FROM clips WHERE clip_id=?", (w["clip_id"],))
    fps = fps or float(clip["fps"] or 30)
    tmp = out.parent / f"_w{window_id}"
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", w["path"], *vsync_args("passthrough"), str(tmp / "%05d.png")], check=True)
    frames = sorted(tmp.glob("[0-9]*.png"))
    label = None
    obj = db.one(con, "SELECT canonical_label FROM objects WHERE id=?", (w["object_id"],)) if w["object_id"] else None
    if obj:
        label = obj["canonical_label"]
    for i, fp in enumerate(frames):
        t = tracks.get(i)
        img = Image.open(fp).convert("RGB")
        if t:
            mask = rle.decode(db.jload(t["mask_rle"])) if t["mask_rle"] else None
            img = tint_mask(img, mask, (t["x1"], t["y1"], t["x2"], t["y2"]), label=label)
        img.save(fp)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", str(out)],
                   check=True)
    for fp in frames:
        fp.unlink(missing_ok=True)
    tmp.rmdir()
    return out


def cutout_png(con, sighting_id: int, out: str | Path | None = None, pad: float = 0.08, size: int = 512) -> Path | None:
    """The object alone: its keyframe cropped to the box, SAM mask as alpha (edges feathered), on
    transparency — what a product shows instead of a bounding box. None when the sighting has no mask."""
    from PIL import ImageFilter
    s = db.one(con, "SELECT * FROM sightings WHERE id=?", (sighting_id,))
    if not s or not s["mask_rle"]:
        return None
    kf = db.one(con, "SELECT jpg_path FROM keyframes WHERE id=?", (s["keyframe_id"],))
    img = Image.open(config.local_path(kf["jpg_path"])).convert("RGB")
    mask = rle.decode(db.jload(s["mask_rle"]))
    if mask.shape[:2] != (img.height, img.width):
        mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) // 255
    x1, y1, x2, y2 = int(s["x1"]), int(s["y1"]), int(s["x2"]), int(s["y2"])
    px, py = int((x2 - x1) * pad) + 4, int((y2 - y1) * pad) + 4
    bx1, by1, bx2, by2 = max(0, x1 - px), max(0, y1 - py), min(img.width, x2 + px), min(img.height, y2 + py)
    crop = img.crop((bx1, by1, bx2, by2)).convert("RGBA")
    a = Image.fromarray((mask[by1:by2, bx1:bx2] > 0).astype(np.uint8) * 255)
    a = a.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.8))      # close pinholes, feather the edge
    crop.putalpha(a)
    # square canvas so every cut-out sits the same way in the UI
    side = max(crop.width, crop.height)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    if side > size:
        canvas = canvas.resize((size, size), Image.LANCZOS)
    out = Path(out) if out else config.FRAMES / "cutouts" / f"s_{sighting_id}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, "PNG", optimize=True)
    return out
