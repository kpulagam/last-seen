"""A MetaClient stand-in for offline tests. Spark grounding "sees" coloured rectangles on a
synthetic frame by scanning pixels; SAM returns box records without mask tokens (full-box masks)."""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from lastseen.meta_client import CallLog

COLOURS = {"red mug": (220, 30, 30), "blue charger": (30, 60, 220), "yellow book": (240, 210, 20),
           "grey charger": (70, 70, 75)}


def _decode(data_url: str) -> Image.Image:
    b = base64.b64decode(data_url.split(",", 1)[1])
    return Image.open(io.BytesIO(b)).convert("RGB")


def find_rects(img: Image.Image) -> list[tuple[str, tuple[int, int, int, int]]]:
    a = np.asarray(img).astype(int)
    out = []
    for name, col in COLOURS.items():
        d = np.abs(a - np.array(col)).sum(-1)
        ys, xs = np.where(d < 60)
        if len(xs) > 200:
            out.append((name, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    return out


@dataclass
class MockClient:
    run_dir: object = None
    calls: list = field(default_factory=list)
    room: str = "kitchen"

    def _log(self, kind, usage=None):
        self.calls.append(CallLog(kind=kind, request={}, status=200, elapsed_s=0.01, usage=usage))

    def spark_json(self, prompt, *, images=None, schema, schema_name="", system=None, max_tokens=4000, kind="spark"):
        self._log(kind, {"prompt_tokens": 1500, "completion_tokens": 300})
        if schema_name == "keyframe_objects":
            img = _decode(images[0])
            W, H = img.size
            objs = []
            for name, (x1, y1, x2, y2) in find_rects(img):
                objs.append({"label": name, "description": f"a {name}", "in_hand": False, "surface": "counter",
                             "confidence": 0.9,
                             "bbox": {"x_min": int(x1 * 1000 / W), "y_min": int(y1 * 1000 / H),
                                      "x_max": int(x2 * 1000 / W), "y_max": int(y2 * 1000 / H)}})
            return {"room": self.room, "surfaces": ["counter", "floor"], "objects": objs}
        if schema_name == "object_attributes":
            m = re.search(r"called it: '([^']+)'", prompt) or re.search(r"called it '([^']+)'", prompt)
            label = m.group(1) if m else "object"
            return {"canonical_label": label, "category": "charger" if "charger" in label else ("mug" if "mug" in label else "book"),
                    "dominant_color": label.split()[0], "secondary_colors": [], "material": "plastic", "brand_text": "",
                    "marks": [], "size_class": "small", "description": label}
        if schema_name == "same_object":
            return {"same": False, "confidence": 0.5, "reason": "indistinguishable in mock"}
        if schema_name == "last_seen_intent":
            raise RuntimeError("mock has no intent model; use query.keyword_intent")
        raise ValueError(schema_name)

    def sam_responses(self, prompt_text, *, image_data_url=None, video_file_id=None, include_confidence=True,
                      stream=True, mask_encoding="one_bit", propagation_direction=None, kind="sam", **kw):
        self._log(kind, {"video_frames_processed": 50} if video_file_id else None)
        # echo the prompted boxes back as records (slightly tightened), no masks
        W = H = None
        recs = []
        for m in re.finditer(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)\|>", prompt_text):
            oid, x1, y1, x2, y2, w, h = map(int, m.groups())
            W, H = w, h
            recs.append(f"{oid}<|box;x1={x1 + 1};y1={y1 + 1};x2={x2 - 1};y2={y2 - 1};w={w};h={h};c=0.95|>")
        if video_file_id:
            return "\n".join(f"<{f}f>" + ",".join(recs) for f in range(0, 50))
        return "<0f>" + ",".join(recs)

    def upload_file(self, path, purpose="user_data"):
        self._log("files.create")
        return "file-mock"
