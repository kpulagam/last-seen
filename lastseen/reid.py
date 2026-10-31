"""Object re-identification: DINOv3/DINOv2 embeddings + a cascade that ends in Spark or the user.

Per sighting, three L2-normalised vectors:
  e_obj   CLS token of the mask-applied crop (background blanked to mid-grey)
  e_patch mean of patch tokens that fall inside the mask (viewpoint-robust)
  e_ring  CLS token of a 2x-expanded box with the object itself blanked (background ring)

Similarity = 0.6*cos(obj) + 0.4*cos(patch), max over an object's exemplars; the ring adds a small
bonus when it agrees. Thresholds were set from the first two real clips (Oct 27, 2026): the same
object across sessions scores 0.50-0.94 on its best exemplar, different categories 0.2-0.4,
and same-category different instances up to ~0.5.

Two structural rules do more than any threshold:
  * co-occurrence: two sightings on the same keyframe are different objects, always;
  * single-home prior: within one home, two sightings of the same kind and colour with no
    contradicting marks are the same object unless two instances were ever seen together.
"""
from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, db, rle
from .meta_client import MetaClient, bytes_to_data_url
from .state import coverage_of, session_coverage

DEFAULT_MODELS = [
    os.environ.get("EMBED_MODEL", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    "facebook/dinov2-small",
]

T_MATCH = float(os.environ.get("REID_T_MATCH", "0.70"))     # auto-match on appearance alone
T_CATEGORY = float(os.environ.get("REID_T_CATEGORY", "0.50"))  # auto-match if Spark's category agrees
T_LABEL = float(os.environ.get("REID_T_LABEL", "0.42"))     # auto-match if the head noun agrees
T_VERIFY = float(os.environ.get("REID_T_VERIFY", "0.35"))   # below this: new object, no Spark call
T_VERIFY_ANY = float(os.environ.get("REID_T_VERIFY_ANY", "0.55"))  # verify even if label/category disagree
T_DEDUPE_ANY = float(os.environ.get("REID_T_DEDUPE_ANY", "0.78"))  # object merge with no label/category agreement
T_CLUSTER = float(os.environ.get("REID_T_CLUSTER", "0.65"))  # within-clip merge on appearance alone
T_CLUSTER_LABEL = float(os.environ.get("REID_T_CLUSTER_LABEL", "0.45"))  # within-clip merge if head nouns agree
MARGIN = 0.04
W_OBJ, W_PATCH = 0.6, 0.4
RING_BONUS = 0.03
MAX_EXEMPLARS = 6
MIN_CROP_SIDE = 36          # px; smaller crops embed as noise (measured: cables/stickers at <40 px)
BLUR_REL = 0.25             # crop quality below this fraction of the clip median = low quality
SIZE_REF = 100              # px; quality = sharpness * clip(short side / SIZE_REF, 0.3, 2.0)
BROAD_CATEGORIES = {"other", "unknown", "clothing", "tool", "paper", "food", "toy", "cosmetics", "bag", "utensil"}
MIN_MASK_COVER = 0.08       # mask area / box area below this = SAM mask failed (slivers); treat the crop as blurry
MASK_COVER_PLAIN = 0.35     # below this the mask probably caught only part of the object (the keys, not their fob):
                            # Spark gets the plain crop instead of the masked one
CHROMA_VETO = float(os.environ.get("REID_CHROMA_VETO", "22"))  # CIELAB a/b distance that marks a colour clash
CHROMA_PENALTY = 0.25       # subtracted from similarity on a colour clash (DINO is colour-blind-ish)


# ------------------------------------------------------------------ embedder
class Embedder:
    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        candidates = [model_id] if model_id else DEFAULT_MODELS
        last_err: Exception | None = None
        for mid in candidates:
            try:
                self.processor = AutoImageProcessor.from_pretrained(mid)
                self.model = AutoModel.from_pretrained(mid)
                self.model_id = mid
                break
            except Exception as e:  # noqa: BLE001 — gated weights / old transformers
                last_err = e
                continue
        else:
            raise RuntimeError(f"could not load any embedding model {candidates}: {last_err}")
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.torch = torch
        self.dim = int(self.model.config.hidden_size)
        self.n_special = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        self.patch = int(getattr(self.model.config, "patch_size", 16))

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        return v / n

    def embed_images(self, images: list[Image.Image], masks=None, batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(images), batch):
            chunk = [im.convert("RGB") for im in images[i:i + batch]]
            inputs = self.processor(images=chunk, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                o = self.model(**inputs)
            cls = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state[:, 0]
            out.append(cls.float().cpu().numpy())
        return self._l2(np.concatenate(out, 0)) if out else np.zeros((0, self.dim), np.float32)

    def embed_obj_patch(self, image: Image.Image, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        inputs = self.processor(images=[image.convert("RGB")], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            o = self.model(**inputs)
        hs = o.last_hidden_state[0].float().cpu().numpy()
        cls = o.pooler_output[0].float().cpu().numpy() if getattr(o, "pooler_output", None) is not None else hs[0]
        patches = hs[self.n_special:]
        side = int(round(np.sqrt(patches.shape[0])))
        if side * side != patches.shape[0] or mask is None or mask.size == 0:
            e_patch = patches.mean(0)
        else:
            m = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).resize((side, side), Image.BILINEAR)
            m = (np.asarray(m) > 64).reshape(-1)
            e_patch = patches[m].mean(0) if m.any() else patches.mean(0)
        return self._l2(cls[None])[0], self._l2(e_patch[None])[0]


# ------------------------------------------------------------- crop builders
def crop_views(frame: Image.Image, box: tuple[int, int, int, int], mask_full: np.ndarray | None,
               pad: float = 0.10, ring_scale: float = 2.0) -> dict:
    W, H = frame.size
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad), int(bh * pad)
    cx1, cy1, cx2, cy2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
    crop = frame.crop((cx1, cy1, cx2, cy2))
    if mask_full is None:
        mask_full = np.zeros((H, W), np.uint8)
        mask_full[y1:y2, x1:x2] = 1
    mcrop = mask_full[cy1:cy2, cx1:cx2]
    arr = np.asarray(crop).copy()
    grey = np.full_like(arr, 128)
    masked = np.where(mcrop[..., None] > 0, arr, grey)
    rw, rh = int(bw * (ring_scale - 1) / 2), int(bh * (ring_scale - 1) / 2)
    rx1, ry1, rx2, ry2 = max(0, x1 - rw), max(0, y1 - rh), min(W, x2 + rw), min(H, y2 + rh)
    ring = np.asarray(frame.crop((rx1, ry1, rx2, ry2))).copy()
    sub = mask_full[ry1:ry2, rx1:rx2]
    ring[sub > 0] = 128
    cover = float((mask_full[y1:y2, x1:x2] > 0).mean()) if (y2 > y1 and x2 > x1) else 0.0
    return {"crop": crop, "crop_masked": Image.fromarray(masked), "ring": Image.fromarray(ring), "mask_crop": mcrop,
            "mask_cover": cover}


def crop_sharpness(img: Image.Image) -> float:
    """Contrast-normalised sharpness: var(Laplacian) / var(gray). Motion blur spreads edges, so the
    Laplacian shrinks relative to the crop's own contrast; a low-contrast but sharp object (white
    napkin on a white counter) is not penalised the way raw Laplacian variance would."""
    a = np.asarray(img.convert("L"), dtype=np.float32)
    if a.size < 64:
        return 0.0
    gv = float(a.var())
    if gv < 4.0:          # textureless crop: nothing to measure
        return 0.0
    try:
        import cv2
        lap = cv2.Laplacian(a, cv2.CV_32F)
    except Exception:  # noqa: BLE001
        lap = (np.roll(a, 1, 0) + np.roll(a, -1, 0) + np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)
    return float(lap.var() / gv)


def crop_quality(sharp: float, box: tuple[int, int, int, int]) -> float:
    """Re-ID usefulness of a crop: sharpness weighted by size. Large smooth objects (a black massage
    gun) have little Laplacian energy per pixel but plenty of pixels for the embedder; tiny crops are
    noise however sharp. Crops under MIN_CROP_SIDE get 0."""
    side = min(box[2] - box[0], box[3] - box[1])
    if side < MIN_CROP_SIDE:
        return 0.0
    return float(sharp) * float(np.clip(side / SIZE_REF, 0.3, 2.0))


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) sRGB uint8 -> (N,3) CIELAB (D65)."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], 1)


def colour_signature(crop_masked: Image.Image, mask: np.ndarray | None = None) -> np.ndarray | None:
    """Mean CIELAB of the object's own pixels (the mask, or every pixel that is not the mid-grey fill).
    A three-number colour memory that DINO lacks: a peach soap bottle and a steel water bottle embed
    alike (cos 0.88) but sit 30+ apart in a/b."""
    arr = np.asarray(crop_masked.convert("RGB"))
    if mask is not None and mask.shape[:2] == arr.shape[:2]:
        sel = np.asarray(mask) > 0
    else:
        sel = ~np.all(arr == 128, axis=-1)
    px = arr[sel]
    if px.shape[0] < 16:
        return None
    if px.shape[0] > 20000:
        px = px[np.random.default_rng(0).choice(px.shape[0], 20000, replace=False)]
    return _srgb_to_lab(px).mean(0).astype(np.float32)


def chroma_distance(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None or len(a) < 3 or len(b) < 3:
        return None
    return float(np.hypot(a[1] - b[1], a[2] - b[2]))


def apply_relative_blur(props) -> None:
    """Zero the quality of crops far below the clip's own median quality (motion-blurred frames).
    Only applied when the clip has real texture: synthetic flat test crops have ~0 variance."""
    qs = [p.quality for p in props if p.quality > 0]
    if not qs:
        return
    med = float(np.median(qs))
    if med <= 0.02:
        return
    for p in props:
        if 0 < p.quality < BLUR_REL * med:
            p.quality = 0.0


# --------------------------------------------------------------- similarity
def similarity(q: dict, ex: dict) -> float:
    s = 0.0
    a, b = q.get("e_obj"), ex.get("e_obj")
    if a is not None and b is not None:
        s += W_OBJ * float(np.dot(a, b))
    a, b = q.get("e_patch"), ex.get("e_patch")
    if a is not None and b is not None:
        s += W_PATCH * float(np.dot(a, b))
    a, b = q.get("e_ring"), ex.get("e_ring")
    if a is not None and b is not None and float(np.dot(a, b)) > 0.5:
        s += RING_BONUS
    d = chroma_distance(q.get("lab"), ex.get("lab"))
    if d is not None and d > CHROMA_VETO:
        s = max(0.0, s - CHROMA_PENALTY)
    return s


_ARTICLES = re.compile(r"^(a|an|the|my|some)\s+", re.I)


def head_noun(label: str | None) -> str:
    s = _ARTICLES.sub("", (label or "").strip().lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s).strip()
    toks = s.split()
    if not toks:
        return ""
    h = toks[-1]
    return h[:-1] if h.endswith("s") and len(h) > 3 and not h.endswith("ss") else h


HARD_ATTRS = ("dominant_color", "brand_text")
_COLOUR_FAMILY = {"black": "dark", "dark": "dark", "charcoal": "dark", "navy": "dark", "dark blue": "dark",
                  "gray": "grey", "grey": "grey", "silver": "grey", "steel": "grey",
                  "white": "light", "cream": "light", "off-white": "light", "beige": "light", "ivory": "light"}


def _norm_colour(c: str) -> str:
    c = (c or "").strip().lower()
    return _COLOUR_FAMILY.get(c, c)


_NO_COLOUR = ("unknown", "none", "", "multicolor", "multicolour", "multi", "various", "mixed")


COLOUR_WORDS = {"black", "white", "grey", "gray", "silver", "red", "orange", "yellow", "green", "blue", "navy",
                "purple", "violet", "pink", "peach", "brown", "tan", "beige", "cream", "gold", "teal", "turquoise",
                "maroon", "olive", "ivory", "charcoal", "steel", "dark", "light", "transparent", "clear"}


def label_colours(label: str | None) -> set[str]:
    return {_norm_colour(w) for w in re.findall(r"[a-z]+", (label or "").lower()) if w in COLOUR_WORDS}


def colour_set(a: dict | None) -> set[str]:
    """Colour families an attribute record admits: dominant + secondary colours, each word of a
    compound ('light blue-gray' -> light, blue, grey) so partial agreement counts. Falls back to
    colour words in the canonical label when the record has no colours."""
    if not a:
        return set()
    raw = [a.get("dominant_color") or ""] + list(a.get("secondary_colors") or [])
    if not any((c or "").strip() for c in raw):
        return label_colours(a.get("canonical_label"))
    out: set[str] = set()
    for c in raw:
        c = (c or "").strip().lower()
        if not c or c in _NO_COLOUR:
            continue
        out.add(_norm_colour(c))
        for w in re.split(r"[\s/,-]+", c):
            if w and w not in _NO_COLOUR and w not in ("and", "with"):
                out.add(_norm_colour(w))
    return out


COMMON_NOUNS = {"cup", "mug", "glass", "bottle", "jar", "container", "cable", "cord", "charger", "box", "bag",
                "plate", "bowl", "towel", "cloth", "sock", "pen", "book", "paper", "toy", "cap", "hat", "shirt",
                "jacket", "sticker", "can", "tub", "roll"}   # several per home: the same name is weak evidence


def dominant_family(a: dict | None) -> str:
    """Colour family of the dominant colour: 'dark gray' -> dark, 'silver' -> grey, 'white' -> light,
    'blue' -> blue; '' when unknown."""
    if not a:
        return ""
    raw = (a.get("dominant_color") or "").strip().lower()
    if not raw or raw in _NO_COLOUR:
        lc = label_colours(a.get("canonical_label"))
        return sorted(lc)[0] if len(lc) == 1 else ""
    for w in [raw] + re.split(r"[\s/,-]+", raw):
        if w in _COLOUR_FAMILY:
            return _COLOUR_FAMILY[w]
    return raw.split()[-1] if raw else ""


def attribute_contradictions(a: dict | None, b: dict | None) -> list[str]:
    """Hard disagreements between two attribute records (both from Spark). Colours contradict when
    the dominant colour families differ (a white cup is not a black cup), or when the two records
    share no colour family at all. A strong same-name visual match may still override this."""
    if not a or not b:
        return []
    out = []
    da, db_ = dominant_family(a), dominant_family(b)
    if da and db_ and da != db_:
        out.append(f"dominant colour: {da} vs {db_}")
    sa, sb = colour_set(a), colour_set(b)
    if sa and sb and not (sa & sb) and not out:
        out.append(f"colour: {'/'.join(sorted(sa))} vs {'/'.join(sorted(sb))}")
    ba, bb = (a.get("brand_text") or "").strip().lower(), (b.get("brand_text") or "").strip().lower()
    if ba and bb and ba != bb and ba not in bb and bb not in ba:
        out.append(f"brand_text: {ba} vs {bb}")
    return out


def colour_compatible(a: dict | None, b: dict | None) -> bool:
    """True when one record's dominant colour is among the colours the other record admits (a black
    fob with silver keys is 'silver keys' from one side and 'black car keys, silver, white' from the
    other): a colour clash between such records is a naming difference, not evidence of two objects."""
    if not a or not b:
        return True
    da, db_ = dominant_family(a), dominant_family(b)
    sa, sb = colour_set(a), colour_set(b)
    if not da or not db_ or da == db_:
        return True
    return da in sb or db_ in sa


def hard_contradictions(contr: list[str], a: dict | None, b: dict | None) -> list[str]:
    """Contradictions that survive the colour-naming allowance: brand text always; colour and
    lightness only when the two records' colours are incompatible."""
    if colour_compatible(a, b):
        return [c for c in contr if not c.startswith(("dominant colour", "colour", "lightness"))]
    return list(contr)


@dataclass
class Candidate:
    object_id: int
    sim: float
    label: str
    category: str | None
    exemplar_sighting_id: int
    contradictions: list[str] = field(default_factory=list)
    head_match: bool = False
    category_match: bool = False
    ex_quality: float = 0.0     # quality of the exemplar that produced `sim`
    cross_room: bool = False    # the object's last known room differs from the query's room: a match = a move
    old_place_unscanned: bool = False   # ... and this walk-through never looked at where it was last seen


def load_object_exemplars(con, category: str | None = None) -> dict[int, list[dict]]:
    """{object_id: [exemplar dicts]} best-quality first, max MAX_EXEMPLARS."""
    sql = ("SELECT s.id AS sighting_id, s.object_id, s.place_id, s.keyframe_id, s.label, s.quality, "
           "o.category, o.canonical_label, e.e_obj, e.e_patch, e.e_ring, e.lab "
           "FROM sightings s JOIN objects o ON o.id=s.object_id JOIN embeddings e ON e.sighting_id=s.id "
           "WHERE s.object_id IS NOT NULL ")
    params: list = []
    if category:
        sql += "AND (o.category=? OR o.category IS NULL OR o.category IN ('unknown','other')) "
        params.append(category)
    sql += "ORDER BY COALESCE(s.quality, 0) DESC, s.ts DESC"
    out: dict[int, list[dict]] = {}
    for r in db.rows(con, sql, params):
        lst = out.setdefault(int(r["object_id"]), [])
        if len(lst) >= MAX_EXEMPLARS:
            continue
        # rows arrive best-quality first: once an object has a sharp exemplar, skip its blurry ones
        if lst and (lst[0].get("quality") or 0) > 0 and (r["quality"] or 0) <= 0:
            continue
        lst.append({"sighting_id": r["sighting_id"], "place_id": r["place_id"], "keyframe_id": r["keyframe_id"],
                    "label": r["canonical_label"], "category": r["category"], "quality": r["quality"] or 0,
                    "e_obj": db.blob_to_vec(r["e_obj"]), "lab": db.blob_to_vec(r["lab"]) if "lab" in r.keys() else None,
                    "e_patch": db.blob_to_vec(r["e_patch"]), "e_ring": db.blob_to_vec(r["e_ring"])})
    return out


def object_keyframes(con, object_id: int) -> set[int]:
    return {int(r["keyframe_id"]) for r in db.rows(con, "SELECT keyframe_id FROM sightings WHERE object_id=?", (object_id,))}


STAY_BONUS = 0.06           # things mostly stay where they were: same (room, surface) as the object's last place
ROOM_BONUS = 0.03           # ... or at least the same room


LIGHTNESS_VETO = 35.0       # CIELAB L difference between the two crops that no lighting change explains (silver vs black)


def rank_candidates(con, q_vecs: dict, q_attrs: dict | None, q_label: str | None, category: str | None,
                    exclude_keyframes: set[int] | None = None, k: int = 3,
                    exclude_objects: set[int] | None = None, q_place: tuple[str, str] | None = None,
                    coverage: dict | None = None) -> list[Candidate]:
    """Objects ranked by best-exemplar similarity. Objects that already have a sighting on one of
    `exclude_keyframes` are skipped (co-occurrence: they are provably a different object), as are
    `exclude_objects` (seen elsewhere in the same clip)."""
    exemplars = load_object_exemplars(con, category)
    q_head = head_noun(q_label)
    cands: list[Candidate] = []
    for oid, exs in exemplars.items():
        if exclude_objects and oid in exclude_objects:
            continue
        if exclude_keyframes and (object_keyframes(con, oid) & exclude_keyframes):
            continue
        best, best_ex = -1.0, None
        for ex in exs:
            s = similarity(q_vecs, ex)
            if s > best:
                best, best_ex = s, ex
        if best_ex is None:
            continue
        obj = db.one(con, "SELECT attributes_json, category, canonical_label, last_place_id FROM objects WHERE id=?", (oid,))
        o_attrs = db.jload(obj["attributes_json"] if obj else None, {}) or {}
        contr = attribute_contradictions(q_attrs, o_attrs)
        cross_room, unscanned = False, False
        if q_place and obj and obj["last_place_id"]:
            lp = db.one(con, "SELECT room, surface FROM places WHERE id=?", (obj["last_place_id"],))
            if lp and _norm_room(lp["room"]) == _norm_room(q_place[0]):
                best += STAY_BONUS if (lp["surface"] or "").lower() == (q_place[1] or "").lower() else ROOM_BONUS
            elif lp and _norm_room(lp["room"]) and _norm_room(q_place[0]):
                cross_room = True
                if coverage is not None:
                    from .state import coverage_of
                    unscanned = coverage_of(coverage, lp["room"], lp["surface"]) != "scanned"
        # a brightness gap no lighting explains (a silver bottle vs a black one) is a contradiction too
        ql, el = q_vecs.get("lab"), best_ex.get("lab")
        if ql is not None and el is not None and len(ql) >= 3 and len(el) >= 3 and abs(float(ql[0]) - float(el[0])) > LIGHTNESS_VETO:
            contr = list(contr) + [f"lightness: L{float(ql[0]):.0f} vs L{float(el[0]):.0f}"]
        o_head = head_noun(obj["canonical_label"] if obj else best_ex["label"])
        o_cat = (obj or {}).get("category") or best_ex["category"] or "other"
        cands.append(Candidate(object_id=oid, sim=round(best, 4), label=(obj or {}).get("canonical_label") or best_ex["label"] or "",
                               category=o_cat, exemplar_sighting_id=best_ex["sighting_id"],
                               contradictions=contr, head_match=bool(q_head and q_head == o_head),
                               category_match=bool(category and category not in BROAD_CATEGORIES and category == o_cat),
                               ex_quality=float(best_ex.get("quality") or 0.0), cross_room=cross_room,
                               old_place_unscanned=unscanned))
    cands.sort(key=lambda c: (c.sim + (0.05 if c.head_match else 0.0)), reverse=True)
    return cands[:k]


# ------------------------------------------------------------------ decision
@dataclass
class Decision:
    action: str                 # 'match' | 'new' | 'verify' | 'ask'
    object_id: int | None = None
    score: float = 0.0
    method: str = ""
    candidates: list[Candidate] = field(default_factory=list)
    spark: dict | None = None
    verified_against: int | None = None     # object Spark compared the sighting with, when it was asked


def _norm_room(r) -> str:
    r = (r or "").strip().lower()
    return "" if r in ("", "unknown", "none") else r


def object_rooms_by_session(con, object_id: int) -> dict[int, set[str]]:
    """{session_id: rooms with >= 2 sightings} — the rooms an object was solidly seen in per walk-through."""
    out: dict[int, set[str]] = {}
    for r in db.rows(con, "SELECT session_id, room, COUNT(*) AS n FROM sightings WHERE object_id=? GROUP BY session_id, room", (object_id,)):
        room = _norm_room(r["room"])
        if room and r["n"] >= 2:
            out.setdefault(int(r["session_id"]), set()).add(room)
    return out


def _distinct_instances(con, a: int, b: int) -> bool:
    """True if objects a and b are provably two things: seen on the same keyframe, or solidly seen in
    different rooms during the same walk-through (a bottle on the kitchen counter and a bottle on the
    bedroom nightstand, minutes apart, are two bottles however alike they look)."""
    if object_keyframes(con, a) & object_keyframes(con, b):
        return True
    ra, rb = object_rooms_by_session(con, a), object_rooms_by_session(con, b)
    for ses in set(ra) & set(rb):
        if ra[ses] and rb[ses] and not (ra[ses] & rb[ses]):
            return True
    return False


def objects_elsewhere_in_clip(con, clip_id: str, room: str | None) -> set[int]:
    """Objects already seen in this clip in a different room: excluded as candidates for a sighting
    in `room` (same rule as co-occurrence, stretched over the walk-through)."""
    room = _norm_room(room)
    if not room:
        return set()
    out: set[int] = set()
    for r in db.rows(con, "SELECT object_id, room, COUNT(*) AS n FROM sightings WHERE clip_id=? AND object_id IS NOT NULL "
                          "GROUP BY object_id, room", (clip_id,)):
        if _norm_room(r["room"]) and _norm_room(r["room"]) != room and r["n"] >= 1:
            out.add(int(r["object_id"]))
    # an object also seen in THIS room in this clip is not excluded (it may be the same thing moved mid-clip)
    same = {int(r["object_id"]) for r in db.rows(con, "SELECT DISTINCT object_id FROM sightings WHERE clip_id=? AND object_id IS NOT NULL AND lower(room)=?", (clip_id, room))}
    return out - same


def decide(con, cands: list[Candidate], q_quality: float = 1.0) -> Decision:
    """q_quality <= 0 marks a blurry/tiny query: its embedding is noise-prone, so it may only
    auto-match an object Spark named the same kind of thing; appearance alone sends it to verify."""
    # "it moved here from a room this clip never scanned" is not a claim we can make: the thing may well
    # still be where it was. Such candidates are dropped; the sighting becomes its own object.
    cands = [c for c in cands if not (c.cross_room and c.old_place_unscanned)]
    if not cands:
        return Decision("new", method="no_candidates")
    top = cands[0]
    if top.contradictions and not (top.sim >= T_MATCH and top.head_match and not any(x.startswith("lightness") for x in top.contradictions)):
        # colour/brand names disagree. Spark naming it the same thing AND a strong visual match
        # overrides that (a navy jacket is 'blue' in daylight and 'dark gray' at night); otherwise
        # the same name alone sends it to a pairwise look, and anything else is a different object.
        if top.head_match and top.sim >= T_LABEL:
            return Decision("verify", None, top.sim, "uncertain_contradiction", cands)
        rest = [c for c in cands if not c.contradictions]
        if not rest:
            return Decision("new", None, top.sim, "contradiction", cands)
        top = rest[0]
    label_floor = T_CATEGORY if head_noun(top.label) in COMMON_NOUNS else T_LABEL
    strong = (top.sim >= T_MATCH or (top.sim >= T_CATEGORY and top.category_match)
              or (top.sim >= label_floor and top.head_match))
    if (q_quality <= 0 or top.ex_quality <= 0) and not top.head_match and top.sim < T_MATCH + 0.1:
        strong = False      # a blurry crop on either side: only the same name (or a near-certain look) may auto-match
    if top.cross_room and top.sim < T_MATCH + 0.1:
        strong = False      # "it moved to another room" is a strong claim: a look-alike there gets a pairwise check first
    # the margin rule applies only when the runner-up is a provably different instance
    runner = next((c for c in cands if c.object_id != top.object_id), None)
    if runner and runner.sim >= T_LABEL and (top.sim - runner.sim) < MARGIN and _distinct_instances(con, top.object_id, runner.object_id):
        return Decision("ask", None, top.sim, "two_instances", cands)
    if strong:
        return Decision("match", top.object_id, top.sim, "embedding", cands)
    if top.sim < T_VERIFY:
        return Decision("new", None, top.sim, "embedding_low", cands)
    if top.sim >= T_VERIFY_ANY or top.head_match or top.category_match:
        return Decision("verify", None, top.sim, "uncertain_move" if top.cross_room else "uncertain", cands)
    # looks a bit like something, but Spark calls it a different kind of thing: not worth a call
    return Decision("new", None, top.sim, "embedding_low_disagree", cands)


def _img_data_url(path: str | Path) -> str:
    im = Image.open(config.local_path(path)).convert("RGB")
    im.thumbnail((384, 384))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return bytes_to_data_url(buf.getvalue(), "image/jpeg")


PAIR_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["same", "confidence", "reason"],
               "properties": {"same": {"type": "boolean"},
                              "confidence": {"type": "number"},
                              "reason": {"type": "string"}}}

PAIR_PROMPT = (
    "Both sightings were recorded in the SAME home, minutes to days apart, by someone filming their rooms. "
    "In one home there is usually exactly one of each item unless it is a consumable or a set, so the default "
    "answer for two sightings of the same kind, colour and size is same=true. "
    "Answer same=false only with positive evidence: a different colour, size, model, brand text, pattern, or a "
    "distinguishing mark present on one and clearly absent on the other. Blur, viewpoint, lighting and crop "
    "differences are NOT evidence of a different object. In particular: a metallic or grey object looks black in "
    "a dim room and silver under a lamp, so black-vs-grey-vs-silver on the same model is lighting, not a different "
    "object; and a set (keys on a ring with a fob, a cable with its plug, a case with its lid) can show a different "
    "part in each crop. Say same=false for those only if a mark, text or shape clearly rules it out. "
    "Sighting A: a '{label_a}' (image 1 is the object{ring_a}). Sighting B: a '{label_b}' (next image is the object{ring_b}). "
    "Return same (bool), confidence (0-1), reason (<=25 words)."
)


def mask_cover(con, sighting_id: int) -> float | None:
    """Fraction of the sighting's box its SAM mask covers (None when there is no mask)."""
    r = db.one(con, "SELECT x1, y1, x2, y2, mask_rle FROM sightings WHERE id=?", (sighting_id,))
    if not r or not r["mask_rle"]:
        return None
    try:
        m = rle.decode(db.jload(r["mask_rle"]))
        sub = m[int(r["y1"]):int(r["y2"]), int(r["x1"]):int(r["x2"])]
        return float((sub > 0).mean()) if sub.size else None
    except Exception:  # noqa: BLE001
        return None


def spark_crop(con, sighting_id: int) -> tuple[str | None, str | None]:
    """(crop to show Spark, surroundings crop). The masked crop when SAM's mask covered a fair share
    of the box; the plain crop when it caught only a part (the keys without their fob, the lid
    without the jar) — Spark judging half an object is how a moved key set becomes two key sets."""
    r = db.one(con, "SELECT crop_path, crop_masked_path, ring_path FROM sightings WHERE id=?", (sighting_id,))
    if not r:
        return None, None
    path = r["crop_masked_path"] or r["crop_path"]
    if r["crop_masked_path"] and r["crop_path"]:
        cover = mask_cover(con, sighting_id)
        if cover is not None and cover < MASK_COVER_PLAIN:
            path = r["crop_path"]
    return path, r["ring_path"]


def absence_context(con, object_id: int, session_id: int) -> dict:
    """What the rest of the footage says about 'is this the same thing as object A, moved?':
    whether A was found anywhere in `session_id`, and whether that walk-through filmed A's last
    place again. {} when A was seen in the session (then it is not a move candidate at all).
    'strong' = A is gone from a spot this walk-through did look at."""
    if db.one(con, "SELECT 1 FROM sightings WHERE object_id=? AND session_id=? LIMIT 1", (object_id, session_id)):
        return {}
    obj = db.one(con, "SELECT last_place_id FROM objects WHERE id=?", (object_id,))
    lp = db.one(con, "SELECT room, surface FROM places WHERE id=?", (obj["last_place_id"],)) if obj and obj["last_place_id"] else None
    ses = db.one(con, "SELECT name FROM sessions WHERE id=?", (session_id,))
    place = f"{lp['room']} {lp['surface']}".replace(" unknown", "").strip() if lp else ""
    scanned = bool(lp) and coverage_of(session_coverage(con, session_id), lp["room"], lp["surface"]) == "scanned"
    return {"place": place, "scanned": scanned, "session": (ses["name"] if ses else "") or "the later walk-through",
            "strong": scanned}


def context_text(ctx: dict | None, b_place: str = "") -> str:
    if not ctx:
        return ""
    where_b = f", on the {b_place}" if b_place else ""
    if ctx.get("scanned"):
        return (f" Context from the rest of the footage: A was last seen on the {ctx['place']}. The later walk-through "
                f"('{ctx['session']}') filmed the {ctx['place']} again and A was not found there or anywhere else in "
                f"it, while B was seen for the first time in that walk-through{where_b}. Things do not vanish from a "
                f"home: an item of the same kind turning up elsewhere while the original is gone from its spot is "
                f"most likely the same item, moved — unless a mark, text, colour or shape clearly rules it out.")
    return (f" Context: A was last seen on the {ctx['place']}; the later walk-through ('{ctx['session']}') did not "
            f"look at that spot, and B appeared{where_b} in it. Judge on appearance only.")


def spark_verify_pair(client: MetaClient, crop_a, ring_a, crop_b, ring_b, label_a: str, label_b: str,
                      context: str = "") -> dict:
    imgs = [_img_data_url(crop_a)]
    has_ra = bool(ring_a and config.local_path(ring_a).exists())
    if has_ra:
        imgs.append(_img_data_url(ring_a))
    imgs.append(_img_data_url(crop_b))
    has_rb = bool(ring_b and config.local_path(ring_b).exists())
    if has_rb:
        imgs.append(_img_data_url(ring_b))
    prompt = PAIR_PROMPT.format(label_a=label_a, label_b=label_b,
                                ring_a=", image 2 its surroundings" if has_ra else "",
                                ring_b=", followed by its surroundings" if has_rb else "") + (context or "")
    return client.spark_json(prompt, images=imgs, schema=PAIR_SCHEMA, schema_name="same_object",
                             max_tokens=4000, kind="spark_pair_verify")


def place_text(con, place_id: int | None) -> str:
    lp = db.one(con, "SELECT room, surface FROM places WHERE id=?", (place_id,)) if place_id else None
    return f"{lp['room']} {lp['surface']}".replace(" unknown", "").strip() if lp else ""


REJECT_STANDS = 0.8         # a Spark 'different' at least this confident is not re-asked by a later pass


def record_verdict(con, object_a: int | None, object_b: int | None, res: dict | None, stage: str) -> None:
    """Remember Spark's pairwise answer so a later pass does not re-litigate a confident 'different'
    with a differently framed question (the absence context is a leading one by design)."""
    if not object_a or not object_b or not isinstance(res, dict) or "same" not in res:
        return
    db.insert(con, "pair_verdicts", {"object_a": int(object_a), "object_b": int(object_b), "same": int(bool(res.get("same"))),
                                     "confidence": float(res.get("confidence") or 0), "reason": str(res.get("reason") or "")[:200],
                                     "stage": stage, "created_at": db.now()})


def prior_reject(con, object_a: int, object_b: int, min_conf: float = REJECT_STANDS) -> dict | None:
    r = db.one(con, "SELECT confidence, reason, stage FROM pair_verdicts WHERE same=0 AND confidence>=? AND "
                    "((object_a=? AND object_b=?) OR (object_a=? AND object_b=?)) ORDER BY confidence DESC LIMIT 1",
               (min_conf, object_a, object_b, object_b, object_a))
    return dict(r) if r else None


def context_allowed(sim: float, attrs_a: dict | None, attrs_b: dict | None) -> bool:
    """The absence context is only added when appearance already makes 'the same thing' plausible:
    a category-level look-alike with compatible colours. Told about a vanished white cap, Spark will
    explain a grey one away as exposure; it should not be told."""
    return sim >= T_CATEGORY and colour_compatible(attrs_a, attrs_b)


def resolve(con, client: MetaClient | None, sighting: dict, q_vecs: dict, q_attrs: dict | None,
            category: str | None, exclude_keyframes: set[int] | None = None,
            exclude_objects: set[int] | None = None, coverage: dict | None = None) -> Decision:
    """Full cascade for one sighting (usually a tracklet's best). Does not write to the DB."""
    q_place = (sighting.get("room"), sighting.get("surface")) if sighting.get("room") else None
    cands = rank_candidates(con, q_vecs, q_attrs, sighting.get("label"), category, exclude_keyframes,
                            exclude_objects=exclude_objects, q_place=q_place, coverage=coverage)
    d = decide(con, cands, q_quality=float(sighting.get("quality") or 0.0) if "quality" in sighting else 1.0)
    if d.action != "verify" or client is None:
        if d.action == "verify":
            d.action = "ask"
        return d
    top = next((c for c in cands if not c.contradictions), cands[0])
    ex = db.one(con, "SELECT label FROM sightings WHERE id=?", (top.exemplar_sighting_id,))
    # the object is A (earlier), the new sighting is B: when A is gone from a spot this walk-through
    # re-filmed, Spark is told so — absence is evidence a crop pair cannot show
    ctx: dict = {}
    o_attrs = _object_attrs(con, top.object_id)
    if top.cross_room and sighting.get("session_id") and context_allowed(top.sim, q_attrs, o_attrs):
        ctx = absence_context(con, top.object_id, int(sighting["session_id"]))
    try:
        crop_a, ring_a = spark_crop(con, top.exemplar_sighting_id)
        crop_b, ring_b = (spark_crop(con, int(sighting["id"])) if sighting.get("id")
                          else (sighting.get("crop_masked_path") or sighting["crop_path"], sighting.get("ring_path")))
        b_place = " ".join(x for x in (sighting.get("room"), sighting.get("surface")) if x and x != "unknown")
        res = spark_verify_pair(client, crop_a, ring_a, crop_b, ring_b,
                                top.label or (ex["label"] if ex else None) or "object", sighting.get("label") or "object",
                                context=context_text(ctx, b_place))
    except Exception as e:  # noqa: BLE001
        d.action = "ask"
        d.spark = {"error": str(e)[:300]}
        return d
    d.spark = res
    conf = float(res.get("confidence", 0))
    if res.get("same") and conf >= 0.6:
        return Decision("match", top.object_id, top.sim, "spark_verify", cands, res, top.object_id)
    if (not res.get("same")) and conf >= 0.7:
        # Spark says two things, the footage says one thing moved: a same-kind look-alike that
        # appeared while the original vanished from a re-filmed spot is left for the owner, not split
        if ctx.get("strong") and (top.category_match or top.head_match) and top.sim >= T_CATEGORY \
                and not hard_contradictions(top.contradictions, q_attrs, o_attrs):
            return Decision("ask", None, top.sim, "spark_reject_vs_absence", cands, res, top.object_id)
        return Decision("new", None, top.sim, "spark_reject", cands, res, top.object_id)
    return Decision("ask", None, top.sim, "spark_unsure", cands, res, top.object_id)


def _object_attrs(con, object_id: int) -> dict:
    o = db.one(con, "SELECT attributes_json FROM objects WHERE id=?", (object_id,))
    return db.jload(o["attributes_json"] if o else None, {}) or {}


# ------------------------------------------------------- within-clip clustering
def cluster_tracklets(tracklets: dict[int, list[dict]]) -> dict[int, int]:
    """Merge geometric tracklets of one clip by embedding similarity.

    tracklets: {tracklet_id: [{"sighting_id", "keyframe_id", "label", "e_obj", "e_patch", "e_ring", "quality"}]}
    Returns {tracklet_id: cluster_root}. Only sightings with quality > 0 drive a merge (blurry and tiny
    crops embed as noise and chain unrelated objects together). Two tracklets merge when their best
    sharp cross-pair similarity is >= T_CLUSTER, or >= T_CLUSTER_LABEL with the same head noun, and
    they never share a keyframe (co-occurrence => different objects). Measured on the first real clips:
    same object across viewpoints maxes at 0.86-0.99; different large objects max at <= 0.54.
    """
    ids = sorted(tracklets)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kfs = {i: {s["keyframe_id"] for s in tracklets[i]} for i in ids}
    rooms = {i: {_norm_room(s.get("room")) for s in tracklets[i]} - {""} for i in ids}
    sharp = {i: [s for s in tracklets[i] if (s.get("quality") or 0) > 0 and s.get("e_obj") is not None] for i in ids}
    heads = {i: head_noun(max(tracklets[i], key=lambda s: s.get("quality") or 0)["label"]) for i in ids}
    pairs = []
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if kfs[a] & kfs[b] or not sharp[a] or not sharp[b]:
                continue
            if rooms[a] and rooms[b] and not (rooms[a] & rooms[b]):
                continue          # one walk-through, two rooms: two objects (a thing is not in the kitchen and the bedroom at once)
            best = max((similarity(sa, sb) for sa in sharp[a] for sb in sharp[b]), default=-1.0)
            same_head = bool(heads[a]) and heads[a] == heads[b]
            if best >= T_CLUSTER or (best >= T_CLUSTER_LABEL and same_head):
                pairs.append((best + (0.05 if same_head else 0), a, b))
    for _, a, b in sorted(pairs, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        ka = set().union(*(kfs[i] for i in ids if find(i) == ra))
        kb = set().union(*(kfs[i] for i in ids if find(i) == rb))
        if ka & kb:
            continue
        parent[rb] = ra
    return {i: find(i) for i in ids}


# ------------------------------------------------------------ object dedupe
def dedupe_objects(con, client: MetaClient | None = None, log=print) -> int:
    """Merge objects that are the same physical item: same category, never co-occurring, and best
    exemplar similarity >= T_MATCH (or >= T_LABEL with the same head noun). Returns merges made."""
    objs = db.rows(con, "SELECT id, category, canonical_label, attributes_json FROM objects ORDER BY id")
    attrs = {o["id"]: db.jload(o["attributes_json"], {}) or {} for o in objs}
    ex = load_object_exemplars(con)
    merged = 0
    alive = {o["id"]: o for o in objs}
    for i, a in enumerate(objs):
        if a["id"] not in alive:
            continue
        for b in objs[i + 1:]:
            if b["id"] not in alive or a["id"] not in alive:
                continue
            ca, cb = a["category"] or "other", b["category"] or "other"
            if ca != cb and "other" not in (ca, cb):
                continue
            if a["id"] not in ex or b["id"] not in ex:
                continue
            if _distinct_instances(con, a["id"], b["id"]):
                continue
            xa = [x for x in ex[a["id"]] if (x.get("quality") or 0) > 0]
            xb = [x for x in ex[b["id"]] if (x.get("quality") or 0) > 0]
            if not xa or not xb:
                continue          # only sharp exemplars may drive a merge
            if attribute_contradictions(attrs.get(a["id"]), attrs.get(b["id"])):
                continue
            best = max((similarity(x, y) for x in xa for y in xb), default=-1.0)
            same_head = head_noun(a["canonical_label"]) == head_noun(b["canonical_label"]) and head_noun(a["canonical_label"])
            same_cat = ca == cb and ca not in ("other", "unknown")
            if best >= T_DEDUPE_ANY or (best >= T_MATCH and (same_cat or same_head)) \
                    or (best >= T_CATEGORY and same_cat and same_head) \
                    or (best >= (T_CATEGORY if head_noun(a["canonical_label"]) in COMMON_NOUNS else T_LABEL) and same_head):
                merge_objects(con, keep=a["id"], drop=b["id"])
                alive.pop(b["id"], None)
                ex[a["id"]] = (ex[a["id"]] + ex[b["id"]])[:MAX_EXEMPLARS]
                merged += 1
                log(f"  merged object {b['id']} '{b['canonical_label']}' into {a['id']} '{a['canonical_label']}' (sim {best:.2f})")
    merged += dedupe_by_name(con, client, alive, attrs, ex, log=log)
    merged += dedupe_by_place(con, client, alive, attrs, ex, log=log)
    return merged


def _session_name(con, session_id: int) -> str:
    r = db.one(con, "SELECT name FROM sessions WHERE id=?", (session_id,))
    return (r["name"] if r else "") or f"walk-through {session_id}"


def _session_sets(con, object_id: int) -> tuple[set[int], dict[int, int]]:
    """Sessions an object was seen in, and its modal place per session."""
    rows = db.rows(con, "SELECT session_id, place_id, COUNT(*) AS n FROM sightings WHERE object_id=? GROUP BY session_id, place_id "
                        "ORDER BY n DESC", (object_id,))
    place: dict[int, int] = {}
    for r in rows:
        place.setdefault(int(r["session_id"]), int(r["place_id"] or 0))
    return set(place), place


def dedupe_by_place(con, client: MetaClient | None, alive: dict, attrs: dict, ex: dict, log=print) -> int:
    """An object seen only in earlier sessions at place P and another seen only in later sessions at
    the same P, compatible in category and colour and never co-seen, is most likely one object that
    Spark named differently from a new angle ('white foam strip' / 'white lint roller'). Spark gets
    the pair; without a client it goes to the user. Only sharp exemplars qualify."""
    objs = sorted(alive.values(), key=lambda o: o["id"])
    info = {o["id"]: _session_sets(con, o["id"]) for o in objs}
    merged = 0
    for i, a in enumerate(objs):
        if a["id"] not in alive:
            continue
        for b in objs[i + 1:]:
            if b["id"] not in alive or a["id"] not in alive:
                continue
            sa, pa = info[a["id"]]
            sb, pb = info[b["id"]]
            if not sa or not sb or (sa & sb):
                continue                      # seen in a common session: handled elsewhere (or two instances)
            first, second = (a, b) if max(sa) < min(sb) else ((b, a) if max(sb) < min(sa) else (None, None))
            if first is None:
                continue
            s1, p1 = info[first["id"]]
            s2, p2 = info[second["id"]]
            if p1[max(s1)] != p2[min(s2)] or not p1[max(s1)]:
                continue                      # last place of the earlier one != first place of the later one
            ca, cb = a["category"] or "other", b["category"] or "other"
            if ca != cb and "other" not in (ca, cb):
                continue
            if attribute_contradictions(attrs.get(a["id"]), attrs.get(b["id"])):
                continue
            xa = [x for x in ex.get(a["id"], []) if (x.get("quality") or 0) > 0]
            xb = [x for x in ex.get(b["id"], []) if (x.get("quality") or 0) > 0]
            if not xa or not xb:
                continue
            best = max((similarity(x, y) for x in xa for y in xb), default=-1.0)
            if best < T_VERIFY:
                continue                      # does not even vaguely look alike
            if prior_reject(con, a["id"], b["id"]):
                continue                      # Spark already told them apart with confidence
            verdict, why = "ask", "same spot, before/after"
            if client is not None:
                xf = xa if first is a else xb
                xs = xb if first is a else xa
                try:
                    crop_a, ring_a = spark_crop(con, xf[0]["sighting_id"])
                    crop_b, ring_b = spark_crop(con, xs[0]["sighting_id"])
                    r = spark_verify_pair(client, crop_a, ring_a, crop_b, ring_b,
                                          first["canonical_label"], second["canonical_label"],
                                          context=" Context: both were seen on the " + place_text(con, p1[max(s1)]) +
                                          f", A in an earlier walk-through and B in a later one ('{_session_name(con, min(s2))}').")
                    record_verdict(con, first["id"], second["id"], r, "dedupe_place")
                    conf = float(r.get("confidence", 0))
                    if r.get("same") and conf >= 0.6:
                        verdict, why = "merge", f"Spark same {conf:.2f}: {r.get('reason', '')}"
                    elif (not r.get("same")) and conf >= 0.6:
                        verdict, why = "keep", f"Spark different {conf:.2f}"
                    else:
                        verdict, why = "ask", f"Spark unsure: {r.get('reason', '')}"
                except Exception as e:  # noqa: BLE001
                    verdict, why = "ask", f"Spark error {str(e)[:80]}"
            if verdict == "merge":
                merge_objects(con, keep=a["id"], drop=b["id"])
                alive.pop(b["id"], None)
                ex[a["id"]] = (xa + xb)[:MAX_EXEMPLARS]
                merged += 1
                log(f"  merged object {b['id']} '{b['canonical_label']}' into {a['id']} '{a['canonical_label']}' by place (sim {best:.2f}; {why})")
            elif verdict == "ask":
                exists = db.one(con, "SELECT id FROM pending_merges WHERE sighting_id=? AND candidate_object_id=? AND resolved=0",
                                (xb[0]["sighting_id"], a["id"]))
                if not exists:
                    db.insert(con, "pending_merges", {"sighting_id": xb[0]["sighting_id"], "candidate_object_id": a["id"],
                                                      "sim": best, "spark_json": db.jdump({"reason": why}), "created_at": db.now()})
                    con.commit()
    return merged


def norm_label(s: str | None) -> str:
    s = _ARTICLES.sub("", (s or "").strip().lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return " ".join(head_noun(w) if i == len(s.split()) - 1 else w for i, w in enumerate(s.split()))


def dedupe_by_name(con, client: MetaClient | None, alive: dict, attrs: dict, ex: dict, log=print) -> int:
    """Single-home prior: two objects Spark named the same thing, never seen together and with no
    contradicting colours, are one object unless a pairwise look says otherwise. Embeddings alone
    miss these when the views differ (scissors flat on a table vs. standing in a cup: cos ~0.2).
    With a client, Spark judges the two best crops; without one, identical labels merge and
    same-head-noun pairs are left for the user."""
    objs = sorted(alive.values(), key=lambda o: o["id"])
    merged = 0
    for i, a in enumerate(objs):
        if a["id"] not in alive:
            continue
        for b in objs[i + 1:]:
            if b["id"] not in alive or a["id"] not in alive:
                continue
            ha, hb = head_noun(a["canonical_label"]), head_noun(b["canonical_label"])
            if not ha or ha != hb or ha in ("object", "item", "thing"):
                continue
            ca, cb = a["category"] or "other", b["category"] or "other"
            if ca != cb and "other" not in (ca, cb):
                continue
            if _distinct_instances(con, a["id"], b["id"]):
                continue
            clash = attribute_contradictions(attrs.get(a["id"]), attrs.get(b["id"]))
            if clash and client is None:
                continue          # Spark can look past a colour name ('white watch' = black strap, white face); we cannot
            same_label = norm_label(a["canonical_label"]) == norm_label(b["canonical_label"])
            xa, xb = ex.get(a["id"]) or [], ex.get(b["id"]) or []
            best = max((similarity(x, y) for x in xa for y in xb), default=-1.0)
            # earlier-only object A vs later-only object B: the footage between them is evidence too
            sa_, pa_ = _session_sets(con, a["id"])
            sb_, pb_ = _session_sets(con, b["id"])
            first, second, ctx, b_place = a, b, {}, ""
            if sa_ and sb_ and not (sa_ & sb_):
                if max(sb_) < min(sa_):
                    first, second = b, a
                s2, p2 = (sb_, pb_) if second is b else (sa_, pa_)
                if context_allowed(best, attrs.get(a["id"]), attrs.get(b["id"])):
                    ctx = absence_context(con, first["id"], min(s2))
                b_place = place_text(con, p2.get(min(s2)))
            if prior_reject(con, a["id"], b["id"]):
                continue          # Spark already told them apart with confidence; no second, leading question
            verdict, why = None, ""
            if client is not None and xa and xb:
                xf, xs = (xa, xb) if first is a else (xb, xa)
                try:
                    crop_a, ring_a = spark_crop(con, xf[0]["sighting_id"])
                    crop_b, ring_b = spark_crop(con, xs[0]["sighting_id"])
                    r = spark_verify_pair(client, crop_a, ring_a, crop_b, ring_b,
                                          first["canonical_label"], second["canonical_label"],
                                          context=context_text(ctx, b_place))
                    record_verdict(con, first["id"], second["id"], r, "dedupe_name")
                    conf = float(r.get("confidence", 0))
                    if r.get("same") and conf >= 0.6:
                        verdict, why = "merge", f"Spark same {conf:.2f}: {r.get('reason', '')}"
                    elif (not r.get("same")) and conf >= 0.7:
                        verdict, why = "keep", f"Spark different {conf:.2f}: {r.get('reason', '')}"
                        strong_prior = (ctx.get("strong") and best >= T_CATEGORY and ca == cb and ca not in ("other", "unknown")
                                        and not hard_contradictions(clash, attrs.get(a["id"]), attrs.get(b["id"])))
                        if strong_prior:
                            # the only thing of its kind that vanished from a re-filmed spot, and a look-alike of the
                            # same kind that appeared elsewhere: Spark's 'different' is not taken on its own
                            verdict, why = "ask", f"Spark different {conf:.2f} ({r.get('reason', '')}) but A vanished from the {ctx['place']} and B appeared"
                            log(f"  kept apart pending review: {a['id']} '{a['canonical_label']}' / {b['id']} '{b['canonical_label']}': {why}")
                    else:
                        verdict, why = "ask", f"Spark unsure: {r.get('reason', '')}"
                except Exception as e:  # noqa: BLE001
                    verdict, why = "ask", f"Spark error {str(e)[:80]}"
            else:
                auto = same_label and ha not in COMMON_NOUNS
                verdict, why = ("merge", "identical label, never co-seen") if auto else ("ask", "same kind of thing")
            if verdict == "ask" and clash and not (ctx.get("strong") and colour_compatible(attrs.get(a["id"]), attrs.get(b["id"]))):
                continue          # Spark was unsure and the colours disagree: leave them apart quietly
            if verdict == "merge":
                merge_objects(con, keep=a["id"], drop=b["id"])
                alive.pop(b["id"], None)
                ex[a["id"]] = (xa + xb)[:MAX_EXEMPLARS]
                merged += 1
                log(f"  merged object {b['id']} '{b['canonical_label']}' into {a['id']} '{a['canonical_label']}' by name (sim {best:.2f}; {why})")
            elif verdict == "ask" and xb:
                exists = db.one(con, "SELECT id FROM pending_merges WHERE sighting_id=? AND candidate_object_id=? AND resolved=0",
                                (xb[0]["sighting_id"], a["id"]))
                if not exists:
                    db.insert(con, "pending_merges", {"sighting_id": xb[0]["sighting_id"], "candidate_object_id": a["id"],
                                                      "sim": best, "spark_json": db.jdump({"reason": why}), "created_at": db.now()})
                    con.commit()
    return merged


def merge_objects(con, keep: int, drop: int) -> None:
    con.execute("UPDATE sightings SET object_id=? WHERE object_id=?", (keep, drop))
    con.execute("UPDATE object_events SET object_id=? WHERE object_id=?", (keep, drop))
    con.execute("UPDATE windows SET object_id=? WHERE object_id=?", (keep, drop))
    con.execute("UPDATE pending_merges SET candidate_object_id=? WHERE candidate_object_id=?", (keep, drop))
    con.execute("UPDATE pair_verdicts SET object_a=? WHERE object_a=?", (keep, drop))
    con.execute("UPDATE pair_verdicts SET object_b=? WHERE object_b=?", (keep, drop))
    con.execute("DELETE FROM pending_merges WHERE sighting_id IN (SELECT id FROM sightings WHERE object_id=?) AND candidate_object_id=?", (keep, keep))
    con.execute("DELETE FROM objects WHERE id=?", (drop,))
    con.commit()
