"""Clip ingestion: keyframes -> Spark grounding -> tracklets -> SAM masks -> crops -> attributes
-> embeddings -> re-identification -> object state.

Run:  python scripts/ingest.py path/to/clip.mp4 --session morning
Every API call is logged under runs/<run_id>/api/. The pipeline is restartable per clip.
"""
from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, db, prompts, reid, rle, sam_parse, video
from .meta_client import MetaClient, MetaAPIError, bytes_to_data_url, image_to_data_url
from .state import session_coverage, update_object_states

# --------------------------------------------------------------------- utils
_ARTICLES = re.compile(r"^(a|an|the|my|some)\s+", re.I)


def norm_label(s: str) -> str:
    s = _ARTICLES.sub("", (s or "").strip().lower())
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:-1] if s.endswith("s") and len(s) > 3 else s  # crude de-plural


def label_compatible(a: str, b: str) -> bool:
    a, b = norm_label(a), norm_label(b)
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return False
    # share the head noun (last token) or most tokens
    return a.split()[-1] == b.split()[-1] or len(ta & tb) >= max(1, min(len(ta), len(tb)) - 1)


def keep_proposal(o: dict, min_conf: float = 0.35) -> bool:
    """Drop grounding proposals we never want as objects: stickers/fragments/blurry things and
    low-confidence guesses. The prompt asks Spark to skip them; this is the backstop."""
    try:
        if float(o.get("confidence", 1.0)) < min_conf:
            return False
    except (TypeError, ValueError):
        pass
    words = set(re.findall(r"[a-z]+", (o.get("label") or "").lower()))
    return not (words & prompts.GROUNDING_SKIP_WORDS)


def clip_id_for(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.name.encode())
    h.update(str(path.stat().st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
    return h.hexdigest()[:12]


def sharpness(img: Image.Image) -> float:
    try:
        import cv2
        g = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())
    except Exception:  # noqa: BLE001
        a = np.asarray(img.convert("L"), dtype=np.float32)
        return float(np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean())


def pil_data_url(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return bytes_to_data_url(buf.getvalue(), "image/jpeg")


# ------------------------------------------------------------------ proposals
@dataclass
class Proposal:
    keyframe_id: int
    frame_idx: int
    label: str
    description: str
    box: tuple[int, int, int, int]       # pixels, half-open
    in_hand: bool
    surface: str
    confidence: float
    tracklet_id: int = -1
    sam_box: tuple[int, int, int, int] | None = None
    sam_conf: float | None = None
    mask_full: np.ndarray | None = None
    mask_crop: np.ndarray | None = None
    sharp: float = 0.0
    quality: float = 0.0
    mask_cover: float = 1.0
    sighting_id: int | None = None


def spark_boxes_to_pixels(objs: list[dict], w: int, h: int) -> list[tuple]:
    out = []
    for o in objs:
        b = o.get("bbox") or {}
        try:
            x1 = int(round(max(0, min(1000, b["x_min"])) * w / 1000))
            y1 = int(round(max(0, min(1000, b["y_min"])) * h / 1000))
            x2 = int(round(max(0, min(1000, b["x_max"])) * w / 1000))
            y2 = int(round(max(0, min(1000, b["y_max"])) * h / 1000))
        except (KeyError, TypeError, ValueError):
            continue
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        out.append((x1, y1, x2, y2))
    return out


def assign_tracklets(props_by_frame: list[list[Proposal]], fps_gap: int = 2) -> int:
    """Link proposals across consecutive keyframes by label compatibility + geometry. Returns #tracklets."""
    next_id = 0
    open_tracks: list[tuple[int, Proposal, int]] = []  # (tracklet_id, last proposal, keyframe ordinal)
    for k, props in enumerate(props_by_frame):
        used: set[int] = set()
        for p in props:
            best, best_score = None, 0.0
            for i, (tid, last, kk) in enumerate(open_tracks):
                if i in used or k - kk > fps_gap or not label_compatible(p.label, last.label):
                    continue
                iou = rle.iou_boxes(p.box, last.box)
                cx = ((p.box[0] + p.box[2]) / 2 - (last.box[0] + last.box[2]) / 2)
                cy = ((p.box[1] + p.box[3]) / 2 - (last.box[1] + last.box[3]) / 2)
                area_p = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                area_l = (last.box[2] - last.box[0]) * (last.box[3] - last.box[1])
                ratio = max(area_p, area_l) / max(1, min(area_p, area_l))
                diag = ((last.box[2] - last.box[0]) ** 2 + (last.box[3] - last.box[1]) ** 2) ** 0.5
                # a loose 'nearby' link is allowed only for the very same label: with just the head noun
                # ('soap bottle' ~ 'water bottle') it chained neighbours on a counter into one tracklet
                same_name = norm_label(p.label) == norm_label(last.label)
                near = (cx * cx + cy * cy) ** 0.5 < 0.5 * diag and same_name
                # box overlap alone links only the same name; a different (head-compatible) name needs a
                # strong overlap on the very next keyframe — a pan over a counter makes neighbours overlap
                if iou >= 0.3 and (same_name or (iou >= 0.55 and k - kk == 1)):
                    score = iou
                else:
                    score = 0.29 if (near and ratio < 2.0) else 0.0
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                tid = open_tracks[best][0]
                open_tracks[best] = (tid, p, k)
                used.add(best)
                p.tracklet_id = tid
            else:
                p.tracklet_id = next_id
                open_tracks.append((next_id, p, k))
                used.add(len(open_tracks) - 1)
                next_id += 1
    return next_id


# ------------------------------------------------------------------ pipeline
@dataclass
class IngestResult:
    clip_id: str
    n_keyframes: int = 0
    n_proposals: int = 0
    n_sightings: int = 0
    n_tracklets: int = 0
    objects_new: int = 0
    objects_matched: int = 0
    objects_pending: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0
    notes: list[str] = field(default_factory=list)


def estimate_cost(calls) -> float:
    total = 0.0
    for c in calls:
        u = c.usage or {}
        if c.kind.startswith("sam"):
            frames = u.get("video_frames_processed")
            total += (frames or 0) * 0.20 / 1000 if frames else 2.5 / 1000
        else:
            total += (u.get("prompt_tokens", 0) or u.get("input_tokens", 0)) * 1.25 / 1e6
            total += (u.get("completion_tokens", 0) or u.get("output_tokens", 0)) * 4.25 / 1e6
    return round(total, 4)


def ingest_clip(src: str | Path, session_name: str | None = None, *, client: MetaClient | None = None,
                embedder: "reid.Embedder | None" = None, con=None, stride_sec: float | None = None,
                max_keyframes: int | None = None, workers: int = 6, log=print) -> IngestResult:
    src = Path(src)
    con = con or db.connect()
    run_dir = config.RUNS / time.strftime("ingest-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _user_log, _logf = log, open(run_dir / "ingest.log", "a")

    def log(msg):  # noqa: F811 — also keep the run's own log file for later diagnosis
        _user_log(msg)
        _logf.write(str(msg) + "\n")
        _logf.flush()

    client = client or MetaClient(run_dir=run_dir)
    client.run_dir = client.run_dir or run_dir
    session_id = db.get_or_create_session(con, session_name)
    clip_id = clip_id_for(src)
    res = IngestResult(clip_id=clip_id)

    # 1. transcode for the API (<=1280 long edge, H.264, faststart) and probe
    api_path = config.CLIPS / f"{clip_id}.mp4"
    if not api_path.exists():
        log(f"[{clip_id}] transcoding {src.name} -> {api_path.name}")
        video.transcode_for_api(src, api_path, max_dim=1280)
    info = video.probe(api_path)
    orig_info = video.probe(src)
    recorded_at = video.recording_start(src, orig_info.duration)  # camera timestamp, else mtime - duration
    existing = db.one(con, "SELECT status FROM clips WHERE clip_id=?", (clip_id,))
    if existing and existing["status"] == "done":
        log(f"[{clip_id}] already ingested; skipping (delete the clip row to redo)")
        return res
    if not existing:
        db.insert(con, "clips", {"clip_id": clip_id, "session_id": session_id, "path": str(src),
                                 "path_api": str(api_path), "recorded_at": recorded_at, "fps": info.fps,
                                 "width": orig_info.width, "height": orig_info.height, "n_frames": info.nframes,
                                 "duration_s": info.duration, "api_width": info.width, "api_height": info.height,
                                 "status": "ingesting", "created_at": db.now()})
    else:
        db.update(con, "clips", clip_id, {"status": "ingesting", "session_id": session_id}, key="clip_id")
    con.commit()

    # 2. keyframes: stride + scene change, sharpest of +-3 frames
    stride = stride_sec or config.KEYFRAME_STRIDE_SEC
    cand = video.keyframe_indices(info, stride_sec=stride, scene_threshold=0.3)
    if max_keyframes and len(cand) > max_keyframes:
        step = len(cand) / max_keyframes
        cand = [cand[int(i * step)] for i in range(max_keyframes)]
    frame_dir = config.FRAMES / clip_id
    want = sorted({max(0, min(info.nframes - 1, k + d)) for k in cand for d in (-3, 0, 3)})
    paths = video.extract_frames(api_path, want, frame_dir)
    keyframes: list[tuple[int, Path, float]] = []
    for k in cand:
        best, best_s = None, -1.0
        for d in (-3, 0, 3):
            idx = max(0, min(info.nframes - 1, k + d))
            p = paths.get(idx)
            if not p:
                continue
            s = sharpness(Image.open(p))
            if s > best_s:
                best, best_s = (idx, p), s
        if best and (not keyframes or best[0] != keyframes[-1][0]):
            keyframes.append((best[0], best[1], best_s))
    med = float(np.median([s for _, _, s in keyframes])) if keyframes else 0.0
    res.n_keyframes = len(keyframes)
    log(f"[{clip_id}] {info.width}x{info.height} {info.fps:.1f}fps {info.nframes} frames; {len(keyframes)} keyframes")

    # 3. Spark grounding per keyframe (parallel; DB writes stay on this thread)
    props_by_frame: list[list[Proposal]] = []
    kf_rows: list[int] = []
    cached: dict[int, dict | None] = {}
    for fidx, fpath, sh in keyframes:
        kf_id = db.one(con, "SELECT id FROM keyframes WHERE clip_id=? AND frame_idx=?", (clip_id, fidx))
        blurry = int(sh < 0.35 * med) if med else 0
        if kf_id:
            kf_id = int(kf_id["id"])
            row = db.one(con, "SELECT spark_json FROM keyframes WHERE id=?", (kf_id,))
            cached[kf_id] = db.jload(row["spark_json"]) if row else None
        else:
            kf_id = db.insert(con, "keyframes", {"clip_id": clip_id, "frame_idx": fidx, "t_s": info.time_at(fidx),
                                                 "sharpness": sh, "blurry": blurry, "jpg_path": str(fpath)})
            cached[kf_id] = None
        kf_rows.append(kf_id)
    con.commit()

    def ground(fpath: Path) -> dict:
        try:
            return client.spark_json(prompts.GROUNDING_PROMPT, images=[image_to_data_url(fpath)],
                                     schema=prompts.GROUNDING_SCHEMA, schema_name="keyframe_objects",
                                     max_tokens=4000, kind="spark_grounding")
        except MetaAPIError as e:
            return {"room": "unknown", "surfaces": [], "objects": [], "_error": str(e)[:200]}

    todo = [(kf_id, fpath) for kf_id, (fidx, fpath, sh) in zip(kf_rows, keyframes) if not cached.get(kf_id)]
    results: dict[int, dict] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(ground, fpath): kf_id for kf_id, fpath in todo}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
    for n, ((fidx, fpath, sh), kf_id) in enumerate(zip(keyframes, kf_rows)):
        g = cached.get(kf_id) or results.get(kf_id) or {"room": "unknown", "surfaces": [], "objects": []}
        if "_error" in g:
            res.notes.append(f"grounding failed on frame {fidx}: {g['_error']}")
        if not cached.get(kf_id):
            db.update(con, "keyframes", kf_id, {"spark_json": db.jdump(g), "room": g.get("room"),
                                                "surfaces_json": db.jdump(g.get("surfaces", []))})
        objs = [o for o in (g.get("objects", []) or []) if keep_proposal(o)]
        boxes = spark_boxes_to_pixels(objs, info.width, info.height)
        plist: list[Proposal] = []
        for o, b in zip([o for o in objs if o.get("bbox")], boxes):
            plist.append(Proposal(keyframe_id=kf_id, frame_idx=fidx, label=o.get("label", "object"),
                                  description=o.get("description", ""), box=b, in_hand=bool(o.get("in_hand")),
                                  surface=o.get("surface", "unknown"), confidence=float(o.get("confidence", 0.5))))
        props_by_frame.append(plist)
        log(f"  kf {n + 1}/{len(keyframes)} frame {fidx}: {g.get('room')} — {len(plist)} objects")
    con.commit()
    fill_unknown_rooms(con, clip_id, log=log)
    res.n_proposals = sum(len(p) for p in props_by_frame)

    # 4. tracklets across keyframes
    res.n_tracklets = assign_tracklets(props_by_frame)

    # 5. SAM box prompts per keyframe (parallel) -> tight boxes + masks; then crops + sightings rows
    def segment(fpath: Path, plist: list[Proposal]) -> list[tuple[int, dict]]:
        frame_img = Image.open(fpath).convert("RGB")
        W, H = frame_img.size
        data_url = image_to_data_url(fpath)
        out: list[tuple[int, dict]] = []
        for start in range(0, len(plist), config.MAX_TRACKED_OBJECTS):
            batch = plist[start:start + config.MAX_TRACKED_OBJECTS]
            objs = [{"id": i + 1, "prompts": [{"frame": 0, "box": p.box}]} for i, p in enumerate(batch)]
            prompt = sam_parse.build_box_prompt(W, H, objs)
            try:
                text = client.sam_responses(prompt, image_data_url=data_url, include_confidence=True,
                                            stream=False, kind="sam_box_image")
                frames = sam_parse.parse_lines(text)
                recs = {r.object_id: r for r in (frames[0].records if frames else [])}
            except MetaAPIError as e:
                recs = {"_error": str(e)[:200]}
            out.append((start, recs))
        return out

    sam_results: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(segment, fpath, plist): kf_id
                for (fidx, fpath, sh), plist, kf_id in zip(keyframes, props_by_frame, kf_rows) if plist}
        for fut in as_completed(futs):
            sam_results[futs[fut]] = fut.result()

    for (fidx, fpath, sh), plist, kf_id in zip(keyframes, props_by_frame, kf_rows):
        if not plist:
            continue
        frame_img = Image.open(fpath).convert("RGB")
        for start, recs in sam_results.get(kf_id, []):
            if "_error" in recs:
                res.notes.append(f"SAM failed on frame {fidx}: {recs['_error']}")
                recs = {}
            batch = plist[start:start + config.MAX_TRACKED_OBJECTS]
            for i, p in enumerate(batch):
                r = recs.get(i + 1)
                if r is not None and rle.iou_boxes(r.box, p.box) >= 0.3:
                    p.sam_box, p.sam_conf = r.box, r.confidence
                    try:
                        p.mask_full = sam_parse.full_frame_mask(r)
                    except Exception as e:  # noqa: BLE001
                        res.notes.append(f"mask decode failed f{fidx} obj{i}: {e}")
                        p.mask_full = None
                else:
                    p.sam_box, p.sam_conf = None, None
        # crops + rows
        kf_room = db.one(con, "SELECT room FROM keyframes WHERE id=?", (kf_id,))["room"]
        for p in plist:
            box = p.sam_box or p.box
            views = reid.crop_views(frame_img, box, p.mask_full)
            p.mask_crop = views["mask_crop"] if p.mask_full is not None else None
            p.mask_cover = views["mask_cover"] if p.mask_full is not None else 1.0
            p.sharp = reid.crop_sharpness(views["crop"])
            p.quality = reid.crop_quality(p.sharp, box)
            if p.mask_cover < reid.MIN_MASK_COVER:
                p.quality = 0.0          # SAM returned slivers: the masked crop is mostly grey fill
                p.mask_crop = None       # embed/colour the plain crop instead
            sid = db.insert(con, "sightings", {
                "quality": p.quality,
                "clip_id": clip_id, "frame_idx": fidx, "keyframe_id": kf_id, "session_id": session_id,
                "tracklet_id": p.tracklet_id, "source": "sam_image" if p.sam_box else "spark",
                "label": p.label, "description": p.description,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3], "conf": p.sam_conf, "spark_conf": p.confidence,
                "mask_rle": db.jdump(rle.encode(p.mask_full)) if p.mask_full is not None else None,
                "in_hand": int(p.in_hand), "surface": p.surface, "room": kf_room,
                "ts": recorded_at + info.time_at(fidx)})
            paths_out = {}
            for key in ("crop", "crop_masked", "ring"):
                out = config.CROPS / f"{sid}_{key}.png"
                views[key].save(out)
                paths_out[key] = str(out)
            db.update(con, "sightings", sid, {"crop_path": paths_out["crop"], "crop_masked_path": paths_out["crop_masked"],
                                              "ring_path": paths_out["ring"]})
            p.sighting_id = sid
            res.n_sightings += 1
        con.commit()
        log(f"  frame {fidx}: {len(plist)} sightings ({sum(1 for p in plist if p.sam_box)} with SAM masks)")

    # 6. places from keyframes (room + surface)  — simple name-based places
    _assign_places(con, clip_id)

    # 7. embeddings for every sighting, then embedding-based clustering of the geometric tracklets
    all_props = [p for pl in props_by_frame for p in pl if p.sighting_id]
    if embedder is None:
        try:
            embedder = reid.Embedder()
            log(f"  embeddings: {embedder.model_id} on {embedder.device}")
        except Exception as e:  # noqa: BLE001
            res.notes.append(f"no embedder ({e}); re-ID will use attributes only")
            embedder = None
    vecs_by_sid: dict[int, dict[str, np.ndarray]] = {}
    if embedder is not None:
        for p in all_props:
            row = db.one(con, "SELECT crop_masked_path, ring_path, crop_path FROM sightings WHERE id=?", (p.sighting_id,))
            cm = Image.open(row["crop_masked_path"] if p.mask_crop is not None else row["crop_path"]).convert("RGB")
            e_obj, e_patch = embedder.embed_obj_patch(cm, p.mask_crop)
            e_ring = embedder.embed_images([Image.open(row["ring_path"]).convert("RGB")])[0]
            lab = reid.colour_signature(cm, p.mask_crop)
            vecs_by_sid[p.sighting_id] = {"e_obj": e_obj, "e_patch": e_patch, "e_ring": e_ring, "lab": lab}
            con.execute("INSERT OR REPLACE INTO embeddings(sighting_id, model, e_obj, e_patch, e_ring, lab) VALUES(?,?,?,?,?,?)",
                        (p.sighting_id, embedder.model_id, db.vec_to_blob(e_obj), db.vec_to_blob(e_patch),
                         db.vec_to_blob(e_ring), db.vec_to_blob(lab)))
        con.commit()
    reid.apply_relative_blur(all_props)
    for p in all_props:
        db.update(con, "sightings", p.sighting_id, {"quality": p.quality})
    con.commit()

    resolve_clip(con, client, clip_id, session_id, all_props, vecs_by_sid, res, workers=workers, log=log)
    db.update(con, "clips", clip_id, {"status": "done"}, key="clip_id")
    con.commit()
    res.api_calls = len(client.calls)
    res.cost_usd = estimate_cost(client.calls)
    for c in client.calls:
        db.insert(con, "api_calls", {"run_id": run_dir.name, "clip_id": clip_id, "model": c.kind.split("_")[0],
                                     "kind": c.kind, "response_path": c.response_path, "usage_json": db.jdump(c.usage),
                                     "latency_ms": int(c.elapsed_s * 1000), "status": c.status, "ts": db.now()})
    con.commit()
    (run_dir / "ingest_result.json").write_text(json.dumps(res.__dict__, indent=1, default=str))
    log(f"[{clip_id}] done: {res.n_sightings} sightings, {res.n_tracklets} tracklets, "
        f"{res.objects_new} new / {res.objects_matched} matched / {res.objects_pending} pending; "
        f"{res.api_calls} calls ≈ ${res.cost_usd}")
    return res


def label_votes(plist: list[Proposal], k: int = 3) -> str:
    """The cluster's grounding labels, most frequent first, as a quoted list for the attributes prompt."""
    counts: dict[str, int] = {}
    for p in plist:
        key = (p.label or "").strip().lower()
        if key:
            counts[key] = counts.get(key, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"'{lbl}' ({n}x)" if n > 1 else f"'{lbl}'" for lbl, n in top) or "'object'"


def spark_attributes(client: MetaClient, best: Proposal, crop_masked_path: str, labels: str | None = None) -> dict:
    """Canonical label / category / colour / brand for a cluster, from its sharpest crop (masked when
    the mask covered the object; plain when SAM caught only part of it — a key set masked down to its
    keys gets named 'silver keys' and its black fob is lost)."""
    path = crop_masked_path
    if best.mask_cover < reid.MASK_COVER_PLAIN and crop_masked_path and "_crop_masked" in crop_masked_path:
        path = crop_masked_path.replace("_crop_masked", "_crop")
    try:
        return client.spark_json(prompts.ATTRIBUTES_PROMPT.format(labels=labels or f"'{best.label}'"),
                                 images=[pil_data_url(Image.open(path))],
                                 schema=prompts.ATTRIBUTES_SCHEMA, schema_name="object_attributes",
                                 max_tokens=4000, kind="spark_attributes")
    except MetaAPIError as e:
        return {"canonical_label": best.label, "category": "other", "dominant_color": "", "secondary_colors": [],
                "material": "", "brand_text": "", "marks": [], "size_class": "small", "description": best.description,
                "_error": str(e)[:200]}


def resolve_clip(con, client, clip_id: str, session_id: int, all_props: list[Proposal],
                 vecs_by_sid: dict[int, dict[str, np.ndarray]], res: IngestResult, *, workers: int = 6,
                 log=print, attributes_fn=None) -> None:
    """Stages 7-8 for one clip whose sightings (and embeddings) are already in the DB:
    cluster the geometric tracklets by appearance, name each cluster (Spark attributes), re-identify
    each cluster against the object table, dedupe objects, update object states.

    attributes_fn(cid, best_proposal, crop_masked_path) -> attrs dict; default calls Spark. A replay can
    pass a function that returns stored attributes so thresholds can be re-tuned without API calls."""
    by_track: dict[int, list[Proposal]] = {}
    for p in all_props:
        by_track.setdefault(p.tracklet_id, []).append(p)
    if vecs_by_sid:
        room_of = {r["id"]: r["room"] for r in db.rows(con, "SELECT id, room FROM sightings WHERE clip_id=?", (clip_id,))}
        tl = {tid: [{"sighting_id": p.sighting_id, "keyframe_id": p.keyframe_id, "label": p.label,
                     "quality": p.quality, "room": room_of.get(p.sighting_id), **vecs_by_sid.get(p.sighting_id, {})} for p in plist]
              for tid, plist in by_track.items()}
        roots = reid.cluster_tracklets(tl)
        clusters: dict[int, list[Proposal]] = {}
        for tid, plist in by_track.items():
            clusters.setdefault(roots[tid], []).extend(plist)
        log(f"  {len(by_track)} geometric tracklets -> {len(clusters)} clusters by embedding")
    else:
        clusters = by_track
    res.n_tracklets = len(clusters)

    # 7b. per-cluster attributes (best sighting) in parallel, then re-ID sequentially against the object table
    for plist in clusters.values():
        plist.sort(key=lambda p: (p.quality, (p.sam_conf or 0.5) * (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])), reverse=True)

    # crop paths are looked up on this thread (sqlite connections are not shared across threads)
    masked_path: dict[int, str | None] = {}
    for cid, plist in clusters.items():
        brow = db.one(con, "SELECT crop_masked_path FROM sightings WHERE id=?", (plist[0].sighting_id,))
        masked_path[cid] = str(config.local_path(brow["crop_masked_path"])) if brow and brow["crop_masked_path"] else None

    def attributes_for(cid: int, best: Proposal) -> tuple[int, dict]:
        path = masked_path.get(cid)
        if attributes_fn is not None:
            return cid, attributes_fn(cid, best, path)
        return cid, spark_attributes(client, best, path, labels=label_votes(clusters[cid]))

    attrs_by_cluster: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(attributes_for, cid, plist[0]) for cid, plist in clusters.items()]
        for fut in as_completed(futs):
            cid, a = fut.result()
            attrs_by_cluster[cid] = a
            if "_error" in a:
                res.notes.append(f"attributes failed for cluster {cid}: {a['_error']}")

    # Things that stayed put are matched first, so that by the time a cluster in another room is
    # weighed as "X moved here", X's old spot has had its chance to claim X. Within that, well-observed
    # clusters go first: a 7-sighting bottle should not lose its identity to a 1-sighting look-alike.
    cov = session_coverage(con, session_id)

    def stays(cid: int) -> bool:
        best = clusters[cid][0]
        if best.sighting_id not in vecs_by_sid:
            return False
        row = db.one(con, "SELECT room, surface, label FROM sightings WHERE id=?", (best.sighting_id,))
        if not row or not row["room"]:
            return False
        a = attrs_by_cluster.get(cid) or {}
        top = reid.rank_candidates(con, vecs_by_sid[best.sighting_id], a, row["label"], a.get("category"),
                                   {p.keyframe_id for p in clusters[cid]}, k=1,
                                   q_place=(row["room"], row["surface"]), coverage=cov)
        return bool(top) and not top[0].cross_room and top[0].sim >= reid.T_VERIFY

    order = sorted(clusters, key=lambda c: (stays(c), len([p for p in clusters[c] if p.quality > 0]), clusters[c][0].quality),
                   reverse=True)
    for cid in order:
        plist = clusters[cid]
        best = plist[0]
        brow = db.one(con, "SELECT * FROM sightings WHERE id=?", (best.sighting_id,))
        attrs = attrs_by_cluster.get(cid) or {"canonical_label": best.label, "category": "other"}
        for p in plist:
            db.update(con, "sightings", p.sighting_id, {"attributes_json": db.jdump(attrs)})
        # candidates are scored against every sighting of the cluster (max), not just the best crop
        cluster_kfs = {p.keyframe_id for p in plist}
        elsewhere = reid.objects_elsewhere_in_clip(con, clip_id, brow["room"])
        decision = reid.Decision("new", method="no_embedding")
        if vecs_by_sid:
            best_d: reid.Decision | None = None
            for p in sorted(plist, key=lambda p: p.quality, reverse=True)[:3]:
                if p.sighting_id not in vecs_by_sid:
                    continue
                prow = db.one(con, "SELECT * FROM sightings WHERE id=?", (p.sighting_id,))
                d = reid.resolve(con, None, prow, vecs_by_sid[p.sighting_id], attrs, attrs.get("category"), cluster_kfs,
                                 exclude_objects=elsewhere, coverage=cov)
                if best_d is None or d.score > best_d.score:
                    best_d = d
            decision = best_d or decision
            if (decision.action == "verify" or (decision.action == "ask" and decision.method.startswith("uncertain"))) \
                    and best.sighting_id in vecs_by_sid:
                decision = reid.resolve(con, client, brow, vecs_by_sid[best.sighting_id], attrs, attrs.get("category"), cluster_kfs,
                                        exclude_objects=elsewhere, coverage=cov)
        if decision.action == "match":
            oid = decision.object_id
            res.objects_matched += 1
        elif best.quality <= 0 and all(p.quality <= 0 for p in plist) and decision.action != "ask":
            oid = None  # blurry/tiny only and nothing to attach to: keep the sightings, make no object
            res.notes.append(f"cluster {cid} '{best.label}' left unassigned (low quality)")
        else:
            name = attrs.get("canonical_label") or best.label
            if db.one(con, "SELECT id FROM objects WHERE lower(canonical_label)=lower(?)", (name,)) and brow["room"]:
                name = f"{name} ({brow['room']})"      # two silver bottles: tell them apart by where they live
            oid = db.insert(con, "objects", {"canonical_label": name,
                                             "category": attrs.get("category"), "attributes_json": db.jdump(attrs),
                                             "first_sighting_id": best.sighting_id, "state": "new", "created_at": db.now()})
            res.objects_new += 1
            if decision.spark and decision.verified_against:
                reid.record_verdict(con, decision.verified_against, oid, decision.spark, "resolve")
            if decision.action == "ask":
                for c in decision.candidates[:2]:
                    db.insert(con, "pending_merges", {"sighting_id": best.sighting_id, "candidate_object_id": c.object_id,
                                                      "sim": c.sim, "spark_json": db.jdump(decision.spark), "created_at": db.now()})
                res.objects_pending += 1
        for p in plist:
            db.update(con, "sightings", p.sighting_id, {"object_id": oid, "match_method": decision.method,
                                                        "match_score": decision.score})
        con.commit()
        log(f"  cluster {cid} '{attrs.get('canonical_label')}' x{len(plist)} -> {decision.action} (obj {oid}, {decision.method}, {decision.score:.2f})")

    # 7c. merge objects that are the same physical item (never co-occurring, high similarity)
    n_merged = reid.dedupe_objects(con, client, log=log)
    if n_merged:
        log(f"  dedupe: {n_merged} object merges")

    # 8. object state
    update_object_states(con, session_id)
    con.commit()


def fill_unknown_rooms(con, clip_id: str, log=print) -> int:
    """Rooms come from Spark one keyframe at a time, and close-ups fool it: an open drawer has no
    room, and two frames of a kitchen counter filmed from 40 cm were called 'bedroom'. The camera
    does not teleport, so (1) an 'unknown' keyframe takes the nearest named room and (2) a run of
    at most two keyframes whose room differs from the SAME room on both sides is relabelled to it."""
    rows = db.rows(con, "SELECT id, frame_idx, room FROM keyframes WHERE clip_id=? ORDER BY frame_idx", (clip_id,))
    norm = lambda r: "" if (r or "").strip().lower() in ("", "unknown", "none") else r.strip().lower()  # noqa: E731
    rooms = [norm(r["room"]) for r in rows]
    known = [(r["frame_idx"], rooms[i]) for i, r in enumerate(rows) if rooms[i]]
    fixed = 0
    for i, r in enumerate(rows):
        if not rooms[i] and known:
            rooms[i] = min(known, key=lambda kr: abs(kr[0] - r["frame_idx"]))[1]
            fixed += 1
    # islands: [.., A, B, A, ..] or [.., A, B, B, A, ..]  ->  A
    i = 0
    while i < len(rooms):
        j = i
        while j + 1 < len(rooms) and rooms[j + 1] == rooms[i]:
            j += 1
        run = j - i + 1
        if 0 < i and j + 1 < len(rooms) and run <= 2 and rooms[i - 1] == rooms[j + 1] and rooms[i - 1] != rooms[i]:
            for k in range(i, j + 1):
                rooms[k] = rooms[i - 1]
                fixed += 1
        i = j + 1
    for r, room in zip(rows, rooms):
        if norm(r["room"]) != room:
            db.update(con, "keyframes", r["id"], {"room": room})
    if fixed:
        con.commit()
        log(f"  {fixed} keyframe room label(s) corrected from their neighbours (unknown or isolated)")
    return fixed


def _assign_places(con, clip_id: str) -> None:
    """Place = (room, surface). Each sighting gets the place of its room + its own surface."""
    for s in db.rows(con, "SELECT id, room, surface FROM sightings WHERE clip_id=? AND place_id IS NULL", (clip_id,)):
        room = (s["room"] or "unknown").strip().lower()
        surf = (s["surface"] or "unknown").strip().lower()
        if surf in ("hand", "hands"):
            surf = "in hand"
        row = db.one(con, "SELECT id FROM places WHERE room=? AND surface=?", (room, surf))
        pid = int(row["id"]) if row else db.insert(con, "places", {"room": room, "surface": surf})
        db.update(con, "sightings", s["id"], {"place_id": pid})
    con.commit()
