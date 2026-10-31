"""Re-run re-identification (clustering -> attributes -> re-ID -> dedupe -> states) on clips that are
already ingested, without touching keyframes, SAM masks, crops or embeddings.

    python scripts/replay.py                      # offline: reuse stored attributes, no API calls
    python scripts/replay.py --spark              # Spark pairwise verification + fresh attributes
    REID_T_MATCH=0.75 python scripts/replay.py    # try other thresholds (see lastseen/reid.py)
    python scripts/replay.py --db copy.sqlite --requality

Use it to tune thresholds in seconds instead of re-ingesting (which costs API calls and minutes).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from lastseen import config, db, reid, rle  # noqa: E402
from lastseen.ingest import IngestResult, Proposal, assign_tracklets, fill_unknown_rooms, keep_proposal, resolve_clip  # noqa: E402


def load_clip_props(con, clip_id: str, requality: bool, retrack: bool = False) -> tuple[list[Proposal], dict[int, dict]]:
    rows = db.rows(con, "SELECT * FROM sightings WHERE clip_id=? ORDER BY frame_idx, id", (clip_id,))
    props: list[Proposal] = []
    sharps: list[float] = []
    for r in rows:
        if not keep_proposal({"label": r["label"], "confidence": r["spark_conf"] if r["spark_conf"] is not None else 1.0}):
            continue   # the new grounding prompt/filter would not have proposed it
        box = (r["x1"], r["y1"], r["x2"], r["y2"])
        p = Proposal(keyframe_id=r["keyframe_id"], frame_idx=r["frame_idx"], label=r["label"] or "object",
                     description=r["description"] or "", box=box, in_hand=bool(r["in_hand"]),
                     surface=r["surface"] or "unknown", confidence=float(r["spark_conf"] or 0.5),
                     tracklet_id=int(r["tracklet_id"] if r["tracklet_id"] is not None else r["id"]),
                     sam_box=box if r["source"] == "sam_image" else None, sam_conf=r["conf"],
                     sighting_id=r["id"])
        q = r["quality"]
        if r["mask_rle"]:
            try:
                m = rle.decode(db.jload(r["mask_rle"]))
                p.mask_cover = float((m[box[1]:box[3], box[0]:box[2]] > 0).mean()) if m is not None else 1.0
            except Exception:  # noqa: BLE001
                p.mask_cover = 1.0
        if requality or q is None:
            try:
                p.sharp = reid.crop_sharpness(Image.open(config.local_path(r["crop_path"])).convert("RGB"))
            except Exception:  # noqa: BLE001
                p.sharp = 0.0
            p.quality = reid.crop_quality(p.sharp, box)
            if p.mask_cover < reid.MIN_MASK_COVER:
                p.quality = 0.0
        else:
            p.sharp = p.quality = float(q)
        sharps.append(p.sharp)
        props.append(p)
    if requality and sharps:
        reid.apply_relative_blur(props)
    if retrack:
        by_frame: dict[int, list[Proposal]] = {}
        for p in props:
            by_frame.setdefault(p.frame_idx, []).append(p)
        n = assign_tracklets([by_frame[k] for k in sorted(by_frame)])
        for p in props:
            db.update(con, "sightings", p.sighting_id, {"tracklet_id": p.tracklet_id})
    for p in props:
        db.update(con, "sightings", p.sighting_id, {"quality": p.quality})
    vecs: dict[int, dict] = {}
    for e in db.rows(con, "SELECT e.*, s.crop_masked_path FROM embeddings e JOIN sightings s ON s.id=e.sighting_id WHERE s.clip_id=?", (clip_id,)):
        lab = db.blob_to_vec(e["lab"])
        if lab is None or requality:
            try:
                lab = reid.colour_signature(Image.open(config.local_path(e["crop_masked_path"])).convert("RGB"))
            except Exception:  # noqa: BLE001
                lab = None
            con.execute("UPDATE embeddings SET lab=? WHERE sighting_id=?", (db.vec_to_blob(lab), e["sighting_id"]))
        vecs[e["sighting_id"]] = {"e_obj": db.blob_to_vec(e["e_obj"]), "e_patch": db.blob_to_vec(e["e_patch"]),
                                  "e_ring": db.blob_to_vec(e["e_ring"]), "lab": lab}
    con.commit()
    return props, vecs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None, help="sqlite path (default: data/lastseen.sqlite)")
    ap.add_argument("--spark", action="store_true", help="call Spark for attributes + pairwise verification")
    ap.add_argument("--requality", action="store_true", help="recompute crop sharpness/quality from the crop files")
    ap.add_argument("--retrack", action="store_true", help="recompute geometric tracklets from boxes + labels")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    con = db.connect(args.db)
    log = (lambda *a, **k: None) if args.quiet else print

    # stored attributes per sighting, captured before resolve_clip overwrites them
    stored = {r["id"]: db.jload(r["attributes_json"], None) for r in db.rows(con, "SELECT id, attributes_json FROM sightings")}

    def stored_attributes(cid, best, path):
        a = stored.get(best.sighting_id)
        if not a:
            a = {"canonical_label": best.label, "category": "other", "dominant_color": "", "brand_text": ""}
        return a

    client = None
    attributes_fn = stored_attributes
    if args.spark:
        from lastseen.meta_client import MetaClient
        client = MetaClient(run_dir=config.RUNS / time.strftime("replay-%Y%m%d-%H%M%S"))
        attributes_fn = None

    with db.tx(con):
        for t in ("pending_merges", "object_events", "pair_verdicts"):
            con.execute(f"DELETE FROM {t}")
        con.execute("UPDATE sightings SET object_id=NULL, match_method=NULL, match_score=NULL")
        con.execute("UPDATE windows SET object_id=NULL")
        con.execute("DELETE FROM objects")
    clips = db.rows(con, "SELECT c.clip_id, c.session_id, c.recorded_at, s.name FROM clips c JOIN sessions s ON s.id=c.session_id "
                         "WHERE c.status='done' ORDER BY c.session_id, c.recorded_at")
    t0 = time.time()
    totals = {"new": 0, "matched": 0, "pending": 0}
    for c in clips:
        if fill_unknown_rooms(con, c["clip_id"], log=log):
            # re-derive each sighting's room/place from its (now filled) keyframe
            for s_ in db.rows(con, "SELECT s.id, k.room FROM sightings s JOIN keyframes k ON k.id=s.keyframe_id WHERE s.clip_id=?", (c["clip_id"],)):
                db.update(con, "sightings", s_["id"], {"room": s_["room"], "place_id": None})
            con.commit()
            from lastseen.ingest import _assign_places
            _assign_places(con, c["clip_id"])
            con.commit()
        props, vecs = load_clip_props(con, c["clip_id"], args.requality, args.retrack)
        res = IngestResult(clip_id=c["clip_id"])
        log(f"[{c['clip_id']}] session '{c['name']}': {len(props)} sightings, {len(vecs)} embeddings, "
            f"{sum(1 for p in props if p.quality > 0)} sharp")
        resolve_clip(con, client, c["clip_id"], c["session_id"], props, vecs, res, log=log, attributes_fn=attributes_fn)
        totals["new"] += res.objects_new
        totals["matched"] += res.objects_matched
        totals["pending"] += res.objects_pending
        for n in res.notes:
            log("  note:", n)
    objs = db.rows(con, """SELECT o.id, o.canonical_label, o.category, o.state, COUNT(s.id) AS n,
                                  SUM(CASE WHEN s.session_id=(SELECT MIN(id) FROM sessions) THEN 1 ELSE 0 END) AS n1
                           FROM objects o LEFT JOIN sightings s ON s.object_id=o.id GROUP BY o.id ORDER BY n DESC""")
    unassigned = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE object_id IS NULL")["n"]
    print(f"\n{len(objs)} objects ({totals['new']} new / {totals['matched']} matched / {totals['pending']} pending), "
          f"{unassigned} unassigned sightings, {time.time() - t0:.1f}s, thresholds "
          f"match={reid.T_MATCH} cat={reid.T_CATEGORY} label={reid.T_LABEL} verify={reid.T_VERIFY} "
          f"cluster={reid.T_CLUSTER}/{reid.T_CLUSTER_LABEL}")
    multi = [o for o in objs if o["n"] > 1]
    print(f"{len(multi)} objects with >1 sighting; states: " +
          ", ".join(f"{k}={v}" for k, v in sorted(
              {s: sum(1 for o in objs if o['state'] == s) for s in {o['state'] for o in objs}}.items())))
    for o in objs[:40]:
        print(f"  #{o['id']:<4} {str(o['canonical_label'])[:30]:<30} {str(o['category'])[:14]:<14} {str(o['state']):<14} "
              f"{o['n']:>2} sightings (session1: {o['n1']})")


if __name__ == "__main__":
    main()
