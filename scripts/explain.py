"""Explain a re-ID decision: python scripts/explain.py <sighting_id> [--db path]
Prints the top candidates (best-exemplar similarity, head/category agreement, contradictions) and what
reid.decide() concludes from them — against the object table as it is now."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import db, reid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sighting_id", type=int, nargs="+")
    ap.add_argument("--db", default=None)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    con = db.connect(args.db)
    for sid in args.sighting_id:
        s = db.one(con, "SELECT * FROM sightings WHERE id=?", (sid,))
        e = db.one(con, "SELECT * FROM embeddings WHERE sighting_id=?", (sid,))
        if not s or not e:
            print(f"#{sid}: no sighting/embedding")
            continue
        q = {"e_obj": db.blob_to_vec(e["e_obj"]), "e_patch": db.blob_to_vec(e["e_patch"]), "e_ring": db.blob_to_vec(e["e_ring"])}
        attrs = db.jload(s["attributes_json"], {}) or {}
        print(f"#{sid} s{s['session_id']} f{s['frame_idx']} '{s['label']}' q={s['quality'] or 0:.3f} "
              f"{s['x2'] - s['x1']}x{s['y2'] - s['y1']} obj={s['object_id']} ({s['match_method']} {s['match_score']}) "
              f"attrs: {attrs.get('canonical_label')}/{attrs.get('category')}/{attrs.get('dominant_color')}")
        excl = {int(r["keyframe_id"]) for r in db.rows(con, "SELECT keyframe_id FROM sightings WHERE tracklet_id=? AND clip_id=?",
                                                        (s["tracklet_id"], s["clip_id"]))}
        cands = reid.rank_candidates(con, q, attrs, s["label"], attrs.get("category"), excl, k=args.k)
        for c in cands:
            print(f"    obj {c.object_id:<4} sim={c.sim:.3f} '{c.label}' cat={c.category} head={'Y' if c.head_match else '-'} "
                  f"cat={'Y' if c.category_match else '-'} contr={c.contradictions}")
        d = reid.decide(con, cands, q_quality=float(s['quality'] or 0))
        print(f"    -> {d.action} obj={d.object_id} score={d.score:.3f} ({d.method})")


if __name__ == "__main__":
    main()
