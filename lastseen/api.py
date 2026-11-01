"""Framework-free API layer: plain functions returning JSON-able dicts.

app/server.py exposes them over HTTP; scripts/export_static.py writes the same payloads to disk so
the demo can be browsed with no server and no API key (GitHub Pages). Media URLs are relative
("media/crops/<name>") so both deployments resolve them.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import config, db, habits, query, reid, render, state


def _decorate(card: dict | None) -> dict | None:
    if not card:
        return card
    card["overlay_url"] = f"media/overlay/{card['sighting_id']}.jpg"
    card["frame_url"] = f"media/frame/{card['sighting_id']}.jpg"       # the untouched keyframe, for the reveal
    card["cutout_url"] = f"media/cutout/{card['sighting_id']}.png"      # the object alone, on transparency
    if card.get("crop"):
        card["crop_url"] = f"media/crops/{card['crop']}"
    return card


def _sighting_row(con, sid: int) -> dict | None:
    return db.one(con, "SELECT s.*, p.room, p.surface, c.fps FROM sightings s LEFT JOIN places p ON p.id=s.place_id "
                       "JOIN clips c ON c.clip_id=s.clip_id WHERE s.id=?", (sid,))


def session_trail(con, object_id: int) -> list[dict]:
    """The object's agreed place per walk-through, in shooting order: what the UI draws as the trail."""
    out = []
    for ses in db.rows(con, "SELECT DISTINCT s.session_id AS id, ses.name, MIN(s.ts) AS t0 FROM sightings s JOIN sessions ses ON ses.id=s.session_id "
                            "WHERE s.object_id=? GROUP BY s.session_id ORDER BY t0", (object_id,)):
        rep = state.session_place(con, object_id, ses["id"])
        if rep:
            n = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE object_id=? AND session_id=?", (object_id, ses["id"]))["n"]
            out.append({"session_id": ses["id"], "session": ses["name"], "place": f"{rep['p_room']} / {rep['p_surface']}",
                        "sighting_id": rep["id"], "overlay_url": f"media/overlay/{rep['id']}.jpg",
                        "when": query.fmt_ts(rep["ts"]), "n": int(n)})
    return out


def ask(con, client, question: str) -> dict:
    res = query.answer(con, client, question)
    res["card"] = _decorate(res.get("card"))
    if res["card"] and res["card"].get("object_id"):
        res["trail"] = session_trail(con, int(res["card"]["object_id"]))
    res["alternatives"] = [_decorate(a) for a in res.get("alternatives", [])]
    res["history"] = [_decorate(a) for a in res.get("history", [])]
    for c in res.get("changes", []):
        c["before_sighting_card"] = _decorate(c.get("before_sighting_card"))
        c["after_sighting_card"] = _decorate(c.get("after_sighting_card"))
    if res.get("kind") == "changes_between":
        h = out_of_place(con, None)      # habits are judged on the latest walk-through
        if h["away"] or h["back"]:
            res["habits"] = h
            n = len(h["away"])
            if n:
                res["text"] = (res.get("text") or "").rstrip() + f" {n} thing{'s are' if n != 1 else ' is'} not where " \
                              f"{'they' if n != 1 else 'it'} usually {'live' if n != 1 else 'lives'}."
    return res


def out_of_place(con, session: int | None = None) -> dict:
    """Habit view: objects the latest (or given) walk-through shows away from / back at their usual place."""
    h = habits.out_of_place(con, session)
    for rec in h["away"] + h["back"]:
        for key in ("now_sighting_id", "usual_sighting_id"):
            s = _sighting_row(con, rec[key])
            rec[key.replace("_sighting_id", "_card")] = _decorate(query.sighting_card(con, s)) if s else None
        rec["when"] = query.fmt_ts(rec.get("ts"))
    return h


def objects(con) -> dict:
    cat = query.catalog(con)
    for o in cat:
        s = query.best_sighting(con, o["id"])
        o["crop_url"] = f"media/crops/{Path(s['crop_path']).name}" if s and s.get("crop_path") else None
        o["cutout_url"] = f"media/cutout/{s['id']}.png" if s and s.get("mask_rle") else None
        o["sighting_id"] = s["id"] if s else None
    # well-observed objects first, then by recency
    cat.sort(key=lambda o: (o.get("well_observed", False), o.get("n_sightings", 0)), reverse=True)
    return {"objects": cat}


def object_sightings(con, object_id: int) -> dict:
    rows = db.rows(con, """SELECT s.*, p.room, p.surface, c.fps, ses.name AS session_name FROM sightings s
                           LEFT JOIN places p ON p.id=s.place_id JOIN clips c ON c.clip_id=s.clip_id
                           LEFT JOIN sessions ses ON ses.id=s.session_id WHERE s.object_id=? ORDER BY s.ts""", (object_id,))
    out = []
    for r in rows:
        card = _decorate(query.sighting_card(con, r))
        card["session"] = r["session_name"]
        card["session_id"] = r["session_id"]
        card["quality"] = round(float(r["quality"] or 0), 3)
        out.append(card)
    sessions = session_trail(con, object_id)
    obj = db.one(con, "SELECT id, canonical_label, user_name, category, state, attributes_json FROM objects WHERE id=?", (object_id,))
    events = db.rows(con, """SELECT e.kind, e.note, e.ts, ses.name AS session FROM object_events e LEFT JOIN sessions ses ON ses.id=e.session_id
                             WHERE e.object_id=? ORDER BY e.ts""", (object_id,))
    for e in events:
        e["when"] = query.fmt_ts(e["ts"])
    return {"object": {**(obj or {}), "label": (obj or {}).get("user_name") or (obj or {}).get("canonical_label"),
                       "attributes": db.jload((obj or {}).get("attributes_json"), {})},
            "sightings": out, "sessions": sessions, "events": events}


def sessions(con) -> list[dict]:
    """Sessions in shooting order with their time span (sightings' timestamps)."""
    rows = db.rows(con, """SELECT s.id, s.name, MIN(x.ts) AS t0, MAX(x.ts) AS t1, COUNT(x.id) AS n
                           FROM sessions s JOIN sightings x ON x.session_id=s.id GROUP BY s.id ORDER BY t0""")
    for r in rows:
        r["when"] = query.fmt_ts(r["t0"])
    return rows


def changes(con, frm: str = "today_start", to: str = "now", session: int | None = None) -> dict:
    """Changes in a time window, or — with `session` — what that walk-through changed relative to
    everything seen before it (the way a person thinks about it: 'what moved in last night's clip')."""
    note = ""
    if session:
        ses = db.one(con, "SELECT MIN(ts) AS t0, MAX(ts) AS t1, (SELECT name FROM sessions WHERE id=?) AS name FROM sightings WHERE session_id=?", (session, session))
        if not ses or ses["t0"] is None:
            return {"changes": [], "from": None, "to": None, "note": "no footage in that session"}
        t_from, t_to = float(ses["t0"]) - 1, float(ses["t1"]) + 1
        prev = db.one(con, "SELECT s.name FROM sessions s JOIN sightings x ON x.session_id=s.id WHERE x.ts < ? ORDER BY x.ts DESC LIMIT 1", (t_from,))
        note = f"'{ses['name']}' compared with '{prev['name']}'" if prev else f"'{ses['name']}' is the first walk-through: everything is new"
    else:
        t_from = query.resolve_time(frm) or (time.time() - 86400)
        t_to = query.resolve_time(to) or time.time()
        t_from, t_to, note = query.window_with_footage(con, t_from, t_to)
    out = state.changes_between(con, t_from, t_to)
    for c in out:
        for key in ("before_sighting", "after_sighting"):
            sid = c.get(key)
            if sid:
                s = _sighting_row(con, sid)
                c[key + "_card"] = _decorate(query.sighting_card(con, s)) if s else None
        c["when"] = query.fmt_ts(c.get("ts"))
    return {"changes": out, "from": t_from, "to": t_to, "note": note, "sessions": sessions(con),
            "habits": out_of_place(con, session)}


def pending(con) -> dict:
    rows = db.rows(con, """SELECT pm.*, s.crop_path AS crop_a, s.label AS label_a, s.object_id AS object_a, o.canonical_label AS label_b,
                           (SELECT crop_path FROM sightings WHERE object_id=o.id ORDER BY COALESCE(quality,0) DESC, ts DESC LIMIT 1) AS crop_b
                           FROM pending_merges pm JOIN sightings s ON s.id=pm.sighting_id
                           JOIN objects o ON o.id=pm.candidate_object_id WHERE pm.resolved=0 ORDER BY pm.sim DESC""")
    seen: set[tuple[int, int]] = set()
    out = []

    def weight(oid: int | None) -> tuple[int, float]:
        r = db.one(con, "SELECT COUNT(*) AS n, MAX(COALESCE(quality,0)) AS q FROM sightings WHERE object_id=?", (oid,)) if oid else None
        return (int(r["n"]), float(r["q"] or 0)) if r else (0, 0.0)

    for r in rows:
        key = (r["object_a"] or 0, r["candidate_object_id"])
        if key in seen or r["object_a"] == r["candidate_object_id"]:
            continue
        seen.add(key)
        wa, wb = weight(r["object_a"]), weight(r["candidate_object_id"])
        # not worth a human's time: blurry one-offs, or pairs that neither look alike nor share a name
        same_head = reid.head_noun(r["label_a"]) and reid.head_noun(r["label_a"]) == reid.head_noun(r["label_b"])
        if (wa[1] <= 0 or wb[1] <= 0) and float(r["sim"] or 0) < 0.8:
            continue
        if float(r["sim"] or 0) < 0.5 and not same_head:
            continue
        r["importance"] = wa[0] + wb[0]
        oa = db.one(con, "SELECT canonical_label FROM objects WHERE id=?", (r["object_a"],)) if r["object_a"] else None
        r["label_a"] = (oa or {}).get("canonical_label") or r["label_a"]
        r["crop_a_url"] = f"media/crops/{Path(r['crop_a']).name}" if r["crop_a"] else None
        r["crop_b_url"] = f"media/crops/{Path(r['crop_b']).name}" if r["crop_b"] else None
        r["spark"] = db.jload(r.get("spark_json"))
        out.append(r)
    out.sort(key=lambda r: (r.get("importance", 0), float(r["sim"] or 0)), reverse=True)
    return {"pending": out[:8]}


def merge(con, pending_id: int, same: bool) -> dict:
    pm = db.one(con, "SELECT * FROM pending_merges WHERE id=?", (pending_id,))
    if not pm:
        return {"ok": False, "error": "no such pending merge"}
    s = db.one(con, "SELECT object_id, session_id FROM sightings WHERE id=?", (pm["sighting_id"],))
    if same and s and s["object_id"] and s["object_id"] != pm["candidate_object_id"]:
        reid.merge_objects(con, keep=pm["candidate_object_id"], drop=s["object_id"])
        con.execute("UPDATE sightings SET match_method='user' WHERE object_id=?", (pm["candidate_object_id"],))
        con.execute("UPDATE pending_merges SET resolved=1, decision='same' WHERE id=?", (pending_id,))
    else:
        con.execute("UPDATE pending_merges SET resolved=1, decision='different' WHERE id=?", (pending_id,))
    con.commit()
    sid = (s or {}).get("session_id")
    if sid:
        state.update_object_states(con, sid)
    return {"ok": True}


def clips(con, progress: dict | None = None) -> dict:
    rows = db.rows(con, """SELECT c.clip_id, c.path, c.status, c.recorded_at, c.duration_s, s.name AS session,
                           (SELECT COUNT(*) FROM sightings x WHERE x.clip_id=c.clip_id) AS n_sightings,
                           (SELECT COUNT(DISTINCT object_id) FROM sightings x WHERE x.clip_id=c.clip_id AND object_id IS NOT NULL) AS n_objects
                           FROM clips c LEFT JOIN sessions s ON s.id=c.session_id ORDER BY c.recorded_at""")
    for r in rows:
        r["name"] = Path(r["path"]).name
        r["when"] = query.fmt_ts(r["recorded_at"])
        r["duration_s"] = round(float(r["duration_s"] or 0), 1)
    return {"clips": rows, "progress": progress or {}}


def cutout_path(con, sighting_id: int) -> Path | None:
    """The object cut out on transparency (rendered once, then cached)."""
    out = config.FRAMES / "cutouts" / f"s_{sighting_id}.png"
    if out.exists():
        return out
    try:
        return render.cutout_png(con, sighting_id, out)
    except Exception:  # noqa: BLE001
        return None


def board(con) -> dict:
    """Everything the memory holds, laid out for a picture of the home over time: each object with its
    cut-out and its place in every walk-through it was seen in."""
    ses = sessions(con)
    objs = []
    for o in query.catalog(con):
        s = query.best_sighting(con, o["id"])
        trail = session_trail(con, o["id"])
        objs.append({**o, "sighting_id": s["id"] if s else None,
                     "cutout_url": f"media/cutout/{s['id']}.png" if s and s.get("mask_rle") else None,
                     "crop_url": f"media/crops/{Path(s['crop_path']).name}" if s and s.get("crop_path") else None,
                     "places": {t["session_id"]: {"place": t["place"], "sighting_id": t["sighting_id"], "when": t["when"], "n": t["n"]} for t in trail}})
    objs.sort(key=lambda o: (o.get("well_observed", False), o.get("n_sightings", 0)), reverse=True)
    return {"sessions": ses, "objects": objs}


def frame_path(con, sighting_id: int) -> Path | None:
    """The keyframe a sighting came from, as shot (no mask)."""
    r = db.one(con, "SELECT k.jpg_path FROM sightings s JOIN keyframes k ON k.id=s.keyframe_id WHERE s.id=?", (sighting_id,))
    return config.local_path(r["jpg_path"]) if r and r["jpg_path"] else None


def overlay_path(con, sighting_id: int) -> Path:
    return render.overlay_png(con, sighting_id)


def window_path(sighting_id: int) -> Path:
    return config.CLIPS / "windows" / f"w_{sighting_id}_overlay.mp4"
