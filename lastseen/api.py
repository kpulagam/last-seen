"""Framework-free API layer: plain functions returning JSON-able dicts.

app/server.py exposes them over HTTP; scripts/export_static.py writes the same payloads to disk so
the demo can be browsed with no server and no API key (GitHub Pages). Media URLs are relative
("media/crops/<name>") so both deployments resolve them.
"""
from __future__ import annotations

import time
from pathlib import Path

from . import config, db, query, reid, render, state


def _decorate(card: dict | None) -> dict | None:
    if not card:
        return card
    card["overlay_url"] = f"media/overlay/{card['sighting_id']}.jpg"
    if card.get("crop"):
        card["crop_url"] = f"media/crops/{card['crop']}"
    return card


def _sighting_row(con, sid: int) -> dict | None:
    return db.one(con, "SELECT s.*, p.room, p.surface, c.fps FROM sightings s LEFT JOIN places p ON p.id=s.place_id "
                       "JOIN clips c ON c.clip_id=s.clip_id WHERE s.id=?", (sid,))


def ask(con, client, question: str) -> dict:
    res = query.answer(con, client, question)
    res["card"] = _decorate(res.get("card"))
    res["alternatives"] = [_decorate(a) for a in res.get("alternatives", [])]
    res["history"] = [_decorate(a) for a in res.get("history", [])]
    for c in res.get("changes", []):
        c["before_sighting_card"] = _decorate(c.get("before_sighting_card"))
        c["after_sighting_card"] = _decorate(c.get("after_sighting_card"))
    return res


def objects(con) -> dict:
    cat = query.catalog(con)
    for o in cat:
        s = query.best_sighting(con, o["id"])
        o["crop_url"] = f"media/crops/{Path(s['crop_path']).name}" if s and s.get("crop_path") else None
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
    # per-session summary: the agreed place and the representative sighting
    sessions = []
    for ses in db.rows(con, "SELECT DISTINCT s.session_id AS id, ses.name FROM sightings s JOIN sessions ses ON ses.id=s.session_id "
                            "WHERE s.object_id=? ORDER BY s.session_id", (object_id,)):
        rep = state.session_place(con, object_id, ses["id"])
        if rep:
            n = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE object_id=? AND session_id=?", (object_id, ses["id"]))["n"]
            sessions.append({"session_id": ses["id"], "session": ses["name"], "place": f"{rep['p_room']} / {rep['p_surface']}",
                             "sighting_id": rep["id"], "overlay_url": f"media/overlay/{rep['id']}.jpg",
                             "when": query.fmt_ts(rep["ts"]), "n": int(n)})
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
    return {"changes": out, "from": t_from, "to": t_to, "note": note, "sessions": sessions(con)}


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


def overlay_path(con, sighting_id: int) -> Path:
    return render.overlay_png(con, sighting_id)


def window_path(sighting_id: int) -> Path:
    return config.CLIPS / "windows" / f"w_{sighting_id}_overlay.mp4"
