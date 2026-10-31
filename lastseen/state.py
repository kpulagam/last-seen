"""Object state machine: stationary / moved / disappeared / possibly_hidden / unobserved / new / returned.

Coverage is name-based and deliberately simple: a place (room, surface) counts as *scanned* in a
session if some keyframe of that session named the same room AND listed the surface, or if the
session saw any object on that surface (a wall of hooks is not in Spark's scene surfaces, but the
key sets hanging on it are sightings with surface 'hook'); *partial* if the room matched but the
surface was never named; *unscanned* otherwise.
"""
from __future__ import annotations

from . import db

STATES = ("new", "stationary", "moved", "disappeared", "possibly_hidden", "unobserved", "returned")

_GENERIC_SURFACES = ("floor", "unknown", "in hand", "hand", "hands", "")


def session_coverage(con, session_id: int) -> dict[tuple[str, str], str]:
    """{(room, surface): 'scanned'|'partial'} for everything this session looked at."""
    cov: dict[tuple[str, str], str] = {}
    rows = db.rows(con, "SELECT k.room, k.surfaces_json FROM keyframes k JOIN clips c ON c.clip_id=k.clip_id "
                        "WHERE c.session_id=?", (session_id,))
    for r in rows:
        room = (r["room"] or "unknown").strip().lower()
        surfaces = [s.strip().lower() for s in (db.jload(r["surfaces_json"], []) or [])]
        cov.setdefault((room, "*"), "partial")
        for s in surfaces:
            cov[(room, s)] = "scanned"
    # surfaces that held a sighting were looked at, whether or not the scene description listed them
    for r in db.rows(con, "SELECT DISTINCT room, surface FROM sightings WHERE session_id=?", (session_id,)):
        room = (r["room"] or "unknown").strip().lower()
        surf = (r["surface"] or "").strip().lower()
        if room != "unknown" and surf not in _GENERIC_SURFACES:
            cov.setdefault((room, "*"), "partial")
            cov[(room, surf)] = "scanned"
    return cov


def coverage_of(cov: dict, room: str, surface: str) -> str:
    room, surface = (room or "unknown").lower(), (surface or "unknown").lower()
    if (room, surface) in cov:
        return "scanned"
    # the same spot named two ways: 'hook' / 'wall hook', 'appliance top' / 'metal appliance top'
    for (r, s), _ in cov.items():
        if r == room and s != "*" and len(s) >= 4 and len(surface) >= 4 and (s in surface or surface in s):
            return "scanned"
    if (room, "*") in cov:
        # the room was seen; treat the surface as scanned only if it is a generic/always-visible one
        return "scanned" if surface in ("floor", "unknown", "in hand") else "partial"
    return "unscanned"


def _event(con, object_id: int, session_id: int, kind: str, from_place: int | None, to_place: int | None,
           sighting_id: int | None, note: str = "", ts: float | None = None) -> None:
    if db.one(con, "SELECT id FROM object_events WHERE object_id=? AND session_id=? AND kind=?",
              (object_id, session_id, kind)):
        return
    db.insert(con, "object_events", {"object_id": object_id, "session_id": session_id, "kind": kind,
                                     "from_place_id": from_place, "to_place_id": to_place,
                                     "evidence_sighting_id": sighting_id, "note": note, "ts": ts or db.now()})


def session_time(con, session_id: int) -> float:
    """When the session's footage was shot (latest sighting, else the clip's recorded_at, else now)."""
    r = db.one(con, "SELECT MAX(ts) AS t FROM sightings WHERE session_id=?", (session_id,))
    if r and r["t"]:
        return float(r["t"])
    r = db.one(con, "SELECT MAX(recorded_at + COALESCE(duration_s, 0)) AS t FROM clips WHERE session_id=?", (session_id,))
    return float(r["t"]) if r and r["t"] else db.now()


def session_place(con, object_id: int, session_id: int | None = None, t_from: float | None = None,
                  t_to: float | None = None) -> dict | None:
    """The object's representative sighting for one session (or time window): the place its sightings
    agree on most (weighted by crop quality), and the sharpest sighting at that place. One blurry
    glimpse on the wrong surface must not read as a move."""
    sql = ("SELECT s.*, p.room AS p_room, p.surface AS p_surface FROM sightings s LEFT JOIN places p ON p.id=s.place_id "
           "WHERE s.object_id=? ")
    params: list = [object_id]
    if session_id is not None:
        sql += "AND s.session_id=? "
        params.append(session_id)
    if t_from is not None:
        sql += "AND s.ts >= ? "
        params.append(t_from)
    if t_to is not None:
        sql += "AND s.ts <= ? "
        params.append(t_to)
    rows = db.rows(con, sql + "ORDER BY s.ts DESC", params)
    if not rows:
        return None
    weight: dict = {}
    for r in rows:
        weight[r["place_id"]] = weight.get(r["place_id"], 0.0) + 1.0 + float(r["quality"] or 0.0)
    best_place = max(weight, key=weight.get)
    at = [r for r in rows if r["place_id"] == best_place]
    return max(at, key=lambda r: (float(r["quality"] or 0.0), r["ts"]))


def witnesses_seen(con, object_id: int, before_session: int, after_session: int) -> tuple[int, int]:
    """Were the object's neighbours re-observed? Witnesses = other objects that shared a keyframe with
    it in `before_session`; returns (n_witnesses, n_seen_again_in_after_session). If the neighbours
    were not seen either, that part of the surface was simply not in view, and 'not found' is weak."""
    kfs = [r["keyframe_id"] for r in db.rows(con, "SELECT DISTINCT keyframe_id FROM sightings WHERE object_id=? AND session_id=?",
                                              (object_id, before_session))]
    if not kfs:
        return 0, 0
    q = ",".join("?" * len(kfs))
    wit = [r["object_id"] for r in db.rows(con, f"SELECT DISTINCT object_id FROM sightings WHERE keyframe_id IN ({q}) "
                                               f"AND object_id IS NOT NULL AND object_id<>?", (*kfs, object_id))]
    if not wit:
        return 0, 0
    qw = ",".join("?" * len(wit))
    seen = db.one(con, f"SELECT COUNT(DISTINCT object_id) AS n FROM sightings WHERE session_id=? AND object_id IN ({qw})",
                  (after_session, *wit))["n"]
    return len(wit), int(seen)


def update_object_states(con, session_id: int) -> dict[str, int]:
    """Re-evaluate every object's state relative to this session. Idempotent."""
    cov = session_coverage(con, session_id)
    looked_at = session_time(con, session_id)
    counts: dict[str, int] = {}
    objects = db.rows(con, "SELECT * FROM objects")
    for o in objects:
        oid = o["id"]
        last_sess = db.one(con, "SELECT session_id, MAX(ts) AS t FROM sightings WHERE object_id=? GROUP BY session_id "
                                "ORDER BY t DESC LIMIT 1", (oid,))
        if not last_sess:
            continue
        last_any = session_place(con, oid, last_sess["session_id"])
        in_sess = session_place(con, oid, session_id)
        prev = None
        if in_sess:
            ps = db.one(con, "SELECT session_id, MAX(ts) AS t FROM sightings WHERE object_id=? AND session_id<>? AND ts<? "
                             "GROUP BY session_id ORDER BY t DESC LIMIT 1", (oid, session_id, in_sess["ts"]))
            prev = session_place(con, oid, ps["session_id"]) if ps else None
        new_state, conf = o["state"], o["state_confidence"] or 0.0
        if in_sess:
            if not prev:
                new_state, conf = "new", 1.0
                _event(con, oid, session_id, "new", None, in_sess["place_id"], in_sess["id"], ts=in_sess["ts"])
            elif prev["place_id"] == in_sess["place_id"]:
                new_state, conf = "stationary", 1.0
                if o["state"] == "disappeared":
                    _event(con, oid, session_id, "returned", prev["place_id"], in_sess["place_id"], in_sess["id"], ts=in_sess["ts"])
                    new_state = "returned"
            else:
                new_state, conf = "moved", 1.0
                _event(con, oid, session_id, "moved", prev["place_id"], in_sess["place_id"], in_sess["id"],
                       f"{prev['p_room']}/{prev['p_surface']} -> {in_sess['p_room']}/{in_sess['p_surface']}", ts=in_sess["ts"])
            db.update(con, "objects", oid, {"last_sighting_id": in_sess["id"], "last_place_id": in_sess["place_id"],
                                            "state": new_state, "state_confidence": conf})
        else:
            c = coverage_of(cov, last_any["p_room"], last_any["p_surface"])
            if c == "scanned":
                n_w, n_seen = witnesses_seen(con, oid, last_sess["session_id"], session_id)
                if n_w >= 2 and n_seen * 2 < n_w:
                    c = "partial"        # its neighbours were not seen either: that spot was not really in view
            if c == "scanned":
                new_state, conf = "disappeared", 0.8
                _event(con, oid, session_id, "disappeared", last_any["place_id"], None, last_any["id"],
                       f"not found at {last_any['p_room']}/{last_any['p_surface']}", ts=looked_at)
            elif c == "partial":
                new_state, conf = "possibly_hidden", 0.5
            else:
                new_state, conf = ("unobserved" if o["state"] not in ("disappeared",) else o["state"]), 0.3
            db.update(con, "objects", oid, {"last_sighting_id": last_any["id"], "last_place_id": last_any["place_id"],
                                            "state": new_state, "state_confidence": conf})
        counts[new_state] = counts.get(new_state, 0) + 1
    con.commit()
    return counts


def _evidence(con, object_id: int, session_id: int, place_id: int | None) -> tuple[int, float]:
    """(#sightings of the object at that place in that session, best crop quality there)."""
    r = db.one(con, "SELECT COUNT(*) AS n, MAX(COALESCE(quality,0)) AS q FROM sightings WHERE object_id=? AND session_id=? "
                    "AND (place_id=? OR ? IS NULL)", (object_id, session_id, place_id, place_id))
    return (int(r["n"]), float(r["q"] or 0.0)) if r else (0, 0.0)


SHARP_Q = 0.25   # one crop this sharp is as good as two blurry ones


def change_confidence(kind: str, before_ev: tuple[int, float] | None, after_ev: tuple[int, float] | None) -> str:
    """'high' when each side of the claim rests on more than one glimpse — two or more sightings, or
    one genuinely sharp one; 'low' otherwise. A single blurry sighting on the wrong surface is the
    usual false move."""
    def solid(ev):
        return bool(ev) and (ev[0] >= 2 or ev[1] >= SHARP_Q)
    if kind == "moved":
        return "high" if solid(before_ev) and solid(after_ev) else "low"
    if kind == "disappeared":
        # absence is the weakest claim: the detector misses small blurry things, so "gone" rests on
        # having seen it well at least once and more than once
        return "high" if before_ev and before_ev[0] >= 2 and before_ev[1] >= SHARP_Q else "low"
    return "high" if solid(after_ev) else "low"


def changes_between(con, t_from: float, t_to: float) -> list[dict]:
    """Objects whose latest place inside [t_from, t_to] differs from their latest place before t_from,
    plus objects seen before t_from that have a 'disappeared' event in the window. Each change carries
    a confidence ('high'/'low') from how many sightings back each side of the claim."""
    out: list[dict] = []
    for o in db.rows(con, "SELECT id, canonical_label, state FROM objects"):
        bs = db.one(con, "SELECT session_id, MAX(ts) AS t FROM sightings WHERE object_id=? AND ts<? GROUP BY session_id "
                         "ORDER BY t DESC LIMIT 1", (o["id"], t_from))
        before = session_place(con, o["id"], bs["session_id"]) if bs else None
        ws = db.one(con, "SELECT session_id, MAX(ts) AS t FROM sightings WHERE object_id=? AND ts>=? AND ts<=? "
                         "GROUP BY session_id ORDER BY t DESC LIMIT 1", (o["id"], t_from, t_to))
        within = session_place(con, o["id"], ws["session_id"]) if ws else None
        if before:
            before = {**before, "room": before["p_room"], "surface": before["p_surface"]}
        if within:
            within = {**within, "room": within["p_room"], "surface": within["p_surface"]}
        b_ev = _evidence(con, o["id"], bs["session_id"], before["place_id"]) if before else None
        a_ev = _evidence(con, o["id"], ws["session_id"], within["place_id"]) if within else None
        rec = None
        if within and before and within["place_id"] != before["place_id"]:
            rec = {"object_id": o["id"], "label": o["canonical_label"], "kind": "moved",
                   "from": f"{before['room']} / {before['surface']}", "to": f"{within['room']} / {within['surface']}",
                   "before_sighting": before["id"], "after_sighting": within["id"], "ts": within["ts"]}
        elif within and not before:
            rec = {"object_id": o["id"], "label": o["canonical_label"], "kind": "new", "from": None,
                   "to": f"{within['room']} / {within['surface']}", "before_sighting": None,
                   "after_sighting": within["id"], "ts": within["ts"]}
        elif before and not within:
            ev = db.one(con, "SELECT ts FROM object_events WHERE object_id=? AND kind='disappeared' AND ts>=? ORDER BY ts DESC LIMIT 1",
                        (o["id"], t_from))
            if ev or o["state"] == "disappeared":
                rec = {"object_id": o["id"], "label": o["canonical_label"], "kind": "disappeared",
                       "from": f"{before['room']} / {before['surface']}", "to": None,
                       "before_sighting": before["id"], "after_sighting": None, "ts": (ev or {}).get("ts", t_to)}
        if rec:
            rec["confidence"] = change_confidence(rec["kind"], b_ev, a_ev)
            rec["evidence"] = {"before": b_ev, "after": a_ev}
            if rec["kind"] == "disappeared" and bs and ws is None:
                # which session did we look in? the latest one inside the window
                latest = db.one(con, "SELECT session_id FROM sightings WHERE ts>=? AND ts<=? ORDER BY ts DESC LIMIT 1", (t_from, t_to))
                if latest:
                    n_w, n_seen = witnesses_seen(con, o["id"], bs["session_id"], latest["session_id"])
                    rec["evidence"]["witnesses"] = [n_w, n_seen]
                    if n_w >= 2 and n_seen * 2 < n_w:
                        rec["confidence"] = "low"
                        rec["note"] = "its neighbours weren't in view either"
            out.append(rec)
    prio = {"moved": 3, "returned": 2, "new": 1, "disappeared": 0}
    out.sort(key=lambda r: (r["confidence"] == "high", prio.get(r["kind"], 0), r["ts"] or 0), reverse=True)
    return out
