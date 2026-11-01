"""Habits: where things usually live, and what is out of place right now.

Every object already has a modal place per walk-through (state.session_place). Its *usual place* is
the place those agree on across the walk-throughs before the one being judged. The judged
walk-through is then compared with it:

* **away** — seen somewhere else, on solid evidence ("your keys are on the kitchen counter; in every
  earlier walk-through they were on the hallway hook");
* **back** — seen at the usual place again after being elsewhere in the previous walk-through.

The wording keeps the evidence count honest: one earlier walk-through gives "in the earlier
walk-through", two agreeing give "in both earlier walk-throughs", three or more give "usually
(n of m)". Nothing here touches re-identification; it is a view over the sightings table.
"""
from __future__ import annotations

from . import db
from .state import SHARP_Q, _evidence, session_place


MIN_SESSIONS = 3    # habits need history: the first two walk-throughs can only show a move


def _solid(ev: tuple[int, float] | None) -> bool:
    return bool(ev) and (ev[0] >= 2 or ev[1] >= SHARP_Q)


def place_label(con, place_id: int | None) -> str:
    p = db.one(con, "SELECT room, surface FROM places WHERE id=?", (place_id,)) if place_id else None
    if not p:
        return "unknown"
    return f"{p['room']} / {p['surface']}" if (p["surface"] or "unknown") != "unknown" else str(p["room"])


def session_order(con) -> list[dict]:
    """Walk-throughs in shooting order: [{id, name, t0}] (only those with sightings)."""
    return db.rows(con, "SELECT ses.id, ses.name, MIN(s.ts) AS t0 FROM sessions ses JOIN sightings s ON s.session_id=ses.id "
                        "GROUP BY ses.id ORDER BY t0")


def object_trail(con, object_id: int, sessions: list[dict]) -> list[dict]:
    """[{session_id, name, place_id, sighting_id, evidence}] for the sessions the object was seen in."""
    trail = []
    for ses in sessions:
        rep = session_place(con, object_id, ses["id"])
        if rep is None:
            continue
        trail.append({"session_id": ses["id"], "name": ses["name"], "place_id": rep["place_id"], "sighting_id": rep["id"],
                      "ts": rep["ts"], "evidence": _evidence(con, object_id, ses["id"], rep["place_id"])})
    return trail


def usual_place(trail: list[dict]) -> dict | None:
    """The place a strict majority of the trail agrees on: {place_id, n_at, n_sessions, solid}.
    None when the trail is shorter than three walk-throughs or no place has a majority — two
    walk-throughs that disagree are a move, not a habit."""
    if len(trail) < MIN_SESSIONS:
        return None
    counts: dict[int, int] = {}
    solid: dict[int, bool] = {}
    for t in trail:
        counts[t["place_id"]] = counts.get(t["place_id"], 0) + 1
        solid[t["place_id"]] = solid.get(t["place_id"], False) or _solid(t["evidence"])
    pid = max(counts, key=counts.get)
    if counts[pid] * 2 <= len(trail):
        return None
    return {"place_id": pid, "n_at": counts[pid], "n_sessions": len(trail), "solid": solid[pid]}


def history_phrase(n_at_earlier: int, n_earlier: int) -> str:
    if n_at_earlier == n_earlier:
        return "in both earlier walk-throughs" if n_earlier == 2 else f"in all {n_earlier} earlier walk-throughs"
    return f"usually ({n_at_earlier} of {n_earlier} earlier walk-throughs)"


def out_of_place(con, session_id: int | None = None) -> dict:
    """What the given (default: latest) walk-through shows away from, or back at, its usual place.
    The usual place is the majority place over every walk-through the object was seen in, the judged
    one included, so a thing put back after a detour counts as back, and a thing that has now been
    seen twice in its new spot stops being 'away'."""
    sessions = session_order(con)
    if not sessions:
        return {"session": None, "session_id": None, "n_earlier": 0, "away": [], "back": []}
    if session_id is None:
        session_id = int(sessions[-1]["id"])
    idx = next((i for i, s in enumerate(sessions) if int(s["id"]) == int(session_id)), None)
    if idx is None or idx == 0:
        return {"session": sessions[idx]["name"] if idx is not None else None, "session_id": session_id,
                "n_earlier": 0, "away": [], "back": []}
    judged = sessions[idx]
    away, back = [], []
    for o in db.rows(con, "SELECT DISTINCT o.id, o.canonical_label, o.user_name FROM objects o JOIN sightings s ON s.object_id=o.id "
                          "WHERE s.session_id=?", (judged["id"],)):
        trail = object_trail(con, o["id"], sessions[:idx + 1])
        if len(trail) < MIN_SESSIONS or trail[-1]["session_id"] != judged["id"]:
            continue
        usual = usual_place(trail)
        if not usual:
            continue
        now = trail[-1]
        earlier = trail[:-1]
        n_at_earlier = sum(1 for t in earlier if t["place_id"] == usual["place_id"])
        label = o["user_name"] or o["canonical_label"]
        rec = {"object_id": o["id"], "label": label, "now_place_id": now["place_id"], "now": place_label(con, now["place_id"]),
               "now_sighting_id": now["sighting_id"], "usual_place_id": usual["place_id"], "usual": place_label(con, usual["place_id"]),
               "usual_sighting_id": next(t["sighting_id"] for t in trail if t["place_id"] == usual["place_id"]),
               "n_at": usual["n_at"], "n_sessions": usual["n_sessions"],
               "confidence": "high" if _solid(now["evidence"]) and usual["solid"] else "low", "ts": now["ts"]}
        if now["place_id"] != usual["place_id"]:
            rec["kind"] = "away"
            rec["history"] = history_phrase(n_at_earlier, len(earlier))
            rec["text"] = f"{label}: now on the {rec['now']} — {rec['history']} it was on the {rec['usual']}."
            away.append(rec)
        elif earlier[-1]["place_id"] != usual["place_id"]:
            rec["kind"] = "back"
            rec["history"] = f"its usual spot, {usual['n_at']} of {usual['n_sessions']} walk-throughs"
            rec["text"] = (f"{label}: back on the {rec['usual']} ({rec['history']}); last time it was on the "
                           f"{place_label(con, earlier[-1]['place_id'])}.")
            back.append(rec)
    key = lambda r: (r["confidence"] != "high", -(r["n_at"]), r["label"])  # noqa: E731
    return {"session": judged["name"], "session_id": int(judged["id"]), "n_earlier": idx,
            "away": sorted(away, key=key), "back": sorted(back, key=key)}


def usual_line(con, object_id: int) -> str:
    """One sentence for a locate answer when the object has a real habit (a majority place over three
    or more walk-throughs): 'Usually it lives on the hallway / hook (2 of 3 walk-throughs).' Else ''."""
    sessions = session_order(con)
    trail = object_trail(con, object_id, sessions)
    usual = usual_place(trail)
    if not usual:
        return ""
    where = place_label(con, usual["place_id"])
    if trail[-1]["place_id"] == usual["place_id"]:
        return f" That is its usual spot ({usual['n_at']} of {usual['n_sessions']} walk-throughs)."
    return f" Usually it lives on the {where} ({usual['n_at']} of {usual['n_sessions']} walk-throughs)."
