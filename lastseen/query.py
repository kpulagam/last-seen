"""Question -> Muse Spark structured intent -> deterministic SQL -> answer card data."""
from __future__ import annotations

import datetime as dt
import math
import re
import time
from pathlib import Path

from . import db, prompts, state
from .meta_client import MetaClient, MetaAPIError


# --------------------------------------------------------------- catalog
def catalog(con) -> list[dict]:
    rows = db.rows(con, """
        SELECT o.id, o.canonical_label, o.category, o.attributes_json, o.state, o.user_name,
               s.ts AS last_ts, p.room, p.surface,
               (SELECT COUNT(*) FROM sightings x WHERE x.object_id=o.id) AS n_sightings,
               (SELECT MAX(COALESCE(quality,0)) FROM sightings x WHERE x.object_id=o.id) AS best_quality
        FROM objects o
        LEFT JOIN sightings s ON s.id = o.last_sighting_id
        LEFT JOIN places p ON p.id = o.last_place_id
        ORDER BY o.id""")
    out = []
    for r in rows:
        a = db.jload(r["attributes_json"], {}) or {}
        out.append({"id": r["id"], "label": r["user_name"] or r["canonical_label"], "category": r["category"],
                    "attributes": {k: a.get(k) for k in ("dominant_color", "brand_text", "marks", "size_class") if a.get(k)},
                    "state": r["state"], "last_place": f"{r['room']} / {r['surface']}" if r["room"] else None,
                    "last_seen": fmt_ts(r["last_ts"]) if r["last_ts"] else None,
                    "n_sightings": int(r["n_sightings"] or 0), "well_observed": bool((r["n_sightings"] or 0) >= 2 and (r["best_quality"] or 0) > 0)})
    return out


def places(con) -> list[str]:
    return [f"{p['room']} / {p['surface']}" for p in db.rows(con, "SELECT room, surface FROM places ORDER BY room, surface")]


ABSOLUTE_TIMES = False      # the static export sets this: "today"/"yesterday" would go stale on a published page


def fmt_ts(ts: float | None) -> str:
    if not ts:
        return ""
    d = dt.datetime.fromtimestamp(ts)
    today = dt.date.today()
    if not ABSOLUTE_TIMES:
        if d.date() == today:
            return d.strftime("today at %-I:%M %p")
        if d.date() == today - dt.timedelta(days=1):
            return d.strftime("yesterday at %-I:%M %p")
    return d.strftime("%a %b %-d at %-I:%M %p")


def resolve_time(token: str, now: float | None = None) -> float | None:
    now = now or time.time()
    t = (token or "").strip().lower()
    if not t:
        return None
    if t == "now":
        return now
    today0 = dt.datetime.combine(dt.date.today(), dt.time.min).timestamp()
    if t in ("today_start", "today"):
        return today0
    if t in ("yesterday_start", "yesterday"):
        return today0 - 86400
    if t in ("morning", "this_morning"):
        m = today0 + 5 * 3600
        return m if m <= now else m - 86400   # before 5 AM, "this morning" means yesterday's
    m = re.match(r"now-(\d+)h", t)
    if m:
        return now - int(m.group(1)) * 3600
    m = re.match(r"now-(\d+)d", t)
    if m:
        return now - int(m.group(1)) * 86400
    try:
        return dt.datetime.fromisoformat(t).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------- intent
def parse_intent(client: MetaClient | None, question: str, cat: list[dict], plist: list[str]) -> dict:
    if client is None:
        return keyword_intent(question, cat)
    prompt = (f"Catalog: {db.jdump(cat)}\nPlaces: {db.jdump(plist)}\nNow: {dt.datetime.now().isoformat(timespec='minutes')}\n"
              f"Question: {question}")
    try:
        return client.spark_json(prompt, schema=prompts.INTENT_SCHEMA, schema_name="last_seen_intent",
                                 system=prompts.INTENT_SYSTEM, max_tokens=1500, kind="spark_intent")
    except MetaAPIError:
        return keyword_intent(question, cat)


def keyword_intent(question: str, cat: list[dict]) -> dict:
    """Offline fallback: crude keyword matching so the app still works without the API."""
    q = question.lower()
    intent = "locate"
    if any(w in q for w in ("what moved", "what changed", "missing", "since", "what's different", "changes")):
        intent = "changes_between"
    elif q.startswith("where has") or "history" in q:
        intent = "history"
    elif q.startswith("what is on") or q.startswith("what's on") or q.startswith("what is in"):
        intent = "list_place"
    cands = []
    for o in cat:
        toks = set(re.findall(r"[a-z]+", (o["label"] or "").lower()))
        hit = len([t for t in toks if t in q and len(t) > 2])
        if hit:
            cands.append({"object_id": o["id"], "match": min(1.0, 0.4 + 0.3 * hit)})
    cands.sort(key=lambda c: c["match"], reverse=True)
    tf = "morning" if "this morning" in q else ("today_start" if "today" in q else ("yesterday_start" if "yesterday" in q else ""))
    return {"intent": intent, "object_candidates": cands[:3], "place": "", "time_from": tf, "time_to": "now", "clarify": ""}


# ---------------------------------------------------------------- answers
def best_sighting(con, object_id: int) -> dict | None:
    """Sharpest sighting at the object's agreed place in its most recent session."""
    rep = state.session_place(con, object_id)
    ls = db.one(con, "SELECT session_id, MAX(ts) AS t FROM sightings WHERE object_id=? GROUP BY session_id ORDER BY t DESC LIMIT 1",
                (object_id,))
    if ls:
        rep = state.session_place(con, object_id, ls["session_id"]) or rep
    if not rep:
        return None
    return db.one(con, """SELECT s.*, p.room, p.surface, c.path_api, c.fps FROM sightings s
                          LEFT JOIN places p ON p.id=s.place_id JOIN clips c ON c.clip_id=s.clip_id
                          WHERE s.id=?""", (rep["id"],))


def sighting_card(con, s: dict) -> dict:
    obj = db.one(con, "SELECT canonical_label, user_name, state, state_confidence FROM objects WHERE id=?", (s["object_id"],))
    return {"sighting_id": s["id"], "object_id": s["object_id"],
            "label": (obj or {}).get("user_name") or (obj or {}).get("canonical_label") or s["label"],
            "state": (obj or {}).get("state"), "clip_id": s["clip_id"], "frame_idx": s["frame_idx"],
            "t_s": round(s["frame_idx"] / (s.get("fps") or 30), 2), "box": [s["x1"], s["y1"], s["x2"], s["y2"]],
            "place": f"{s.get('room')} / {s.get('surface')}" if s.get("room") else None,
            "ts": s["ts"], "when": fmt_ts(s["ts"]), "crop": Path(s["crop_path"]).name if s.get("crop_path") else None,
            "has_mask": bool(s.get("mask_rle"))}


def window_with_footage(con, t_from: float, t_to: float) -> tuple[float, float, str]:
    """A 'what changed since X' needs footage on both sides of X. If the window holds no sightings
    (asked at 2 AM about 'this morning' when both clips were last night), compare the latest
    session with everything before it instead, and say so."""
    n = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE ts>=? AND ts<=?", (t_from, t_to))
    before = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE ts<?", (t_from,))
    if n and n["n"] and before and before["n"]:
        return t_from, t_to, ""
    latest = db.one(con, "SELECT s.id, s.name, MIN(x.ts) AS t0, MAX(x.ts) AS t1 FROM sessions s JOIN sightings x ON x.session_id=s.id "
                         "GROUP BY s.id ORDER BY t1 DESC LIMIT 1")
    if not latest or latest["t0"] is None:
        return t_from, t_to, ""
    prev = db.one(con, "SELECT COUNT(*) AS n FROM sightings WHERE ts<?", (latest["t0"] - 1,))
    if not prev or not prev["n"]:
        return t_from, t_to, ""
    return float(latest["t0"]) - 1, float(latest["t1"]) + 1, f"comparing your latest clip ('{latest['name']}') with the ones before it"


def answer(con, client: MetaClient | None, question: str) -> dict:
    cat = catalog(con)
    intent = parse_intent(client, question, cat, places(con))
    kind = intent.get("intent", "unsupported")
    out: dict = {"question": question, "intent": intent, "kind": kind, "text": "", "card": None, "alternatives": [],
                 "changes": [], "history": [], "clarify": intent.get("clarify") or ""}
    cands = [c for c in (intent.get("object_candidates") or []) if c.get("match", 0) >= 0.5]
    cands.sort(key=lambda c: c["match"], reverse=True)

    if kind == "locate":
        if not cands:
            near = [o["label"] for o in cat][:6]
            out["text"] = "I haven't seen that in any clip yet."
            out["suggestions"] = near
            return out
        # weigh name matches by how well each object was observed: a single blurry glimpse that Spark
        # happened to call 'massage gun' must not tie with the massage gun seen 13 times
        objmap = {o["id"]: o for o in cat}

        def strength(c: dict) -> float:
            o = objmap.get(c["object_id"]) or {}
            n = o.get("n_sightings") or 1
            return c["match"] * (1 + 0.25 * math.log2(1 + n)) * (1.0 if o.get("well_observed") else 0.6)

        cands.sort(key=strength, reverse=True)
        runner_solid = len(cands) > 1 and (objmap.get(cands[1]["object_id"]) or {}).get("well_observed", False)
        ambiguous = (len(cands) > 1 and runner_solid and strength(cands[1]) / max(1e-6, strength(cands[0])) > 0.8) \
            or bool(out["clarify"])
        if ambiguous:
            out["text"] = out["clarify"] or "Two things match. Which one do you mean?"
            for c in cands[:2]:
                s = best_sighting(con, c["object_id"])
                if s:
                    out["alternatives"].append(sighting_card(con, s))
            return out
        for c in cands[1:3]:   # weaker look-alikes are offered, not asked about
            s = best_sighting(con, c["object_id"])
            if s:
                out["alternatives"].append(sighting_card(con, s))
        s = best_sighting(con, cands[0]["object_id"])
        if not s:
            out["text"] = "I know that object but have no sighting for it."
            return out
        card = sighting_card(con, s)
        out["card"] = card
        obj = db.one(con, "SELECT state, state_confidence FROM objects WHERE id=?", (s["object_id"],))
        txt = f"Last seen on the {card['place']}, {card['when']}." if card["place"] else f"Last seen {card['when']}."
        mv = db.one(con, """SELECT e.ts, p.room, p.surface FROM object_events e LEFT JOIN places p ON p.id=e.from_place_id
                            WHERE e.object_id=? AND e.kind='moved' ORDER BY e.ts DESC LIMIT 1""", (s["object_id"],))
        if mv and mv["room"] and abs((mv["ts"] or 0) - (s["ts"] or 0)) < 6 * 3600:
            txt += f" Before that it was on the {mv['room']} / {mv['surface']}."
        if obj and obj["state"] == "disappeared":
            txt += " I scanned that spot later and it wasn't there anymore."
        elif obj and obj["state"] == "possibly_hidden":
            txt += " That spot was only partly visible in the latest scan, so it may be hidden there."
        if out["alternatives"]:
            txt += f" (I also saw a {out['alternatives'][0]['label']} once — tap it if that's what you meant.)"
        out["text"] = txt
        return out

    if kind == "changes_between":
        t_from = resolve_time(intent.get("time_from")) or (time.time() - 24 * 3600)
        t_to = resolve_time(intent.get("time_to")) or time.time()
        t_from, t_to, note = window_with_footage(con, t_from, t_to)
        out["window_note"] = note
        changes = state.changes_between(con, t_from, t_to)
        place_filter = (intent.get("place") or "").lower().strip()
        if place_filter:
            changes = [c for c in changes if place_filter in ((c.get("from") or "") + " " + (c.get("to") or "")).lower()]
        for c in changes:
            for key in ("before_sighting", "after_sighting"):
                sid = c.get(key)
                if sid:
                    s = db.one(con, "SELECT s.*, p.room, p.surface, c.fps FROM sightings s LEFT JOIN places p ON p.id=s.place_id "
                                    "JOIN clips c ON c.clip_id=s.clip_id WHERE s.id=?", (sid,))
                    c[key + "_card"] = sighting_card(con, s) if s else None
            c["when"] = fmt_ts(c.get("ts"))
        out["changes"] = changes
        if not changes:
            out["text"] = "Nothing I can see moved in that window."
        else:
            sure = [c for c in changes if c.get("confidence") == "high"]
            unsure = [c for c in changes if c.get("confidence") != "high"]

            def summary(cs: list[dict]) -> str:
                moved = [c for c in cs if c["kind"] == "moved"]
                gone = [c for c in cs if c["kind"] == "disappeared"]
                new = [c for c in cs if c["kind"] == "new"]
                bits = []
                if moved:
                    bits.append(f"{len(moved)} moved" + (f" ({', '.join(c['label'] for c in moved[:3])})" if len(moved) <= 3 else ""))
                if gone:
                    bits.append(f"{len(gone)} gone")
                if new:
                    bits.append(f"{len(new)} new")
                return ", ".join(bits)

            if sure:
                out["text"] = (f"{note[0].upper() + note[1:]}: " if note else "") + summary(sure) + "."
                if unsure:
                    out["text"] += f" {len(unsure)} more I only glimpsed once — listed as less sure."
            else:
                out["text"] = (f"{note[0].upper() + note[1:]}: " if note else "") + \
                    f"nothing I'm sure of; {len(unsure)} possible change(s) from single glimpses."
        return out

    if kind == "history":
        if not cands:
            out["text"] = "Which object?"
            return out
        rows = db.rows(con, """SELECT s.*, p.room, p.surface, c.fps FROM sightings s LEFT JOIN places p ON p.id=s.place_id
                               JOIN clips c ON c.clip_id=s.clip_id WHERE s.object_id=? ORDER BY s.ts""", (cands[0]["object_id"],))
        last_place = None
        for r in rows:
            pl = f"{r['room']} / {r['surface']}"
            if pl != last_place:
                out["history"].append(sighting_card(con, r))
                last_place = pl
        out["text"] = f"{len(out['history'])} place(s) over {len(rows)} sightings."
        return out

    if kind == "list_place":
        pf = (intent.get("place") or "").lower()
        rows = db.rows(con, """SELECT o.id FROM objects o JOIN places p ON p.id=o.last_place_id
                               WHERE (p.room || ' / ' || p.surface) LIKE ?""", (f"%{pf}%",))
        for r in rows:
            s = best_sighting(con, r["id"])
            if s:
                out["alternatives"].append(sighting_card(con, s))
        out["text"] = f"{len(out['alternatives'])} object(s) last seen there."
        return out

    out["text"] = "I only answer questions about where things in your clips are."
    return out
