"""Habit rules on a hand-made history (no API, no footage):  python tests/test_habits.py

keys:   hook, counter, hook      -> back on its usual spot (2 of 3)
remote: sofa, sofa, dresser      -> away; "in both earlier walk-throughs" it was on the sofa
cup:    counter, table, shelf    -> no majority: nothing
watch:  counter, sofa            -> two walk-throughs: a move, not a habit
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import db, habits  # noqa: E402


def make_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(db.SCHEMA)
    for i, name in enumerate(("mon", "tue", "wed"), start=1):
        db.insert(con, "sessions", {"id": i, "name": name, "started_at": 1000.0 * i})
        db.insert(con, "clips", {"clip_id": f"c{i}", "session_id": i, "status": "done"})
    places = {}
    for pid, (room, surf) in enumerate([("hallway", "hook"), ("kitchen", "counter"), ("living room", "sofa"),
                                        ("living room", "dresser"), ("kitchen", "table"), ("kitchen", "shelf")], start=1):
        db.insert(con, "places", {"id": pid, "room": room, "surface": surf})
        places[surf] = pid
    objs = {"keys": 1, "remote": 2, "cup": 3, "watch": 4}
    for name, oid in objs.items():
        db.insert(con, "objects", {"id": oid, "canonical_label": name, "category": name, "state": "stationary", "created_at": 0.0})
    trails = {"keys": ["hook", "counter", "hook"], "remote": ["sofa", "sofa", "dresser"],
              "cup": ["counter", "table", "shelf"], "watch": ["counter", "sofa"]}
    sid = 0
    for name, trail in trails.items():
        for ses, surf in enumerate(trail, start=1):
            for k in range(2):                       # two sharp sightings per session = solid evidence
                sid += 1
                db.insert(con, "sightings", {"id": sid, "clip_id": f"c{ses}", "frame_idx": k, "session_id": ses, "label": name,
                                             "object_id": objs[name], "place_id": places[surf], "quality": 0.4,
                                             "ts": 1000.0 * ses + k, "x1": 0, "y1": 0, "x2": 10, "y2": 10})
    con.commit()
    return con


def main() -> None:
    con = make_db()
    h = habits.out_of_place(con)
    by = {r["label"]: r for r in h["away"] + h["back"]}
    print(h["session"], {k: (v["kind"], v["text"]) for k, v in by.items()})
    assert h["session"] == "wed" and h["n_earlier"] == 2
    assert by["keys"]["kind"] == "back" and "hallway / hook" in by["keys"]["text"] and "2 of 3" in by["keys"]["text"]
    assert by["remote"]["kind"] == "away" and "in both earlier walk-throughs" in by["remote"]["text"]
    assert "cup" not in by and "watch" not in by
    assert habits.usual_line(con, 1).startswith(" That is its usual spot (2 of 3")
    assert habits.usual_line(con, 2).startswith(" Usually it lives on the living room / sofa (2 of 3")
    assert habits.usual_line(con, 4) == ""
    # judged on the second walk-through, nothing has a habit yet
    h2 = habits.out_of_place(con, session_id=2)
    assert not h2["away"] and not h2["back"], h2
    print("OK")


if __name__ == "__main__":
    main()
