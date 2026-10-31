"""SQLite storage for Last Seen."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import config

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, name TEXT, started_at REAL, notes TEXT);

CREATE TABLE IF NOT EXISTS clips(
  clip_id TEXT PRIMARY KEY, session_id INT REFERENCES sessions(id), path TEXT, path_api TEXT,
  recorded_at REAL, fps REAL, width INT, height INT, n_frames INT, duration_s REAL,
  api_width INT, api_height INT, status TEXT DEFAULT 'new', created_at REAL);

CREATE TABLE IF NOT EXISTS places(
  id INTEGER PRIMARY KEY, room TEXT, surface TEXT, centroid BLOB, n_keyframes INT DEFAULT 0);

CREATE TABLE IF NOT EXISTS keyframes(
  id INTEGER PRIMARY KEY, clip_id TEXT REFERENCES clips(clip_id), frame_idx INT, t_s REAL,
  sharpness REAL, blurry INT DEFAULT 0, jpg_path TEXT, room TEXT, surfaces_json TEXT,
  place_id INT REFERENCES places(id), embedding BLOB, spark_json TEXT,
  UNIQUE(clip_id, frame_idx));

CREATE TABLE IF NOT EXISTS objects(
  id INTEGER PRIMARY KEY, canonical_label TEXT, category TEXT, attributes_json TEXT,
  first_sighting_id INT, last_sighting_id INT, last_place_id INT REFERENCES places(id),
  state TEXT DEFAULT 'new', state_confidence REAL, user_name TEXT, created_at REAL);

CREATE TABLE IF NOT EXISTS sightings(
  id INTEGER PRIMARY KEY, clip_id TEXT REFERENCES clips(clip_id), frame_idx INT,
  keyframe_id INT REFERENCES keyframes(id), session_id INT, tracklet_id INT,
  source TEXT, label TEXT, description TEXT,
  x1 INT, y1 INT, x2 INT, y2 INT, conf REAL, spark_conf REAL,
  mask_rle TEXT, crop_path TEXT, crop_masked_path TEXT, ring_path TEXT,
  in_hand INT DEFAULT 0, surface TEXT, room TEXT, place_id INT REFERENCES places(id),
  object_id INT REFERENCES objects(id), match_method TEXT, match_score REAL,
  attributes_json TEXT, quality REAL, ts REAL);
CREATE INDEX IF NOT EXISTS ix_sight_obj ON sightings(object_id, ts);
CREATE INDEX IF NOT EXISTS ix_sight_clip ON sightings(clip_id, frame_idx);

CREATE TABLE IF NOT EXISTS embeddings(
  sighting_id INT PRIMARY KEY REFERENCES sightings(id), model TEXT,
  e_obj BLOB, e_patch BLOB, e_ring BLOB, lab BLOB);

CREATE TABLE IF NOT EXISTS object_events(
  id INTEGER PRIMARY KEY, object_id INT REFERENCES objects(id), session_id INT, kind TEXT,
  from_place_id INT, to_place_id INT, evidence_sighting_id INT, note TEXT, ts REAL);

CREATE TABLE IF NOT EXISTS windows(
  id INTEGER PRIMARY KEY, object_id INT, sighting_id INT, clip_id TEXT, start_frame INT, end_frame INT,
  trigger TEXT, file_id TEXT, path TEXT, status TEXT DEFAULT 'pending', created_at REAL);

CREATE TABLE IF NOT EXISTS tracks(
  id INTEGER PRIMARY KEY, window_id INT REFERENCES windows(id), frame_idx INT,
  x1 INT, y1 INT, x2 INT, y2 INT, conf REAL, mask_rle TEXT);
CREATE INDEX IF NOT EXISTS ix_tracks_win ON tracks(window_id, frame_idx);

CREATE TABLE IF NOT EXISTS pending_merges(
  id INTEGER PRIMARY KEY, sighting_id INT, candidate_object_id INT, sim REAL, spark_json TEXT,
  resolved INT DEFAULT 0, decision TEXT, created_at REAL);

CREATE TABLE IF NOT EXISTS pair_verdicts(
  id INTEGER PRIMARY KEY, object_a INT, object_b INT, same INT, confidence REAL, reason TEXT, stage TEXT, created_at REAL);

CREATE TABLE IF NOT EXISTS api_calls(
  id INTEGER PRIMARY KEY, run_id TEXT, clip_id TEXT, model TEXT, kind TEXT, request_path TEXT,
  response_path TEXT, usage_json TEXT, cost_usd REAL, latency_ms INT, status INT, ts REAL);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else config.DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p), timeout=30)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    _migrate(con)
    return con


def _migrate(con: sqlite3.Connection) -> None:
    cols = {r["name"] for r in con.execute("PRAGMA table_info(sightings)")}
    if "quality" not in cols:
        con.execute("ALTER TABLE sightings ADD COLUMN quality REAL")
        con.commit()
    cols = {r["name"] for r in con.execute("PRAGMA table_info(embeddings)")}
    if "lab" not in cols:
        con.execute("ALTER TABLE embeddings ADD COLUMN lab BLOB")
        con.commit()


def reset(con: sqlite3.Connection, keep_grounding: bool = True) -> None:
    """Drop every derived row. With keep_grounding, keyframe rows (and their Spark grounding JSON,
    the most expensive call) and the clip rows they reference survive, marked 'new' so the next
    ingest of the same clip reuses the grounding and redoes the rest."""
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        for t in ("tracks", "windows", "pending_merges", "pair_verdicts", "object_events", "embeddings", "sightings",
                  "objects", "places", "api_calls"):
            con.execute(f"DELETE FROM {t}")
        if keep_grounding:
            con.execute("UPDATE keyframes SET place_id=NULL, embedding=NULL")
            con.execute("UPDATE clips SET status='new', session_id=NULL")
        else:
            con.execute("DELETE FROM keyframes")
            con.execute("DELETE FROM clips")
        con.execute("DELETE FROM sessions")
        con.commit()
    finally:
        con.execute("PRAGMA foreign_keys=ON")


@contextmanager
def tx(con: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise


# ------------------------------------------------------------------ helpers
def vec_to_blob(v: np.ndarray | None) -> bytes | None:
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b: bytes | None) -> np.ndarray | None:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32)


def insert(con: sqlite3.Connection, table: str, row: dict[str, Any]) -> int:
    cols = ", ".join(row.keys())
    qs = ", ".join("?" for _ in row)
    cur = con.execute(f"INSERT INTO {table}({cols}) VALUES({qs})", list(row.values()))
    return int(cur.lastrowid)


def update(con: sqlite3.Connection, table: str, row_id: int, row: dict[str, Any], key: str = "id") -> None:
    sets = ", ".join(f"{k}=?" for k in row)
    con.execute(f"UPDATE {table} SET {sets} WHERE {key}=?", list(row.values()) + [row_id])


def rows(con: sqlite3.Connection, sql: str, params: tuple | list = ()) -> list[dict]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def one(con: sqlite3.Connection, sql: str, params: tuple | list = ()) -> dict | None:
    r = con.execute(sql, params).fetchone()
    return dict(r) if r else None


def now() -> float:
    return time.time()


def jdump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def jload(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


def get_or_create_session(con: sqlite3.Connection, name: str | None = None) -> int:
    if name:
        r = one(con, "SELECT id FROM sessions WHERE name=?", (name,))
        if r:
            return int(r["id"])
    sid = insert(con, "sessions", {"name": name or time.strftime("session-%Y%m%d-%H%M%S"),
                                   "started_at": now(), "notes": ""})
    con.commit()
    return sid
