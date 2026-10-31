"""SAM 3.1 video tracking windows: a ~150-frame excerpt around a sighting, tracked both ways from the
sighting's box, so the answer card can play the object being followed. Billed per frame (50-frame
minimum), so windows are created lazily (at query time) or on demand for the objects that matter."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config, db, rle, sam_parse
from .video import vsync_args
from .meta_client import MetaClient, MetaAPIError

WINDOW_FRAMES = 150


def cut_frames(video_path: str | Path, start: int, end: int, out: str | Path) -> Path:
    """Frame-exact excerpt [start, end] (inclusive) re-encoded to H.264; window frame 0 == source frame `start`."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vf",
                    f"select='between(n\\,{start}\\,{end})',setpts=N/FRAME_RATE/TB", *vsync_args("passthrough"), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(out)], check=True, capture_output=True)
    return out


def ensure_window(con, client: MetaClient | None, sighting_id: int, span: int = WINDOW_FRAMES,
                  log=print) -> dict | None:
    """Return the window row for this sighting, creating (and tracking) it if needed."""
    w = db.one(con, "SELECT * FROM windows WHERE sighting_id=? AND status='done'", (sighting_id,))
    if w:
        return w
    s = db.one(con, "SELECT * FROM sightings WHERE id=?", (sighting_id,))
    if not s:
        return None
    clip = db.one(con, "SELECT * FROM clips WHERE clip_id=?", (s["clip_id"],))
    n = int(clip["n_frames"] or 0)
    start = max(0, int(s["frame_idx"]) - span // 2)
    end = min(n - 1, start + span - 1)
    start = max(0, min(start, end - 49))  # at least 50 frames when possible
    out = config.CLIPS / "windows" / f"w_{sighting_id}.mp4"
    if not out.exists():
        cut_frames(str(config.local_path(clip["path_api"])), start, end, out)
    wid = db.insert(con, "windows", {"object_id": s["object_id"], "sighting_id": sighting_id, "clip_id": s["clip_id"],
                                     "start_frame": start, "end_frame": end, "trigger": "query", "path": str(out),
                                     "status": "pending", "created_at": db.now()})
    con.commit()
    if client is None:
        return db.one(con, "SELECT * FROM windows WHERE id=?", (wid,))
    try:
        fid = client.upload_file(out)
        local_f = int(s["frame_idx"]) - start
        prompt = sam_parse.build_box_prompt(int(clip["api_width"]), int(clip["api_height"]),
                                            [{"id": 1, "prompts": [{"frame": local_f, "box": (s["x1"], s["y1"], s["x2"], s["y2"])}]}])
        text = client.sam_responses(prompt, video_file_id=fid, include_confidence=True,
                                    propagation_direction="both", stream=True, kind="sam_video_window")
        frames = sam_parse.parse_lines(text)
        n_rows = 0
        for fr in frames:
            for r in fr.records:
                if r.object_id != 1:
                    continue
                try:
                    full = sam_parse.full_frame_mask(r)
                    m = db.jdump(rle.encode(full))
                except Exception:  # noqa: BLE001
                    m = None
                db.insert(con, "tracks", {"window_id": wid, "frame_idx": fr.frame, "x1": r.box[0], "y1": r.box[1],
                                          "x2": r.box[2], "y2": r.box[3], "conf": r.confidence, "mask_rle": m})
                n_rows += 1
        db.update(con, "windows", wid, {"file_id": fid, "status": "done"})
        con.commit()
        log(f"  window {wid}: frames {start}-{end}, {n_rows} tracked frames")
    except MetaAPIError as e:
        db.update(con, "windows", wid, {"status": f"error: {str(e)[:200]}"})
        con.commit()
        log(f"  window {wid} failed: {e}")
    return db.one(con, "SELECT * FROM windows WHERE id=?", (wid,))
