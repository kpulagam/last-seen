"""Re-render every tracked "play the moment" clip with the current mask style (lastseen/render.py).

    python scripts/rerender.py            # all done windows
    python scripts/rerender.py --db copy.sqlite

Overlay frames (media/overlay/<sighting>.jpg) are drawn on request, so they pick up a style change by
themselves; the tracked clips are cached MP4s and need this pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import api, db, render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    con = db.connect(args.db)
    rows = db.rows(con, "SELECT id, sighting_id FROM windows WHERE status='done' ORDER BY id")
    for w in rows:
        out = api.window_path(w["sighting_id"])
        if out.exists():
            out.unlink()
        p = render.overlay_video(con, w["id"])
        print(f"  window {w['id']} (sighting {w['sighting_id']}): {p}")
    print(f"{len(rows)} clip(s) re-rendered")


if __name__ == "__main__":
    main()
