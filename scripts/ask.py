"""Ask a question from the terminal.

    python scripts/ask.py "where did I leave my passport?"
    python scripts/ask.py --offline "what moved since this morning"   # keyword intent, no Spark call
    python scripts/ask.py --window "where is my charger"                # also build the SAM-tracked answer clip
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import db, query, render, windows  # noqa: E402
from lastseen.meta_client import MetaClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--window", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    con = db.connect(args.db)
    client = None if args.offline else MetaClient()
    res = query.answer(con, client, args.question)
    if args.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(res["text"])
        if res.get("card"):
            c = res["card"]
            print(f"  {c['label']} — {c['place']} — {c['when']} — clip {c['clip_id']} frame {c['frame_idx']}")
            png = render.overlay_png(con, c["sighting_id"])
            print(f"  overlay: {png}")
            if args.window:
                w = windows.ensure_window(con, client, c["sighting_id"])
                if w and w["status"] == "done":
                    mp4 = render.overlay_video(con, w["id"])
                    print(f"  tracked clip: {mp4}")
        for a in res.get("alternatives", []):
            print(f"  ? {a['label']} — {a['place']} — {a['when']}")
        for ch in res.get("changes", []):
            ev = ch.get("evidence") or {}
            print(f"  [{ch.get('confidence', '?'):4s}] {ch['kind']:12s} {ch['label']}: {ch.get('from')} -> {ch.get('to')} "
                  f"({ch.get('when')}; sightings before/after {ev.get('before')}/{ev.get('after')})")
        for h in res.get("history", []):
            print(f"  {h['when']}: {h['place']}")


if __name__ == "__main__":
    main()
