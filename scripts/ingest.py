"""Ingest one or more clips.

    python scripts/ingest.py ~/Downloads/IMG_0001.MOV --session morning
    python scripts/ingest.py clips/*.mov --session evening --stride 2 --max-keyframes 40
    python scripts/ingest.py --no-embed ...     # skip DINOv3 (re-ID becomes attributes/Spark only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import db  # noqa: E402
from lastseen.ingest import ingest_clip  # noqa: E402
from lastseen.meta_client import MetaClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--session", default=None)
    ap.add_argument("--stride", type=float, default=None, help="keyframe stride in seconds (default 2)")
    ap.add_argument("--max-keyframes", type=int, default=None)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--reset", action="store_true", help="wipe derived data first; keyframe grounding and the API cache are kept")
    ap.add_argument("--reset-all", action="store_true", help="wipe everything including cached grounding")
    args = ap.parse_args()
    con = db.connect()
    if args.reset or args.reset_all:
        db.reset(con, keep_grounding=not args.reset_all)
        print("database reset")
    client = MetaClient()
    embedder = None
    if not args.no_embed:
        try:
            from lastseen.reid import Embedder
            embedder = Embedder()
            print(f"embedder: {embedder.model_id} on {embedder.device}")
        except Exception as e:  # noqa: BLE001
            print(f"embedder unavailable ({e}); continuing without embeddings")
    for c in args.clips:
        ingest_clip(c, args.session, client=client, embedder=embedder, con=con, stride_sec=args.stride,
                    max_keyframes=args.max_keyframes)


if __name__ == "__main__":
    main()
