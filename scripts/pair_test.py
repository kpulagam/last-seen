"""Ask Spark whether two sightings are the same object, exactly as the re-ID code would.

    python scripts/pair_test.py --a 6 --b 152 --object 8 --session 2
    python scripts/pair_test.py --a 6 --b 152 --no-context

--a / --b are sighting ids (A = earlier object's exemplar, B = the later sighting). --object/--session
add the absence context the pipeline adds for a cross-room candidate (object A's id, B's session).
Prints which crop was sent (masked or plain), the context sentence, and Spark's verdict.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import config, db, reid  # noqa: E402
from lastseen.meta_client import MetaClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True, help="sighting id of the earlier object (A)")
    ap.add_argument("--b", type=int, required=True, help="sighting id of the later sighting (B)")
    ap.add_argument("--object", type=int, help="object id of A, for the absence context")
    ap.add_argument("--session", type=int, help="session id of B, for the absence context")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    con = db.connect(args.db)
    client = MetaClient(run_dir=config.RUNS / "pair_test")
    sa = db.one(con, "SELECT * FROM sightings WHERE id=?", (args.a,))
    sb = db.one(con, "SELECT * FROM sightings WHERE id=?", (args.b,))
    crop_a, ring_a = reid.spark_crop(con, args.a)
    crop_b, ring_b = reid.spark_crop(con, args.b)
    print(f"A #{args.a} '{sa['label']}' ({sa['room']}/{sa['surface']}): {Path(crop_a).name}  cover={reid.mask_cover(con, args.a)}")
    print(f"B #{args.b} '{sb['label']}' ({sb['room']}/{sb['surface']}): {Path(crop_b).name}  cover={reid.mask_cover(con, args.b)}")
    ctx = ""
    if not args.no_context and args.object and args.session:
        c = reid.absence_context(con, args.object, args.session)
        b_place = " ".join(x for x in (sb["room"], sb["surface"]) if x and x != "unknown")
        ctx = reid.context_text(c, b_place)
        print("context:", c)
        print(ctx.strip() or "(none)")
    r = reid.spark_verify_pair(client, crop_a, ring_a, crop_b, ring_b, sa["label"] or "object", sb["label"] or "object", context=ctx)
    print("Spark:", r)
    print(f"{len(client.calls)} call(s); log in {config.RUNS / 'pair_test' / 'api'}")


if __name__ == "__main__":
    main()
