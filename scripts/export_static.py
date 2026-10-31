"""Export the app as a static site: the same UI, pre-computed answers, real crops and overlays.
No server, no API key — good for GitHub Pages and for judges who want to click around.

    python scripts/export_static.py site/ -q "where is my massage gun?" -q "what moved since this morning?"
    python scripts/export_static.py site/ --db copy.sqlite --questions questions.txt --offline

Questions are answered with Spark intent parsing unless --offline (keyword intent). Each answer's
overlay, crops and any already-built SAM window clip are copied into site/media/.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import api, config, db, query  # noqa: E402

query.ABSOLUTE_TIMES = True     # a published page must not say "today"

DEFAULT_QUESTIONS = [
    "where did I leave my car keys?",
    "where are my black car keys?",
    "where is my wallet?",
    "where is my watch?",
    "where is the remote?",
    "where is my massage gun?",
    "what did the after walk-through change?",
    "what moved since this morning?",
    "where are my keys?",
    "where is my laptop?",
    "where is my black laptop?",
    "where is my water bottle?",
]


def slug(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", q.lower()).strip("-")[:60]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--db", default=None)
    ap.add_argument("-q", "--question", action="append", default=[])
    ap.add_argument("--questions", help="file with one question per line")
    ap.add_argument("--offline", action="store_true", help="keyword intent instead of Spark")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "api" / "ask").mkdir(parents=True, exist_ok=True)
    (out / "api" / "objects").mkdir(parents=True, exist_ok=True)
    for d in ("crops", "overlay", "window"):
        (out / "media" / d).mkdir(parents=True, exist_ok=True)
    con = db.connect(args.db)
    client = None
    if not args.offline:
        from lastseen.meta_client import MetaClient
        client = MetaClient(run_dir=config.RUNS / "export")
    questions = list(args.question)
    if args.questions:
        questions += [l.strip() for l in Path(args.questions).read_text().splitlines() if l.strip()]
    if not questions:
        questions = DEFAULT_QUESTIONS

    media: set[str] = set()

    def collect(obj) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("crop_url", "overlay_url", "crop_a_url", "crop_b_url", "url") and isinstance(v, str) and v.startswith("media/"):
                    media.add(v)
                else:
                    collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)

    def dump(rel: str, payload) -> None:
        collect(payload)
        (out / rel).write_text(json.dumps(payload, default=str))

    dump("api/objects.json", api.objects(con))
    dump("api/changes.json", api.changes(con, "now-72h", "now"))
    for ses in api.sessions(con)[1:]:
        dump(f"api/changes_session_{ses['id']}.json", api.changes(con, session=ses["id"]))
    dump("api/pending.json", api.pending(con))
    dump("api/clips.json", api.clips(con))
    for o in api.objects(con)["objects"]:
        dump(f"api/objects/{o['id']}.json", api.object_sightings(con, o["id"]))
    answers = {}
    for q in questions:
        res = api.ask(con, client, q)
        answers[q] = f"api/ask/{slug(q)}.json"
        dump(answers[q], res)
        # window clip if one was already built for the answer sighting
        card = res.get("card")
        if card:
            w = api.window_path(card["sighting_id"])
            if w.exists():
                media.add(f"media/window/{card['sighting_id']}.mp4")
        print(f"  {q!r}: {res.get('text', '')[:90]}")
    dump("api/questions.json", {"questions": [{"q": q, "file": f} for q, f in answers.items()]})

    copied = 0
    for m in sorted(media):
        dst = out / m
        if m.startswith("media/crops/"):
            src = config.local_path(config.CROPS / Path(m).name)
        elif m.startswith("media/overlay/"):
            sid = int(Path(m).stem)
            try:
                src = api.overlay_path(con, sid)
            except Exception as e:  # noqa: BLE001
                print("  overlay failed", sid, e)
                continue
        elif m.startswith("media/window/"):
            src = api.window_path(int(Path(m).stem))
        else:
            continue
        if src and Path(src).exists():
            shutil.copyfile(src, dst)
            copied += 1
    html = (config.ROOT / "app" / "static" / "index.html").read_text()
    html = html.replace("<script>", "<script>window.LASTSEEN_STATIC = true;", 1)
    (out / "index.html").write_text(html)
    print(f"wrote {out}: {len(questions)} answers, {copied} media files")


if __name__ == "__main__":
    main()
