"""Contact sheets for eyeballing a run.

    python scripts/sheet.py objects     -> runs/objects_sheet_*.jpg : one row per object, crops grouped by session
    python scripts/sheet.py sightings   -> runs/sightings_sheet_*.jpg : every sighting with its ids and decision
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw  # noqa: E402

from lastseen import config, db  # noqa: E402

CELL = 128


def _thumb(path: str | None, cell: int = CELL) -> Image.Image:
    im = Image.new("RGB", (cell, cell), (225, 222, 215))
    try:
        path = config.local_path(path)
        if path and Path(path).exists():
            t = Image.open(path).convert("RGB")
            t.thumbnail((cell - 4, cell - 4))
            im.paste(t, (2 + (cell - 4 - t.width) // 2, 2 + (cell - 4 - t.height) // 2))
    except Exception:  # noqa: BLE001
        pass
    return im


def _save_split(sheet: Image.Image, stem: str, max_h: int = 3800) -> list[Path]:
    out = []
    n = (sheet.height + max_h - 1) // max_h
    for i in range(n):
        part = sheet.crop((0, i * max_h, sheet.width, min(sheet.height, (i + 1) * max_h)))
        p = config.RUNS / f"{stem}_{i}.jpg"
        part.save(p, quality=80)
        out.append(p)
    return out


def objects_sheet(per_session: int = 4, db_path=None) -> list[Path]:
    con = db.connect(db_path)
    sessions = [r["id"] for r in db.rows(con, "SELECT id FROM sessions ORDER BY id")]
    objs = db.rows(con, """SELECT o.id, o.canonical_label, o.category, o.state, COUNT(s.id) AS n
                           FROM objects o LEFT JOIN sightings s ON s.object_id=o.id GROUP BY o.id ORDER BY n DESC, o.id""")
    unassigned = db.rows(con, "SELECT COUNT(*) AS n FROM sightings WHERE object_id IS NULL")[0]["n"]
    cols = len(sessions) * per_session
    row_h = CELL + 36
    sheet = Image.new("RGB", (220 + cols * CELL + 16, 40 + row_h * (len(objs) + 1)), (245, 243, 238))
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), f"{len(objs)} objects, {unassigned} unassigned sightings. Columns: " +
           "  ".join(f"session {s}" for s in sessions), fill=(0, 0, 0))
    for ri, o in enumerate(objs):
        y = 40 + ri * row_h
        d.text((8, y + 8), f"#{o['id']} {o['canonical_label'] or ''}"[:34], fill=(0, 0, 0))
        d.text((8, y + 24), f"{o['category'] or ''} · {o['state'] or ''} · {o['n']} sightings", fill=(90, 90, 90))
        for si, sid in enumerate(sessions):
            rows = db.rows(con, "SELECT crop_path, quality, frame_idx FROM sightings WHERE object_id=? AND session_id=? "
                                "ORDER BY COALESCE(quality,0) DESC LIMIT ?", (o["id"], sid, per_session))
            for ci, r in enumerate(rows):
                x = 220 + (si * per_session + ci) * CELL
                sheet.paste(_thumb(r["crop_path"]), (x, y))
                d.text((x + 3, y + CELL + 2), f"f{r['frame_idx']} q{(r['quality'] or 0):.1f}", fill=(90, 90, 90))
            if si:
                d.line([(220 + si * per_session * CELL - 2, y), (220 + si * per_session * CELL - 2, y + CELL)], fill=(200, 60, 0), width=2)
    return _save_split(sheet, "objects_sheet")


def sightings_sheet(db_path=None) -> list[Path]:
    con = db.connect(db_path)
    rows = db.rows(con, """SELECT s.id, s.session_id, s.label, s.object_id, s.crop_path, s.quality, s.match_method
                           FROM sightings s ORDER BY s.object_id, s.session_id, s.id""")
    cols = 10
    rws = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL, rws * (CELL + 34)), (245, 243, 238))
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        x, y = (i % cols) * CELL, (i // cols) * (CELL + 34)
        sheet.paste(_thumb(r["crop_path"]), (x, y))
        d.text((x + 3, y + CELL + 2), f"#{r['id']} s{r['session_id']} o{r['object_id']} q{(r['quality'] or 0):.1f}", fill=(0, 0, 0))
        d.text((x + 3, y + CELL + 16), f"{(r['label'] or '')[:16]} {(r['match_method'] or '')[:9]}", fill=(60, 60, 60))
    return _save_split(sheet, "sightings_sheet")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--db=")]
    dbp = next((a[5:] for a in sys.argv[1:] if a.startswith("--db=")), None)
    which = args[0] if args else "objects"
    paths = objects_sheet(db_path=dbp) if which == "objects" else sightings_sheet(db_path=dbp)
    print("\n".join(str(p) for p in paths))
