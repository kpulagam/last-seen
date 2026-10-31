"""Evaluate re-identification against hand labels.

1. python scripts/eval.py sheet            -> writes runs/label_sheet.html (all crops with sighting ids)
2. fill runs/labels.csv with two columns: sighting_id,true_object   (any consistent name per physical object)
3. python scripts/eval.py score            -> pairwise cross-session re-ID precision/recall/F1 + per-object summary

Pairs are counted only across different sessions (within-clip tracklet linking is not re-ID).
"""
from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import config, db  # noqa: E402


def sheet() -> None:
    con = db.connect()
    rows = db.rows(con, """SELECT s.id, s.label, s.object_id, s.session_id, s.crop_path, s.tracklet_id, s.clip_id,
                           o.canonical_label FROM sightings s LEFT JOIN objects o ON o.id=s.object_id
                           ORDER BY s.object_id, s.session_id, s.id""")
    cells = []
    for r in rows:
        cells.append(f'<div class="c"><img src="file://{r["crop_path"]}"><div><b>#{r["id"]}</b> obj {r["object_id"]} '
                     f'<i>{r["canonical_label"] or r["label"]}</i><br>s{r["session_id"]} t{r["tracklet_id"]}</div></div>')
    html = ("<style>body{font:13px sans-serif}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}"
            ".c img{width:100%;aspect-ratio:1;object-fit:cover}</style><div class='g'>" + "".join(cells) + "</div>")
    out = config.RUNS / "label_sheet.html"
    out.write_text(html)
    tmpl = config.RUNS / "labels.csv"
    if not tmpl.exists():
        with open(tmpl, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sighting_id", "true_object"])
            for r in rows:
                w.writerow([r["id"], ""])
    print(f"sheet: {out}\nlabels template: {tmpl}  (fill true_object; same name = same physical object)")


def score() -> None:
    con = db.connect()
    labels: dict[int, str] = {}
    with open(config.RUNS / "labels.csv") as f:
        for row in csv.DictReader(f):
            if row.get("true_object", "").strip():
                labels[int(row["sighting_id"])] = row["true_object"].strip().lower()
    rows = {r["id"]: r for r in db.rows(con, "SELECT id, object_id, session_id, match_method, match_score FROM sightings")}
    ids = [i for i in labels if i in rows]
    tp = fp = fn = tn = 0
    for a, b in combinations(ids, 2):
        if rows[a]["session_id"] == rows[b]["session_id"]:
            continue
        same_true = labels[a] == labels[b]
        same_pred = rows[a]["object_id"] == rows[b]["object_id"]
        if same_true and same_pred:
            tp += 1
        elif same_pred and not same_true:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"cross-session pairs: tp={tp} fp={fp} fn={fn} tn={tn}")
    print(f"re-ID precision={p:.3f} recall={r:.3f} F1={f1:.3f}")
    # per true object: how many predicted objects it was split into; per predicted object: how many true objects merged
    split: dict[str, set] = {}
    merge: dict[int, set] = {}
    for i in ids:
        split.setdefault(labels[i], set()).add(rows[i]["object_id"])
        merge.setdefault(rows[i]["object_id"], set()).add(labels[i])
    print("split objects (true -> predicted ids):", {k: sorted(v) for k, v in split.items() if len(v) > 1})
    print("merged objects (predicted id -> true):", {k: sorted(v) for k, v in merge.items() if len(v) > 1})
    methods: dict[str, int] = {}
    for i in ids:
        methods[rows[i]["match_method"] or "none"] = methods.get(rows[i]["match_method"] or "none", 0) + 1
    print("match methods:", methods)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sheet"
    {"sheet": sheet, "score": score}[cmd]()
