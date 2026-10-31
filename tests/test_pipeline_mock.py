"""Offline end-to-end test of the pipeline with synthetic clips and a mock API.

    python tests/test_pipeline_mock.py
Two sessions: in session 2 the red mug moves, the yellow book disappears, a grey charger appears.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="lastseen-test-"))
os.environ["MODEL_API_KEY"] = "mock"
os.environ["LASTSEEN_DATA"] = str(TMP)

from lastseen import config  # noqa: E402

# redirect data dirs into the temp folder
config.DATA = TMP / "data"
config.CLIPS, config.FRAMES, config.CROPS, config.RUNS = config.DATA / "clips", config.DATA / "frames", config.DATA / "crops", TMP / "runs"
config.DB_PATH = config.DATA / "lastseen.sqlite"
for d in (config.CLIPS, config.FRAMES, config.CROPS, config.RUNS):
    d.mkdir(parents=True, exist_ok=True)

from PIL import Image, ImageDraw  # noqa: E402

from lastseen import db, query, render, state, windows  # noqa: E402
from lastseen.ingest import ingest_clip  # noqa: E402
from tests.mock_client import MockClient  # noqa: E402

W, H = 640, 360


def make_clip(path: Path, layout: dict[str, tuple[int, int]], nframes: int = 120, fps: int = 30) -> None:
    """Objects at fixed positions; the camera 'pans' by shifting everything slowly."""
    tmp = path.parent / (path.stem + "_frames")
    tmp.mkdir(parents=True, exist_ok=True)
    cols = {"red mug": (220, 30, 30), "blue charger": (30, 60, 220), "yellow book": (240, 210, 20), "grey charger": (70, 70, 75)}
    sizes = {"red mug": (60, 70), "blue charger": (60, 40), "yellow book": (90, 60), "grey charger": (60, 40)}
    for i in range(nframes):
        img = Image.new("RGB", (W, H), (236, 233, 226))
        d = ImageDraw.Draw(img)
        d.rectangle([20, 300, 620, 310], fill=(120, 90, 60))
        shift = int(i * 0.5)
        for name, (x, y) in layout.items():
            w, h = sizes[name]
            d.rectangle([x + shift, y, x + shift + w, y + h], fill=cols[name])
        img.save(tmp / f"{i:05d}.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
    shutil.rmtree(tmp)


class FakeEmbedder:
    """Deterministic 'embeddings' from colour histograms so re-ID has something to match on."""
    model_id, device, dim = "fake-colour", "cpu", 48

    def _vec(self, img):
        import numpy as np
        a = np.asarray(img.convert("RGB").resize((32, 32))).reshape(-1, 3)
        hist = np.concatenate([np.histogram(a[:, c], bins=16, range=(0, 255))[0] for c in range(3)]).astype("float32")
        return hist / (np.linalg.norm(hist) + 1e-8)

    def embed_images(self, images, masks=None, batch=16):
        import numpy as np
        return np.stack([self._vec(im) for im in images])

    def embed_obj_patch(self, image, mask):
        v = self._vec(image)
        return v, v


def main() -> None:
    con = db.connect(config.DB_PATH)
    client = MockClient(run_dir=config.RUNS / "mock")
    emb = FakeEmbedder()
    clips = TMP / "clips"
    clips.mkdir(exist_ok=True)
    c1 = clips / "morning.mp4"
    c2 = clips / "evening.mp4"
    make_clip(c1, {"red mug": (80, 200), "blue charger": (300, 240), "yellow book": (450, 220)})
    make_clip(c2, {"red mug": (420, 120), "blue charger": (300, 240), "grey charger": (120, 250)})
    # make the second clip look later
    now = time.time()
    os.utime(c1, (now - 30 * 3600, now - 30 * 3600))
    os.utime(c2, (now - 60, now - 60))

    client.room = "kitchen"
    r1 = ingest_clip(c1, "morning", client=client, embedder=emb, con=con, stride_sec=1.0)
    client.room = "kitchen"
    r2 = ingest_clip(c2, "evening", client=client, embedder=emb, con=con, stride_sec=1.0)
    print("ingest 1:", r1.__dict__)
    print("ingest 2:", r2.__dict__)

    objs = db.rows(con, "SELECT o.id, o.canonical_label, o.state, p.room, p.surface FROM objects o LEFT JOIN places p ON p.id=o.last_place_id")
    for o in objs:
        print("object", dict(o))
    evs = db.rows(con, "SELECT object_id, session_id, kind, note FROM object_events ORDER BY id")
    for e in evs:
        print("event", dict(e))

    # expectations
    labels = {o["canonical_label"]: o["state"] for o in objs}
    assert "red mug" in labels and "blue charger" in labels and "yellow book" in labels and "grey charger" in labels, labels
    assert labels["blue charger"] in ("stationary",), labels
    assert labels["grey charger"] == "new", labels
    assert labels["yellow book"] == "disappeared", labels
    # the mug moved within the kitchen counter -> same place name -> stationary (place is name-based). Document this.
    print("mug state:", labels["red mug"])

    res = query.answer(con, None, "where did I leave my blue charger?")
    print("ASK:", res["text"], res["card"] and (res["card"]["label"], res["card"]["place"], res["card"]["when"]))
    assert res["card"] and res["card"]["label"] == "blue charger"
    png = render.overlay_png(con, res["card"]["sighting_id"])
    assert png.exists()
    res = query.answer(con, None, "what changed since this morning?")
    print("ASK:", res["text"], [(c["kind"], c["label"]) for c in res["changes"]])
    kinds = {(c["kind"], c["label"]) for c in res["changes"]}
    assert ("disappeared", "yellow book") in kinds and ("new", "grey charger") in kinds, kinds
    # window + overlay video (mock SAM returns 50 frames)
    w = windows.ensure_window(con, client, res["changes"][0].get("after_sighting") or res["changes"][0].get("before_sighting"))
    print("window:", w and dict(w))
    mp4 = render.overlay_video(con, w["id"])
    print("overlay video:", mp4, mp4 and mp4.stat().st_size)
    print("API calls:", len(client.calls))
    print("OK — temp dir", TMP)


if __name__ == "__main__":
    main()
