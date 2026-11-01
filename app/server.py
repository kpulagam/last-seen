"""Last Seen web app. Run:  uvicorn app.server:app --reload --port 8008   (from the project root)"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lastseen import api, config, db, render, windows  # noqa: E402
from lastseen.ingest import ingest_clip  # noqa: E402
from lastseen.meta_client import MetaClient  # noqa: E402

app = FastAPI(title="Last Seen")
STATIC = ROOT / "app" / "static"
UPLOADS = config.DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

_client: MetaClient | None = None
_embedder = None
_progress: dict[str, dict] = {}
_lock = threading.Lock()


def client() -> MetaClient:
    global _client
    if _client is None:
        _client = MetaClient(run_dir=config.RUNS / time.strftime("app-%Y%m%d-%H%M%S"))
    return _client


def embedder():
    global _embedder
    if _embedder is None:
        try:
            from lastseen.reid import Embedder
            _embedder = Embedder()
        except Exception as e:  # noqa: BLE001
            print("embedder unavailable:", e)
            _embedder = False
    return _embedder or None


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text()


# ------------------------------------------------------------------ clips
@app.post("/api/clips")
async def upload_clip(file: UploadFile = File(...), session: str = Form("")):
    dest = UPLOADS / f"{int(time.time())}_{Path(file.filename).name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    key = dest.name
    _progress[key] = {"name": file.filename, "status": "queued", "log": [], "started": time.time()}

    def run():
        def log(msg):
            _progress[key]["log"].append(str(msg))
            _progress[key]["status"] = "ingesting"
        try:
            con = db.connect()
            res = ingest_clip(dest, session or None, client=client(), embedder=embedder(), con=con, log=log)
            _progress[key].update({"status": "done", "result": res.__dict__})
        except Exception as e:  # noqa: BLE001
            _progress[key].update({"status": "error", "error": str(e)[:500]})
            raise

    threading.Thread(target=run, daemon=True).start()
    return {"key": key}


@app.get("/api/clips")
def list_clips():
    return api.clips(db.connect(), _progress)


# ------------------------------------------------------------------- ask
class Ask(BaseModel):
    q: str
    offline: bool = False


@app.post("/api/ask")
def ask(body: Ask):
    return api.ask(db.connect(), None if body.offline else client(), body.q)


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Muse Voice Transcribe: the browser records a WAV (mono 16 kHz PCM) and we bias recognition
    toward the names of the objects this home knows."""
    wav = await audio.read()
    con = db.connect()
    names = [o["label"] for o in api.objects(con)["objects"] if o.get("label")]
    try:
        text = client().transcribe(wav, keywords=names[:60])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"transcription failed: {str(e)[:200]}")
    return {"text": text}


@app.post("/api/window/{sighting_id}")
def make_window(sighting_id: int):
    con = db.connect()
    w = windows.ensure_window(con, client(), sighting_id)
    if not w or w["status"] != "done":
        raise HTTPException(502, f"window failed: {w and w['status']}")
    render.overlay_video(con, w["id"])
    return {"url": f"media/window/{sighting_id}.mp4", "frames": [w["start_frame"], w["end_frame"]]}


@app.get("/api/objects")
def objects():
    return api.objects(db.connect())


@app.get("/api/objects/{object_id}/sightings")
def object_sightings(object_id: int):
    return api.object_sightings(db.connect(), object_id)


@app.get("/api/changes")
def changes(frm: str = "today_start", to: str = "now", session: int | None = None):
    return api.changes(db.connect(), frm, to, session)


@app.get("/api/pending")
def pending():
    return api.pending(db.connect())


class Merge(BaseModel):
    pending_id: int
    same: bool


@app.post("/api/merge")
def merge(body: Merge):
    res = api.merge(db.connect(), body.pending_id, body.same)
    if not res.get("ok"):
        raise HTTPException(404, res.get("error", ""))
    return res


# ------------------------------------------------------------------ media
@app.get("/media/crops/{name}")
def media_crop(name: str):
    p = config.local_path(config.CROPS / name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/media/overlay/{sighting_id}.jpg")
def media_overlay(sighting_id: int):
    con = db.connect()
    try:
        return FileResponse(render.overlay_png(con, sighting_id))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(404, str(e))


@app.get("/api/board")
def board():
    return api.board(db.connect())


@app.get("/media/cutout/{sighting_id}.png")
def media_cutout(sighting_id: int):
    p = api.cutout_path(db.connect(), sighting_id)
    if not p or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png")


@app.get("/media/frame/{sighting_id}.jpg")
def media_frame(sighting_id: int):
    p = api.frame_path(db.connect(), sighting_id)
    if not p or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/media/window/{sighting_id}.mp4")
def media_window(sighting_id: int):
    p = api.window_path(sighting_id)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
# the exported static demo (scripts/export_static.py docs/demo), served for a local check before publishing
if (ROOT / "docs" / "demo").is_dir():
    app.mount("/demo", StaticFiles(directory=str(ROOT / "docs" / "demo"), html=True), name="demo")
