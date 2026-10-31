"""Configuration: paths, API key, model names. Reads .env from the project root if present."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ["LASTSEEN_DATA"]) if os.environ.get("LASTSEEN_DATA") else ROOT / "data"   # env: relocate all data
CLIPS = DATA / "clips"      # original + faststart-remuxed clips
FRAMES = DATA / "frames"    # extracted keyframes  <clip_id>/<frame_idx>.jpg
CROPS = DATA / "crops"      # object crops          <sighting_id>.png
RUNS = ROOT / "runs"        # per-run logs and raw API responses
DB_PATH = Path(os.environ["LASTSEEN_DB"]) if os.environ.get("LASTSEEN_DB") else DATA / "lastseen.sqlite"

for _d in (DATA, CLIPS, FRAMES, CROPS, RUNS):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv(ROOT / ".env")

BASE_URL = os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")
API_KEY = os.environ.get("MODEL_API_KEY", "")

SPARK_MODEL = os.environ.get("SPARK_MODEL", "muse-spark-1.3")
SAM_MODEL = os.environ.get("SAM_MODEL", "sam-3.1")

# Ingestion knobs
KEYFRAME_STRIDE_SEC = float(os.environ.get("KEYFRAME_STRIDE_SEC", "2.0"))
MAX_TRACKED_OBJECTS = 16          # SAM 3.1 hard limit per video request
KEYFRAME_MAX_DIM = int(os.environ.get("KEYFRAME_MAX_DIM", "1024"))  # resize for Spark to control tokens


def local_path(p: str | Path | None) -> Path | None:
    """Resolve a stored absolute path when the data folder has moved (another machine, a mounted
    copy): fall back to the same file name under this install's data folders."""
    if not p:
        return None
    q = Path(p)
    if q.exists():
        return q
    name = q.name
    for root in (CROPS, FRAMES, CLIPS, CLIPS / "windows"):
        cand = root / name
        if cand.exists():
            return cand
    if q.parent.name and (FRAMES / q.parent.name / name).exists():   # frames/<clip_id>/<n>.jpg
        return FRAMES / q.parent.name / name
    return q


def require_api_key() -> str:
    if not API_KEY:
        raise SystemExit(
            "MODEL_API_KEY is not set. Create a .env file in the project root with\n"
            "MODEL_API_KEY=<your key from dev.meta.ai>\n"
        )
    return API_KEY
