"""Video utilities built on ffmpeg/ffprobe (must be on PATH)."""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_FFMPEG_MAJOR: int | None = None


def ffmpeg_major() -> int:
    """Major version of the ffmpeg on PATH (4.4 on older Linux, 7 or 8 on current Homebrew)."""
    global _FFMPEG_MAJOR
    if _FFMPEG_MAJOR is None:
        try:
            out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout
            m = re.search(r"ffmpeg version n?(\d+)", out)
            _FFMPEG_MAJOR = int(m.group(1)) if m else 4
        except Exception:  # noqa: BLE001
            _FFMPEG_MAJOR = 4
    return _FFMPEG_MAJOR


def vsync_args(mode: str) -> list[str]:
    """`-vsync` became `-fps_mode` in ffmpeg 5.1 and was removed in 8. mode: 'passthrough' | 'cfr'."""
    if ffmpeg_major() >= 5:
        return ["-fps_mode", mode]
    return ["-vsync", "0" if mode == "passthrough" else mode]


@dataclass
class VideoInfo:
    path: Path
    width: int          # display width after rotation
    height: int         # display height after rotation
    fps: float
    nframes: int
    duration: float
    rotation: int = 0
    codec: str = ""

    def frame_at(self, t: float) -> int:
        return max(0, min(self.nframes - 1, int(round(t * self.fps))))

    def time_at(self, frame: int) -> float:
        return frame / self.fps if self.fps else 0.0


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({p.returncode}): {' '.join(cmd)}\n{p.stderr[-2000:]}")
    return p


def probe(path: str | Path) -> VideoInfo:
    path = Path(path)
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,codec_name,side_data_list:stream_tags=rotate:format=duration",
                "-of", "json", str(path)]).stdout
    data = json.loads(out)
    st = data["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    fr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1"
    num, den = fr.split("/") if "/" in fr else (fr, "1")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(st.get("duration") or data.get("format", {}).get("duration") or 0.0)
    nframes = int(st.get("nb_frames") or 0)
    if not nframes:
        nframes = int(round(duration * fps))
    rotation = 0
    tags = st.get("tags") or {}
    if tags.get("rotate"):
        rotation = int(float(tags["rotate"])) % 360
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(float(sd["rotation"])) % 360
    if rotation in (90, 270):
        w, h = h, w
    return VideoInfo(path=path, width=w, height=h, fps=fps, nframes=nframes, duration=duration,
                     rotation=rotation, codec=st.get("codec_name", ""))


def recording_start(path: str | Path, duration: float | None = None) -> float:
    """Epoch seconds when the clip STARTED recording. Prefers the camera's own timestamp
    (iPhone: com.apple.quicktime.creationdate, with its local offset; else creation_time, UTC),
    which survives AirDrop and copies; falls back to the file's mtime minus the duration."""
    import datetime as _dt
    path = Path(path)
    dur = float(duration or 0.0)
    try:
        out = _run(["ffprobe", "-v", "error", "-show_entries",
                    "format_tags=com.apple.quicktime.creationdate,creation_time:format=duration",
                    "-of", "json", str(path)]).stdout
        data = json.loads(out)
        fmt = data.get("format", {})
        tags = fmt.get("tags") or {}
        dur = float(duration if duration is not None else (fmt.get("duration") or 0.0))
        raw = tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time")
        if raw:
            raw = raw.strip().replace("Z", "+00:00")
            m = re.match(r"^(.*?)([+-]\d{2}):?(\d{2})$", raw)       # normalise -0500 -> -05:00 (py3.10)
            if m and "T" in raw:
                raw = f"{m.group(1)}{m.group(2)}:{m.group(3)}"
            elif "T" in raw and not re.search(r"[+-]\d{2}:\d{2}$", raw):
                raw = raw + "+00:00"
            if "." in raw:                                          # drop fractional seconds
                raw = re.sub(r"\.\d+", "", raw)
            t = _dt.datetime.fromisoformat(raw)
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.timezone.utc)
            start = t.timestamp()
            # Apple writes the START of the recording; generic creation_time is usually the end (mtime-like)
            if "com.apple.quicktime.creationdate" not in tags:
                start -= dur
            return start
    except Exception:  # noqa: BLE001
        pass
    return path.stat().st_mtime - dur


def remux_faststart(src: str | Path, dst: str | Path) -> Path:
    """Lossless remux with the moov atom at the front (SAM uploads prefer this)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)])
    return dst


def transcode_for_api(src: str | Path, dst: str | Path, max_dim: int = 1280, fps: float | None = None,
                      crf: int = 23) -> Path:
    """Re-encode to H.264 MP4 (yuv420p, faststart), scaled so the long edge <= max_dim.

    Phones often record HEVC 4K/60; a 1080p-or-smaller H.264 clip is cheaper to upload and
    keeps frame indices manageable. Rotation metadata is baked in by ffmpeg's autorotate.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale='if(gt(iw,ih),min({max_dim},iw),-2)':'if(gt(iw,ih),-2,min({max_dim},ih))'"
    # Phones record variable frame rate (iPhone: 30 fps nominal, 120 fps timebase); force constant
    # 30 fps so frame indices are stable between ffprobe, extraction and the API.
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf, "-r", str(fps or 30), *vsync_args("cfr"),
           "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-an", "-movflags", "+faststart", str(dst)]
    _run(cmd)
    return dst


def extract_frame(video: str | Path, frame_idx: int, out: str | Path, max_dim: int | None = None) -> Path:
    """Write one exact frame (by index) as JPEG."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = f"select=eq(n\\,{frame_idx})"
    if max_dim:
        vf += f",scale='if(gt(iw,ih),min({max_dim},iw),-2)':'if(gt(iw,ih),-2,min({max_dim},ih))'"
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", vf, *vsync_args("passthrough"), "-frames:v", "1",
          "-q:v", "2", str(out)])
    return out


def extract_frames(video: str | Path, frame_idxs: list[int], out_dir: str | Path,
                   max_dim: int | None = None) -> dict[int, Path]:
    """Extract several exact frames in one ffmpeg pass. Returns {frame_idx: path}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not frame_idxs:
        return {}
    idxs = sorted(set(int(i) for i in frame_idxs))
    result: dict[int, Path] = {}
    # ffmpeg's expression parser gives up ("Cannot allocate memory") past ~100 eq() terms, so a long
    # clip is pulled in chunks; each chunk is one decode pass over the file.
    chunk = 40
    for start in range(0, len(idxs), chunk):
        part = idxs[start:start + chunk]
        sel = "+".join(f"eq(n\\,{i})" for i in part)
        vf = f"select='{sel}'"
        if max_dim:
            vf += f",scale='if(gt(iw,ih),min({max_dim},iw),-2)':'if(gt(iw,ih),-2,min({max_dim},ih))'"
        pattern = str(out_dir / "_tmp_%05d.jpg")
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", vf, *vsync_args("passthrough"), "-q:v", "2", pattern])
        for k, idx in enumerate(part, start=1):
            tmp = out_dir / f"_tmp_{k:05d}.jpg"
            final = out_dir / f"{idx:06d}.jpg"
            if tmp.exists():
                tmp.replace(final)
                result[idx] = final
    return result


def scene_change_frames(video: str | Path, threshold: float = 0.3) -> list[int]:
    """Frame indices where ffmpeg's scene detector fires (handheld walk-throughs: expect few)."""
    p = _run(["ffmpeg", "-v", "info", "-i", str(video), "-vf",
              f"select='gt(scene,{threshold})',showinfo", *vsync_args("passthrough"), "-f", "null", "-"], check=False)
    frames: list[int] = []
    for line in p.stderr.splitlines():
        if "showinfo" in line and " n:" in line:
            try:
                n = int(line.split(" n:")[1].split()[0])
                frames.append(n)
            except (IndexError, ValueError):
                pass
    return frames


def keyframe_indices(info: VideoInfo, stride_sec: float = 2.0, scene_threshold: float | None = 0.3) -> list[int]:
    """Fixed-stride keyframes, plus scene-change frames (if enabled), deduped and sorted."""
    step = max(1, int(round(stride_sec * info.fps)))
    idxs = set(range(0, max(1, info.nframes), step))
    idxs.add(max(0, info.nframes - 1))
    if scene_threshold is not None:
        for f in scene_change_frames(info.path, scene_threshold):
            idxs.add(min(f, info.nframes - 1))
    return sorted(i for i in idxs if 0 <= i < max(1, info.nframes))


def cut_clip(video: str | Path, start_s: float, end_s: float, out: str | Path) -> Path:
    """Re-encoded excerpt [start_s, end_s] (re-encode so the cut is frame-accurate)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, end_s - start_s)
    _run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-i", str(video), "-t", f"{dur:.3f}",
          "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
          "-movflags", "+faststart", str(out)])
    return out
