"""Muse Voice Transcribe round trip without a microphone (macOS): synthesize a question with `say`,
convert to mono 16 kHz WAV with ffmpeg, send it to /v1/asr/transcribe, print the transcript.

    python scripts/voice_test.py "where did I leave my massage gun"
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lastseen import config  # noqa: E402
from lastseen.meta_client import MetaClient  # noqa: E402


def main() -> None:
    text = " ".join(sys.argv[1:]) or "where did I leave my massage gun"
    tmp = Path(tempfile.mkdtemp())
    aiff, wav = tmp / "q.aiff", tmp / "q.wav"
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)], check=True)
    client = MetaClient(run_dir=config.RUNS / "voice-test")
    out = client.transcribe(wav.read_bytes(), keywords=["massage gun", "scissors", "Last Seen"])
    print(f"said:  {text}\nheard: {out}")


if __name__ == "__main__":
    main()
