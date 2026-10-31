"""Thin client for Meta Model API (https://api.meta.ai/v1).

Only `requests` is required. Every call is logged to runs/<run_id>/api/NNN_<kind>.json
(request without media payloads, response headers, timing, usage) so results can be
inspected after the fact.

Endpoints used:
  POST /v1/files                 (purpose=user_data)  -> file_id
  POST /v1/responses             model=sam-3.1        -> special-token text (boxes+masks)
  POST /v1/chat/completions      model=muse-spark-1.3 -> text / structured JSON / tool calls
"""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import requests

from . import config


class MetaAPIError(RuntimeError):
    def __init__(self, status: int, body: str, request_summary: dict | None = None):
        super().__init__(f"HTTP {status}: {body[:800]}")
        self.status = status
        self.body = body
        self.request_summary = request_summary or {}


@dataclass
class CallLog:
    kind: str
    request: dict
    status: int
    elapsed_s: float
    usage: dict | None = None
    response_excerpt: str = ""
    response_path: str = ""


@dataclass
class MetaClient:
    api_key: str = field(default_factory=config.require_api_key)
    base_url: str = config.BASE_URL
    run_dir: Path | None = None
    timeout_s: float = 600.0
    max_retries: int = 4
    calls: list[CallLog] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ----------------------------------------------------------------- utils
    def _headers(self, json_body: bool = True) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}"}
        if json_body:
            h["Content-Type"] = "application/json"
        return h

    def _log(self, kind: str, request: dict, status: int, elapsed: float,
             usage: dict | None, response_text: str) -> None:
        entry = CallLog(kind=kind, request=request, status=status, elapsed_s=round(elapsed, 2),
                        usage=usage, response_excerpt=response_text[:400])
        with self._lock:
            n = len(self.calls)
            self.calls.append(entry)
            if self.run_dir is not None:
                api_dir = self.run_dir / "api"
                api_dir.mkdir(parents=True, exist_ok=True)
                p = api_dir / f"{n:03d}_{kind}.json"
                p.write_text(json.dumps({
                    "kind": kind, "request": request, "status": status, "elapsed_s": entry.elapsed_s,
                    "usage": usage, "response": response_text,
                }, indent=1))
                entry.response_path = str(p)

    # ------------------------------------------------------------------ cache
    # Identical requests (same model, prompt, images, schema) are answered from data/api_cache so
    # re-running ingest to tune re-ID costs nothing. Set LASTSEEN_NO_CACHE=1 to bypass.
    @staticmethod
    def _cache_path(path: str, body: dict) -> Path | None:
        import os
        if os.environ.get("LASTSEEN_NO_CACHE"):
            return None
        key = hashlib.sha1((path + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        d = config.DATA / "api_cache"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json"

    @staticmethod
    def _cache_get(p: Path | None) -> dict | None:
        if p is None or not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _cache_put(p: Path | None, value: dict) -> None:
        if p is not None:
            p.write_text(json.dumps(value))

    @staticmethod
    def _summarize(body: dict) -> dict:
        """Copy of the request body with media payloads replaced by a size marker."""
        def strip(x: Any) -> Any:
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    if k in ("image_url", "video_url", "url") and isinstance(v, str) and v.startswith("data:"):
                        out[k] = f"<data-url {len(v)} chars>"
                    else:
                        out[k] = strip(v)
                return out
            if isinstance(x, list):
                return [strip(i) for i in x]
            return x
        return strip(body)

    def _post(self, path: str, body: dict, kind: str, stream: bool = False) -> requests.Response:
        url = f"{self.base_url}{path}"
        summary = self._summarize(body)
        delay = 0.5
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            r = requests.post(url, headers=self._headers(), data=json.dumps(body),
                              timeout=self.timeout_s, stream=stream)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                time.sleep(wait + random.random() * 0.3)
                delay = min(delay * 2, 20)
                continue
            if r.status_code >= 400:
                self._log(kind, summary, r.status_code, time.time() - t0, None, r.text)
                raise MetaAPIError(r.status_code, r.text, summary)
            return r
        raise MetaAPIError(599, "retries exhausted", summary)

    # ----------------------------------------------------------------- files
    def upload_file(self, path: str | Path, purpose: str = "user_data") -> str:
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        t0 = time.time()
        with open(path, "rb") as f:
            r = requests.post(f"{self.base_url}/files", headers=self._headers(json_body=False),
                              files={"file": (path.name, f, mime)}, data={"purpose": purpose},
                              timeout=self.timeout_s)
        if r.status_code >= 400:
            self._log("files.create", {"filename": path.name}, r.status_code, time.time() - t0, None, r.text)
            raise MetaAPIError(r.status_code, r.text)
        data = r.json()
        self._log("files.create", {"filename": path.name, "bytes": path.stat().st_size},
                  r.status_code, time.time() - t0, None, json.dumps(data))
        return data["id"]

    # ------------------------------------------------------------- voice
    def transcribe(self, wav_bytes: bytes, keywords: list[str] | None = None,
                   model: str = "muse-voice-transcribe-1.0") -> str:
        """Muse Voice Transcribe, one complete recording (RIFF/WAVE, mono 16-bit PCM, 16 or 24 kHz,
        <= 10 min). PUSH_TO_TALK: the clip is the turn. `keywords` bias recognition toward the
        household's own object names ("massage gun", "Stanley cup")."""
        req = {"model": model, "audioEncoding": "WAV", "mode": "PUSH_TO_TALK", "languageBias": ["English"]}
        if keywords:
            req["keywords"] = [k for k in keywords if k][:100]
        t0 = time.time()
        r = requests.post(f"{self.base_url}/asr/transcribe", headers={**self._headers(json_body=False), "Accept": "application/json"},
                          files={"request": (None, json.dumps(req), "application/json"),
                                 "audio": ("question.wav", wav_bytes, "audio/wav")}, timeout=self.timeout_s)
        summary = {"model": model, "bytes": len(wav_bytes), "keywords": len(req.get("keywords", []))}
        if r.status_code >= 400:
            self._log("voice_transcribe", summary, r.status_code, time.time() - t0, None, r.text)
            raise MetaAPIError(r.status_code, r.text, summary)
        data = r.json()
        self._log("voice_transcribe", summary, r.status_code, time.time() - t0,
                  {"audio_ms": data.get("audioDurationMs")}, json.dumps(data))
        return (data.get("transcript") or "").strip()

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/models", headers=self._headers(json_body=False), timeout=60)
        if r.status_code >= 400:
            raise MetaAPIError(r.status_code, r.text)
        data = r.json()
        return [m.get("id") for m in data.get("data", [])]

    # ------------------------------------------------------------------- SAM
    def sam_responses(self, prompt_text: str, *, image_data_url: str | None = None,
                      image_url: str | None = None, video_file_id: str | None = None,
                      video_url: str | None = None, mask_encoding: str = "one_bit",
                      include_confidence: bool = True, propagation_direction: str | None = None,
                      stream: bool = True, kind: str = "sam") -> str:
        """Call sam-3.1 and return the complete output_text (special-token grammar).

        Exactly one media argument must be given. Streaming is used by default so
        long video responses arrive frame by frame; the deltas are concatenated.
        """
        media: dict
        if image_data_url or image_url:
            media = {"type": "input_image", "image_url": image_data_url or image_url}
        elif video_file_id:
            media = {"type": "input_video", "file_id": video_file_id}
        elif video_url:
            media = {"type": "input_video", "video_url": video_url}
        else:
            raise ValueError("one of image_data_url/image_url/video_file_id/video_url required")

        metadata: dict[str, str] = {"mask_encoding": mask_encoding,
                                    "include_confidence": "true" if include_confidence else "false"}
        if propagation_direction:
            metadata["propagation_direction"] = propagation_direction
        body = {
            "model": config.SAM_MODEL,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": prompt_text}, media]}],
            "metadata": metadata,
            "stream": bool(stream),
        }
        t0 = time.time()
        cp = self._cache_path("/responses", body)
        hit = self._cache_get(cp)
        if hit is not None:
            self._log(kind + "_cached", self._summarize(body) | {"_prompt_len": len(prompt_text)}, 200,
                      0.0, hit.get("usage"), hit.get("text", ""))
            return hit.get("text", "")
        if stream:
            r = self._post("/responses", body, kind, stream=True)
            text_parts: list[str] = []
            usage = None
            for ev in _iter_sse(r):
                et = ev.get("type")
                if et == "response.output_text.delta":
                    text_parts.append(ev.get("delta", ""))
                elif et == "response.completed":
                    usage = (ev.get("response") or {}).get("usage")
                elif et in ("response.failed", "error"):
                    raise MetaAPIError(500, json.dumps(ev)[:800], self._summarize(body))
            text = "".join(text_parts)
        else:
            r = self._post("/responses", body, kind, stream=False)
            data = r.json()
            usage = data.get("usage")
            text = _responses_output_text(data)
        self._log(kind, self._summarize(body) | {"_prompt_len": len(prompt_text)}, 200,
                  time.time() - t0, usage, text)
        self._cache_put(cp, {"text": text, "usage": usage})
        return text

    # ----------------------------------------------------------------- Spark
    def spark_chat(self, messages: list[dict], *, json_schema: dict | None = None,
                   tools: list[dict] | None = None, tool_choice: Any = None,
                   max_tokens: int = 4000, temperature: float | None = None,
                   reasoning_effort: str | None = None, kind: str = "spark") -> dict:
        """Chat Completions call on Muse Spark. Returns the raw response JSON."""
        body: dict[str, Any] = {"model": config.SPARK_MODEL, "messages": messages, "max_tokens": max_tokens}
        if json_schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if temperature is not None:
            body["temperature"] = temperature
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        t0 = time.time()
        cp = self._cache_path("/chat/completions", body)
        hit = self._cache_get(cp)
        if hit is not None:
            self._log(kind + "_cached", self._summarize(body), 200, 0.0, hit.get("usage"),
                      json.dumps(hit.get("choices", [{}])[0].get("message", {}))[:4000])
            return hit
        try:
            r = self._post("/chat/completions", body, kind)
        except MetaAPIError as e:
            # Some deployments reject reasoning_effort on chat completions; retry without it once.
            if e.status == 400 and "reasoning_effort" in body:
                body.pop("reasoning_effort")
                r = self._post("/chat/completions", body, kind)
            else:
                raise
        data = r.json()
        self._log(kind, self._summarize(body), r.status_code, time.time() - t0,
                  data.get("usage"), json.dumps(data.get("choices", [{}])[0].get("message", {}))[:4000])
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        if msg.get("content") and finish != "length":
            self._cache_put(cp, data)
        return data

    def spark_json(self, prompt: str, *, images: list[str] | None = None, schema: dict,
                   schema_name: str = "Result", system: str | None = None,
                   max_tokens: int = 4000, reasoning_effort: str | None = "low", kind: str = "spark_json") -> dict:
        """Ask Spark a question with optional images (data URLs) and get schema-constrained JSON.

        Muse Spark is a reasoning model and reasoning shares the output budget: measured ~1,100-1,500
        reasoning tokens per structured answer, so max_tokens must stay >= 4000 and effort low.
        """
        content: list[dict] = [{"type": "text", "text": prompt}]
        for url in images or []:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        data = self.spark_chat(messages, json_schema={"name": schema_name, "schema": schema},
                               max_tokens=max(max_tokens, 4000), reasoning_effort=reasoning_effort, kind=kind)
        choice = data["choices"][0]
        msg = choice["message"]
        txt = msg.get("content") or ""
        if not txt and choice.get("finish_reason") == "length":
            raise MetaAPIError(200, f"output budget exhausted by reasoning (usage={data.get('usage')})")
        try:
            return json.loads(txt)
        except json.JSONDecodeError as e:
            raise MetaAPIError(200, f"non-JSON content from structured output: {txt[:300]}") from e

    def spark_text(self, prompt: str, *, images: list[str] | None = None, system: str | None = None,
                   max_tokens: int = 2000, kind: str = "spark_text") -> str:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for url in images or []:
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        data = self.spark_chat(messages, max_tokens=max_tokens, kind=kind)
        return data["choices"][0]["message"].get("content") or ""


# --------------------------------------------------------------------- helpers
def image_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def bytes_to_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def _iter_sse(r: requests.Response) -> Iterator[dict]:
    """Yield parsed JSON events from a server-sent-events response."""
    buf: list[str] = []
    for raw in r.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip("\r")
        if line == "":
            if buf:
                payload = "\n".join(buf)
                buf = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
        # ignore "event:" / "id:" lines; the JSON carries "type"
    if buf:
        payload = "\n".join(buf)
        if payload.strip() != "[DONE]":
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                pass


def _responses_output_text(data: dict) -> str:
    """Extract concatenated output_text from a non-streaming Responses API result."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    parts: list[str] = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "".join(parts)
