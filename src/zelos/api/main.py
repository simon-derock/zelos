"""Zelos HTTP API (FastAPI). Same tools as the CLI/MCP server.

GET  /api/health                         liveness, version, whether the retriever is loaded
POST /api/inspect   (multipart file)     QC certificate (0 LLM calls)
POST /api/explain   (multipart file)     gated explanation + ledger (provider=template|workers_ai); also returns `qc`
POST /api/triage    (multipart files)    per-image verdicts against a batch reference (4 to 12 images)
POST /api/ask       {"question": ""}     cited answer or ABSTAIN
Run: uvicorn zelos.api.main:app --port 8000

Public-demo hardening (all configurable by environment):
  ZELOS_ENV                 dev | prod (default prod). CORS defaults to "*" only in dev.
  ZELOS_CORS_ORIGINS        comma-separated allowed origins (overrides the defaults)
  ZELOS_RATE_PER_MIN        requests per client per minute (default 20; a triage costs one per image)
  ZELOS_RATE_GLOBAL_PER_MIN overall ceiling per minute (default 10 x the per-client limit)
  ZELOS_PROXY_HOPS          trusted reverse proxies in front of the app (default 1 in prod, 0 in dev)
Uploads are held in memory only, validated by magic bytes and header (PNG/JPEG/TIFF, <= 25 MB,
<= 40 megapixels) and never written to disk or logged.
"""
from __future__ import annotations

import os
import struct
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import anyio
import anyio.to_thread
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import zelos.common.config  # noqa: F401  (loads .env before the environment is read below)

MAX_BYTES = 25 * 1024 * 1024  # per image
MAX_PIXELS = 40_000_000  # decoded pixels per image (decompression-bomb guard)
MIN_SIDE = 16
TRIAGE_MIN, TRIAGE_MAX = 4, 12
MAX_TRIAGE_BYTES = 120 * 1024 * 1024  # aggregate per triage request
FORM_OVERHEAD = 1024 * 1024
PROD_ORIGINS = ["https://zelos.philipsimonderock.com", "https://papaya-figolla-a9dd1f.netlify.app"]
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
VERSION = "1.0.0"


# ----------------------------------------------------------------------------- upload validation
def _png_dims(b: bytes) -> tuple[int, int] | None:
    if len(b) >= 24 and b[12:16] == b"IHDR":
        w, h = struct.unpack(">II", b[16:24])
        return w, h
    return None


def _jpeg_dims(b: bytes) -> tuple[int, int] | None:
    i, n = 2, len(b)
    while i + 4 <= n:
        if b[i] != 0xFF:
            i += 1
            continue
        m = b[i + 1]
        if m == 0xFF:  # fill byte
            i += 1
            continue
        if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xD9, 0xDA):  # EOI / start of scan before any frame header
            return None
        seg = struct.unpack(">H", b[i + 2 : i + 4])[0]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                return None
            h, w = struct.unpack(">HH", b[i + 5 : i + 9])
            return w, h
        i += 2 + seg
    return None


def _tiff_dims(b: bytes) -> tuple[int, int] | None:
    e = "<" if b[:2] == b"II" else ">"
    big = struct.unpack(e + "H", b[2:4])[0] == 43
    try:
        if big:
            off = struct.unpack(e + "Q", b[8:16])[0]
            n = struct.unpack(e + "Q", b[off : off + 8])[0]
            base, size = off + 8, 20
        else:
            off = struct.unpack(e + "I", b[4:8])[0]
            n = struct.unpack(e + "H", b[off : off + 2])[0]
            base, size = off + 2, 12
        if n > 4096:
            return None
        dims: dict[int, int] = {}
        for k in range(n):
            ent = b[base + k * size : base + (k + 1) * size]
            if len(ent) < size:
                return None
            tag, typ = struct.unpack(e + "HH", ent[:4])
            if tag not in (256, 257):
                continue
            vo = 12 if big else 8
            if typ == 3:
                dims[tag] = struct.unpack(e + "H", ent[vo : vo + 2])[0]
            elif typ == 4:
                dims[tag] = struct.unpack(e + "I", ent[vo : vo + 4])[0]
            elif typ == 16 and big:
                dims[tag] = struct.unpack(e + "Q", ent[vo : vo + 8])[0]
        if 256 in dims and 257 in dims:
            return dims[256], dims[257]
    except struct.error:
        return None
    return None


def image_dims(data: bytes) -> tuple[int, int]:
    """(width, height) read from the container header, without decoding. Raises ValueError if not PNG/JPEG/TIFF."""
    d: tuple[int, int] | None = None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        d = _png_dims(data)
    elif data[:3] == b"\xff\xd8\xff":
        d = _jpeg_dims(data)
    elif data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        d = _tiff_dims(data)
    else:
        raise ValueError("unsupported file type: upload a PNG, JPEG or TIFF image")
    if d is None or d[0] <= 0 or d[1] <= 0:
        raise ValueError("could not read the image header; the file may be corrupt")
    return d


def validate_image_bytes(data: bytes) -> None:
    w, h = image_dims(data)
    if min(w, h) < MIN_SIDE:
        raise ValueError(f"image too small ({w}x{h}); minimum side is {MIN_SIDE} px")
    if w * h > MAX_PIXELS:
        raise ValueError(f"image too large ({w * h / 1e6:.0f} megapixels); the limit is {MAX_PIXELS // 1_000_000} MP")


async def read_capped(file: UploadFile, cap: int | None = None) -> bytes:
    """Read at most `cap` bytes; never trusts Content-Length."""
    cap = MAX_BYTES if cap is None else cap
    chunks, total = [], 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            raise HTTPException(413, f"image too large (limit {cap // (1024 * 1024)} MB)")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(400, "empty upload")
    return data


def _safe_id(name: str | None, fallback: str) -> str:
    base = os.path.basename((name or "").replace("\\", "/")).rsplit(".", 1)[0]
    base = "".join(ch for ch in base if ch.isprintable())[:80].strip()
    return base or fallback


# ----------------------------------------------------------------------------- rate limiting
class SlidingWindow:
    """In-memory sliding-window counter keyed by client. Thread-safe."""

    def __init__(self, limit: int, window: float = 60.0, max_keys: int = 20000):
        self.limit, self.window, self.max_keys = max(1, int(limit)), float(window), max_keys
        self._hits: dict[str, deque[tuple[float, int]]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, cost: int = 1, now: float | None = None) -> float | None:
        """Record `cost` units. Returns None if allowed, else the seconds until a retry can succeed."""
        now = time.monotonic() if now is None else now
        cost = max(1, min(int(cost), self.limit))
        with self._lock:
            dq = self._hits.get(key)
            if dq is None:
                if len(self._hits) >= self.max_keys:
                    self._prune(now)
                dq = self._hits.setdefault(key, deque())
            while dq and dq[0][0] <= now - self.window:
                dq.popleft()
            used = sum(c for _, c in dq)
            if used + cost > self.limit:
                need = used + cost - self.limit
                freed = 0
                for t, c in dq:
                    freed += c
                    if freed >= need:
                        return max(0.1, t + self.window - now)
                return self.window
            dq.append((now, cost))
            return None

    def _prune(self, now: float) -> None:
        for k in [k for k, dq in self._hits.items() if not dq or dq[-1][0] <= now - self.window]:
            del self._hits[k]
        if len(self._hits) >= self.max_keys:  # still full: drop the oldest half
            for k in list(self._hits)[: self.max_keys // 2]:
                del self._hits[k]


# ----------------------------------------------------------------------------- body size limit (ASGI)
class _TooLarge(Exception):
    pass


class BodyLimit:
    """Aborts a request whose body exceeds a per-path cap, counting bytes as they arrive."""

    def __init__(self, app: Any, cap_for: Callable[[str], int]):
        self.app, self.cap_for = app, cap_for

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        cap = self.cap_for(scope["path"])
        started = False

        async def _send(msg: dict) -> None:
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        async def reject() -> None:
            body = JSONResponse({"detail": "request body too large"}, status_code=413)
            await body(scope, receive, send)

        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > cap:
                        await reject()
                        return
                except ValueError:
                    pass
        total = 0

        async def _receive() -> dict:
            nonlocal total
            msg = await receive()
            if msg["type"] == "http.request":
                total += len(msg.get("body", b""))
                if total > cap:
                    raise _TooLarge
            return msg

        try:
            await self.app(scope, _receive, _send)
        except _TooLarge:
            if not started:
                await reject()


# ----------------------------------------------------------------------------- app
class Ask(BaseModel):
    question: str
    provider: str = "template"


def _retriever_loader() -> Callable[[], Any]:
    lock = threading.Lock()
    state: dict[str, Any] = {"tried": False, "obj": None}

    def get() -> Any:
        with lock:
            if not state["tried"]:
                state["tried"] = True
                try:
                    from zelos.rag.retriever import HybridRetriever

                    state["obj"] = HybridRetriever.load()
                except Exception:  # noqa: BLE001 - retrieval is optional; the agent falls back to no evidence
                    state["obj"] = None
            return state["obj"]

    get.loaded = lambda: state["obj"] is not None  # type: ignore[attr-defined]
    return get


def _client_key(request: Request, hops: int) -> str:
    peer = request.client.host if request.client else "unknown"
    if hops > 0:
        xff = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(xff) >= hops:
            return xff[-hops]
    return peer


def create_app(env: dict[str, str] | None = None) -> FastAPI:
    E = os.environ if env is None else env
    mode = E.get("ZELOS_ENV", "prod").lower()
    dev = mode == "dev"
    origins_raw = E.get("ZELOS_CORS_ORIGINS", "").strip()
    if origins_raw:
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    else:
        origins = ["*"] if dev else list(PROD_ORIGINS)
    per_min = int(E.get("ZELOS_RATE_PER_MIN", "20"))
    global_min = int(E.get("ZELOS_RATE_GLOBAL_PER_MIN", str(per_min * 10)))
    hops = int(E.get("ZELOS_PROXY_HOPS", "0" if dev else "1"))
    client_limit = SlidingWindow(per_min)
    global_limit = SlidingWindow(global_min)
    retriever = _retriever_loader()
    limiter = anyio.CapacityLimiter(max(1, int(E.get("ZELOS_MAX_CONCURRENCY", "2"))))

    preload = E.get("ZELOS_PRELOAD", "0" if dev else "1") == "1"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if preload:  # warm the retriever off the request path so the first explanation is not slow
            threading.Thread(target=retriever, daemon=True).start()
        yield

    app = FastAPI(title="Zelos", version=VERSION, lifespan=lifespan,
                  description="Evidence-gated QC for organ-on-a-chip images. Research use only.")
    app.state.retriever = retriever
    app.state.cors_origins = origins

    def cap_for(path: str) -> int:
        if path.endswith("/triage"):
            return MAX_TRIAGE_BYTES + FORM_OVERHEAD
        if path.endswith("/ask"):
            return 64 * 1024
        return MAX_BYTES + FORM_OVERHEAD

    app.add_middleware(BodyLimit, cap_for=cap_for)
    wildcard = "*" in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=None if wildcard else LOCAL_ORIGIN_REGEX,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
        max_age=600,
    )

    def throttle(request: Request, cost: int = 1) -> None:
        wait = global_limit.hit("*", cost) or client_limit.hit(_client_key(request, hops), cost)
        if wait is not None:
            raise HTTPException(429, "rate limit reached, try again shortly",
                                headers={"Retry-After": str(max(1, int(wait + 0.999)))})

    async def work(fn: Callable[..., Any], *args: Any) -> Any:
        return await anyio.to_thread.run_sync(lambda: fn(*args), limiter=limiter)

    def qc_one(data: bytes, image_id: str) -> dict:
        from zelos.vision.pipeline import inspect

        try:
            return inspect(data, image_id=image_id)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # noqa: BLE001 - decoder errors surface as a client error, never a stack trace
            raise HTTPException(400, "could not decode the image") from e

    async def upload_qc(file: UploadFile) -> dict:
        data = await read_capped(file)
        try:
            validate_image_bytes(data)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return await work(qc_one, data, _safe_id(file.filename, "upload"))

    @app.get("/")
    def root() -> dict:
        return {"name": "Zelos", "docs": "/docs", "health": "/api/health",
                "note": "Research tool. Not a medical device. No clinical or diagnostic claims."}

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "version": VERSION, "env": mode, "retriever_loaded": bool(retriever.loaded()),
                "limits": {"max_mb": MAX_BYTES // (1024 * 1024), "max_megapixels": MAX_PIXELS // 1_000_000,
                           "triage_images": [TRIAGE_MIN, TRIAGE_MAX], "rate_per_min": per_min}}

    @app.post("/api/inspect")
    async def inspect_ep(request: Request, file: UploadFile = File(...)) -> dict:
        throttle(request)
        return await upload_qc(file)

    @app.post("/api/explain")
    async def explain_ep(request: Request, file: UploadFile = File(...), provider: str = "template") -> dict:
        from zelos.agent.loop import run_inspect

        if provider not in {"template", "workers_ai"}:
            raise HTTPException(400, "provider must be template or workers_ai")
        throttle(request)
        qc = await upload_qc(file)
        out = await work(lambda: run_inspect(qc, retriever=retriever(), provider=provider))
        out["qc"] = qc  # additive: lets a client render metrics and percentiles without a second call
        return out

    @app.post("/api/triage")
    async def triage_ep(request: Request, files: list[UploadFile] = File(...)) -> list[dict]:
        if len(files) < TRIAGE_MIN:
            raise HTTPException(400, f"session triage needs at least {TRIAGE_MIN} images so the batch reference "
                                     f"is meaningful (got {len(files)})")
        if len(files) > TRIAGE_MAX:
            raise HTTPException(400, f"at most {TRIAGE_MAX} images per session (got {len(files)})")
        throttle(request, cost=len(files))
        items: list[tuple[str, bytes]] = []
        total, seen = 0, set()
        for i, f in enumerate(files):
            data = await read_capped(f)
            total += len(data)
            if total > MAX_TRIAGE_BYTES:
                raise HTTPException(413, f"session too large (limit {MAX_TRIAGE_BYTES // (1024 * 1024)} MB in total)")
            try:
                validate_image_bytes(data)
            except ValueError as e:
                raise HTTPException(400, f"{_safe_id(f.filename, f'image {i + 1}')}: {e}") from e
            iid = _safe_id(f.filename, f"image-{i + 1}")
            n, base = 2, iid
            while iid in seen:
                iid, n = f"{base}-{n}", n + 1
            seen.add(iid)
            items.append((iid, data))
        return await work(batch_qc, items)

    @app.post("/api/ask")
    async def ask_ep(request: Request, body: Ask) -> dict:
        from zelos.agent.loop import run_ask

        if len(body.question) > 1000:
            raise HTTPException(413, "question too long")
        if body.provider not in {"template", "workers_ai"}:
            raise HTTPException(400, "provider must be template or workers_ai")
        throttle(request)
        return await work(lambda: run_ask(body.question, retriever=retriever(), provider=body.provider))

    return app


def batch_qc(items: list[tuple[str, bytes]]) -> list[dict]:
    """QC every image against a reference made of the batch itself (median/MAD of its own metrics).

    Pass 1 keeps only each image's metrics (one decoded image in memory at a time); pass 2 scores each image
    with the same pipeline entry point as /api/inspect, in batch mode, so verdicts, thresholds, causes and
    routes follow whatever the single-image path does.
    """
    import cv2
    import numpy as np
    import pandas as pd

    from zelos.common.config import load_config
    from zelos.vision.metrics import compute_metrics
    from zelos.vision.pipeline import inspect
    from zelos.vision.scoring import fit_reference

    max_side = load_config()["vision"]["max_side"]
    rows: list[dict] = []
    for iid, data in items:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(400, f"{iid}: could not decode the image")
        try:
            rows.append(compute_metrics(img, max_side))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"{iid}: could not analyse the image") from e
        del img
    ref = fit_reference(pd.DataFrame(rows))
    out = []
    for iid, data in items:
        q = inspect(data, image_id=iid, reference=ref, mode="batch")
        out.append({"id": iid, "level": q["level"], "qc_score": q["qc_score"], "causes": q["causes"],
                    "density_state": q["density_state"], "tool_call_id": q["tool_call_id"],
                    "route": q.get("route"), "reference": f"batch:n={len(items)}", "shape": q["image"]["shape"]})
    return out


app = create_app()
