"""Zelos HTTP API (FastAPI). Same tools as the CLI/MCP server.

GET  /api/health                      liveness + active providers
POST /api/inspect   (multipart file)  QC certificate (0 LLM calls)
POST /api/explain   (multipart file)  gated explanation + ledger (provider=template|workers_ai)
POST /api/ask       {"question": ""}  cited answer or ABSTAIN
Run: uvicorn zelos.api.main:app --port 8000
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Zelos", version="1.0.0", description="Evidence-gated QC for organ-on-a-chip images. Research use only.")
MAX_BYTES = 25 * 1024 * 1024


@lru_cache
def retriever():
    try:
        from zelos.rag.retriever import HybridRetriever

        return HybridRetriever.load()
    except Exception:
        return None


async def _qc(file: UploadFile) -> dict:
    from zelos.vision.pipeline import inspect

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image too large")
    try:
        return inspect(data, image_id=(file.filename or "upload").rsplit(".", 1)[0])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "retriever_loaded": retriever.cache_info().currsize > 0}


@app.post("/api/inspect")
async def inspect_ep(file: UploadFile = File(...)) -> dict:
    return await _qc(file)


@app.post("/api/explain")
async def explain_ep(file: UploadFile = File(...), provider: str = "template") -> dict:
    from zelos.agent.loop import run_inspect

    if provider not in {"template", "workers_ai"}:
        raise HTTPException(400, "provider must be template or workers_ai")
    return run_inspect(await _qc(file), retriever=retriever(), provider=provider)


class Ask(BaseModel):
    question: str
    provider: str = "template"


@app.post("/api/ask")
def ask_ep(body: Ask) -> dict:
    from zelos.agent.loop import run_ask

    if len(body.question) > 1000:
        raise HTTPException(413, "question too long")
    return run_ask(body.question, retriever=retriever(), provider=body.provider)
