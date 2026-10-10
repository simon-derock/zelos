"""Zelos MCP server (FastMCP). Exposes the QC/evidence/grounding tools to any MCP-capable client.

Tools: inspect_image, explain_image, retrieve_evidence, check_claims, triage_folder.
Heavy modules (vision, retrieval) are imported lazily so the server starts and lists tools offline.
Run:  python -m zelos.mcp_server   (stdio transport)
Research tool. Not a medical device. No clinical or diagnostic claims.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastmcp import FastMCP

mcp = FastMCP("zelos")
IMAGE_EXT = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


@lru_cache(maxsize=1)
def _retriever():
    try:
        from zelos.rag.retriever import HybridRetriever

        return HybridRetriever.load()
    except Exception:  # noqa: BLE001 - retrieval is optional
        return None


@mcp.tool
def inspect_image(path: str) -> dict:
    """Compute deterministic QC metrics, z-scores, level (PASS/REVIEW/FAIL) and causes for one image file."""
    from zelos.vision.pipeline import inspect

    return inspect(path)


@mcp.tool
def explain_image(path: str, provider: str = "template") -> dict:
    """Inspect an image, retrieve evidence, explain it, and return the gated card and claim ledger.
    provider: "template" (no LLM, offline) or "workers_ai"."""
    from zelos.agent.loop import run_inspect
    from zelos.vision.pipeline import inspect

    return run_inspect(inspect(path), _retriever(), provider)


@mcp.tool
def retrieve_evidence(query: str, k: int = 5) -> dict:
    """Retrieve literature passages (chunk_id, text, title, url, license) for a query; abstains if unsupported."""
    r = _retriever()
    if r is None:
        return {"hits": [], "abstained": True, "reason": "retriever unavailable"}
    return r.search_with_abstain(query, k=k)


@mcp.tool
def check_claims(claims_json: str, tool_outputs_json: str, evidence_json: str) -> dict:
    """Verify claims against tool outputs and retrieved evidence. Returns the claim ledger with
    supported/unsupported status per claim and the grounded rate. Inputs are JSON strings."""
    from zelos.grounding.extract import parse_response
    from zelos.grounding.gate import verify

    raw = json.loads(claims_json)
    claims = raw["claims"] if isinstance(raw, dict) and "claims" in raw else raw
    if isinstance(claims, str):
        claims = parse_response(claims)["claims"]
    return verify(claims, json.loads(tool_outputs_json), json.loads(evidence_json))


@mcp.tool
def triage_folder(path: str) -> dict:
    """Run QC over every image in a folder and return session statistics and the re-image list (template summary)."""
    from zelos.agent.loop import run_triage
    from zelos.vision.pipeline import inspect

    files = sorted(p for p in Path(path).iterdir() if p.suffix.lower() in IMAGE_EXT)
    return run_triage([inspect(str(p)) for p in files], _retriever(), "template")


def tool_names() -> list[str]:
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    return sorted(t.name for t in tools)


if __name__ == "__main__":
    mcp.run()
