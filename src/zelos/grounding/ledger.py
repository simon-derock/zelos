"""Claim ledger schema and helpers.

Entry: {claim_id, text, type, evidence:[tool_call_id|chunk_id|claim_id], status, reason}
status: supported | unsupported | abstain | refused
"""
from __future__ import annotations

LEDGER_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Zelos claim ledger entry",
    "type": "object",
    "required": ["claim_id", "text", "type", "evidence", "status", "reason"],
    "properties": {
        "claim_id": {"type": "string", "pattern": "^c[0-9]+$"},
        "text": {"type": "string"},
        "type": {"enum": ["numeric", "literature", "inference"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "status": {"enum": ["supported", "unsupported", "abstain", "refused"]},
        "reason": {"type": "string"},
    },
    "additionalProperties": True,
}

STATUSES = {"supported", "unsupported", "abstain", "refused"}


def entry(claim_id: str, text: str, type_: str, evidence: list[str], status: str, reason: str, **extra) -> dict:
    assert status in STATUSES
    return {"claim_id": claim_id, "text": text, "type": type_, "evidence": list(evidence),
            "status": status, "reason": reason, **extra}


def summarize(entries: list[dict]) -> dict:
    n_sup = sum(e["status"] == "supported" for e in entries)
    n_uns = sum(e["status"] == "unsupported" for e in entries)
    judged = n_sup + n_uns
    return {"n_supported": n_sup, "n_unsupported": n_uns,
            "grounded_rate": (n_sup / judged) if judged else None}


def validate_entry(e: dict) -> list[str]:
    errs = [f"missing {k}" for k in LEDGER_SCHEMA["required"] if k not in e]
    if e.get("status") not in STATUSES:
        errs.append("bad status")
    if e.get("type") not in {"numeric", "literature", "inference"}:
        errs.append("bad type")
    return errs
