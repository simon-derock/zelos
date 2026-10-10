"""Claim extraction.

Inputs : raw LLM text (strict JSON {"claims": [...]} preferred; free text tolerated).
Outputs: parse_response(text) -> {"claims": [{text, type, evidence}], "abstain": bool, "reason": str, "format": str}
         extract_claims(text) -> list of claim dicts.
Types: numeric (contains a number), literature (cites a [chunk_id]), inference (otherwise).
"""
from __future__ import annotations

import json
import re

TYPES = {"numeric", "literature", "inference"}
_BRACKET = re.compile(r"\[([^\[\]]{2,120})\]")
_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")
_ABSTAIN = re.compile(r"^\s*ABSTAIN\b[:\-\s]*(.*)$", re.I | re.S)


def citations_in(text: str) -> list[str]:
    """Return bracketed citation ids inside text, e.g. '[qc-1a2b]' -> 'qc-1a2b'."""
    out: list[str] = []
    for m in _BRACKET.finditer(text or ""):
        for part in re.split(r"[;,]\s*", m.group(1)):
            part = part.strip()
            if part and " " not in part and part not in out:
                out.append(part)
    return out


def infer_type(text: str, evidence: list[str]) -> str:
    has_chunk = any("::" in e for e in evidence) or any("::" in c for c in citations_in(text))
    no_cites = _BRACKET.sub("", text or "")
    if has_chunk:
        return "literature"
    if _NUM.search(no_cites):
        return "numeric"
    return "inference"


def _clean_claim(c: dict | str) -> dict | None:
    if isinstance(c, str):
        c = {"text": c}
    if not isinstance(c, dict):
        return None
    text = str(c.get("text", "")).strip()
    if not text:
        return None
    ev_raw = c.get("evidence") or c.get("citations") or []
    if isinstance(ev_raw, str):
        ev_raw = [ev_raw]
    ev = [str(e).strip().strip("[]") for e in ev_raw if str(e).strip()]
    for cid in citations_in(text):
        if cid not in ev:
            ev.append(cid)
    typ = str(c.get("type", "")).lower()
    if typ not in TYPES:
        typ = infer_type(text, ev)
    return {"text": text, "type": typ, "evidence": ev}


def _json_obj(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def parse_response(text: str) -> dict:
    """Parse an LLM response into claims. Never raises."""
    text = text or ""
    m = _ABSTAIN.match(text)
    if m and not text.lstrip().startswith("{"):
        return {"claims": [], "abstain": True, "reason": m.group(1).strip()[:300], "format": "abstain"}
    obj = _json_obj(text)
    if isinstance(obj, dict):
        if obj.get("abstain") or str(obj.get("status", "")).upper() == "ABSTAIN":
            return {"claims": [], "abstain": True, "reason": str(obj.get("reason", ""))[:300], "format": "json"}
        raw = obj.get("claims")
        if isinstance(raw, list):
            claims = [c for c in (_clean_claim(x) for x in raw) if c]
            if not claims and any(isinstance(x, str) and x.upper().startswith("ABSTAIN") for x in raw):
                return {"claims": [], "abstain": True, "reason": "model abstained", "format": "json"}
            return {"claims": claims, "abstain": False, "reason": "", "format": "json"}
    if isinstance(obj, list):
        claims = [c for c in (_clean_claim(x) for x in obj) if c]
        return {"claims": claims, "abstain": False, "reason": "", "format": "json"}
    claims = [c for c in (_clean_claim(s) for s in split_sentences(text)) if c]
    return {"claims": claims, "abstain": False, "reason": "", "format": "text"}


def extract_claims(text: str) -> list[dict]:
    return parse_response(text)["claims"]
