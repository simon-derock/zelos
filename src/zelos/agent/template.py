"""Deterministic explainer: claims supported by construction (no LLM).

Inputs : a QC dict (vision schema) / triage stats / evidence list.
Outputs: list of claim dicts {text, type, evidence}. Numbers are copied from the tool output and cited with
its tool_call_id; the literature sentence is a <=25-word excerpt of a retrieved chunk with its chunk_id.
"""
from __future__ import annotations

import re

from zelos.grounding.literature import content_words

MAX_EXCERPT_WORDS = 25


def fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3g}"


def best_excerpt(text: str, query: str = "", n: int = MAX_EXCERPT_WORDS) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "")) if len(s.split()) >= 4]
    if not sents:
        return " ".join((text or "").split()[:n])
    q = content_words(query)
    best = max(sents, key=lambda s: (len(q & content_words(s)), -abs(len(s.split()) - n)))
    words = best.split()
    out = " ".join(words[:n])
    return out if len(words) <= n else out.rstrip(".,;:") + " ..."


def excerpt_claim(chunk: dict, query: str = "") -> dict:
    ex = best_excerpt(chunk.get("text", ""), query)
    return {"text": f"Retrieved evidence: \"{ex}\" [{chunk['chunk_id']}]", "type": "literature",
            "evidence": [chunk["chunk_id"]]}


def explain_qc(qc: dict, evidence: list[dict]) -> list[dict]:
    tid = qc["tool_call_id"]
    img = (qc.get("image") or {}).get("id", "image")
    claims = [{"text": f"Image {img} has QC score {fmt(qc['qc_score'])} and level {qc['level']} [{tid}].",
               "type": "numeric", "evidence": [tid]}]
    for c in qc.get("causes", [])[:3]:
        t = f"Cause {c['label']}: {c['note']}; {c['metric']} = {fmt(c['value'])} (z = {fmt(c['z'])}) [{tid}]."
        claims.append({"text": t, "type": "numeric", "evidence": [tid]})
    if qc.get("causes"):
        claims.append({"text": f"Recommended action for {qc['causes'][0]['label']}: {qc['causes'][0]['action']} [{tid}]",
                       "type": "inference", "evidence": [tid]})
    if qc.get("density_state"):
        claims.append({"text": f"Culture density state is {qc['density_state']} [{tid}].",
                       "type": "inference", "evidence": [tid]})
    if evidence:
        q = " ".join(f"{c['label']} {c['note']}" for c in qc.get("causes", [])) or "image quality"
        claims.append(excerpt_claim(evidence[0], q))
    return claims


def explain_triage(stats: dict, evidence: list[dict]) -> list[dict]:
    tid = stats["tool_call_id"]
    cnt = stats["counts"]
    claims = [{"text": f"Session has {stats['n']} images: {cnt.get('PASS', 0)} PASS, {cnt.get('REVIEW', 0)} REVIEW, "
                       f"{cnt.get('FAIL', 0)} FAIL [{tid}].", "type": "numeric", "evidence": [tid]}]
    if stats["dominant_causes"]:
        top = stats["dominant_causes"][0]
        claims.append({"text": f"Most frequent cause is {top['label']} in {top['count']} images [{tid}].",
                       "type": "numeric", "evidence": [tid]})
    if stats["reimage"]:
        claims.append({"text": f"{len(stats['reimage'])} positions are listed for re-imaging [{tid}].",
                       "type": "numeric", "evidence": [tid]})
    if evidence:
        claims.append(excerpt_claim(evidence[0], stats["dominant_causes"][0]["label"] if stats["dominant_causes"] else ""))
    return claims


def explain_ask(question: str, evidence: list[dict]) -> list[dict]:
    return [excerpt_claim(e, question) for e in evidence[:2]]
