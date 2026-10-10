"""Rule-based intent router (no LLM).

route(request) -> "inspect" | "triage" | "ask" | "certify" | "refuse"
request keys (any subset): text/question/query, image/path/qc, images/folder/qc_list, task, certify.
Refuse = diagnosis / treatment / clinical-advice requests, requests to invent data or citations, and
prompt-injection patterns. Order: refuse > explicit task > certify > triage > inspect > ask.
"""
from __future__ import annotations

import re

_REFUSE = re.compile(
    r"(diagnos(e|is|tic|ing)\b|\bprescrib|\bwhich drug\b|\bdos(e|age)s? (for|of) (a |the |my )?(patient|person|child|human)|"
    r"\bshould (i|we|the patient|my patient)\b.*\b(take|use|receive|start)\b|\bclinical (advice|decision|recommendation)|"
    r"\bmedical advice\b|\btreat (my|the|this|a|our) (patient|disease|cancer|condition|infection)|\btherapy for\b|"
    r"\bcure (for|the|my)\b|\bis (it|this|the patient) (cancer|malignant|benign)\b|"
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompt)|disregard (the |your |all )?(rules|instructions|system)|"
    r"(reveal|print|show|repeat) (your |the )?(system prompt|hidden prompt|instructions)|\byou are now\b|\bpretend (to be|you)\b|"
    r"\bjailbreak\b|\bdeveloper mode\b|"
    r"(invent|fabricate|make up|fake|forge|hallucinate|guess)\b.{0,40}\b(number|value|metric|score|citation|reference|paper|doi|pmcid|data|result)|"
    r"\b(cite|give|provide|list)\b.{0,30}\b(paper|reference|citation|study)\b.{0,40}\b(does not exist|doesn't exist|nonexistent|non-existent|fake|made[- ]up)|"
    r"(skip|bypass|disable|turn off) (the )?(gate|grounding|check|verification|safety))", re.I)

_TRIAGE = re.compile(r"\b(triage|session|folder|batch|all images|which images|re-?image list)\b", re.I)
_CERT = re.compile(r"\b(certif(y|icate)|sign(ed)? (json|record)|attest)\b", re.I)


def _strings(request: dict) -> str:
    return " ".join(str(v) for v in request.values() if isinstance(v, (str, int, float)))


def route(request: dict) -> str:
    text = _strings(request)
    if _REFUSE.search(text):
        return "refuse"
    task = str(request.get("task", "")).lower()
    if task in {"inspect", "triage", "ask", "certify"}:
        return task
    if request.get("certify") or _CERT.search(text):
        return "certify"
    if any(request.get(k) for k in ("qc_list", "images", "folder")) or (
            _TRIAGE.search(text) and not any(request.get(k) for k in ("image", "path", "qc"))):
        return "triage"
    if any(request.get(k) for k in ("image", "path", "qc")):
        return "inspect"
    return "ask"
