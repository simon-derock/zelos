"""Numeric grounding: every number in a claim must come from a tool output.

Inputs : claim text; tool outputs (dicts, any nesting).
Outputs: extract_numbers(text) -> [{raw, value, decimals, pct, sci}]
         flatten_numeric(obj) -> [(path, float)]
         match_number(num, leaves, rel_tol) -> matched leaf path or None
A claimed number matches a tool value t if: |c - t| <= rel_tol*max(|c|,|t|); or (claim has >=1 decimals)
|c - t| <= half a unit in the claim's last decimal (i.e. the tool value rounded to the claim's decimals);
or, for percentages, c == 100*t (fraction shown as %) or c/100 == t; sign is ignored when the claim
gives no explicit sign (magnitude statements such as "4.2 SD below"). Numbers inside ids and
bracketed citations (qc-1a2b, PMC123::s1::0, 220429_01, 3D) are not treated as claims.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(
    r"(?P<sci>[-+]?\d+(?:\.\d+)?[eE][-+]?\d+)|"
    r"(?P<num>[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?|[-+]?\.\d+)"
    r"(?P<pct>\s?%|\s?percent\b)?"
)
_BRACKET = re.compile(r"\[[^\[\]]*\]")
_ID_LIKE = re.compile(r"\b[A-Za-z]+[-_]?[A-Za-z0-9]*\d[A-Za-z0-9_-]*\b")  # PMC123, qc-1a2b, log2


def extract_numbers(text: str) -> list[dict]:
    t = _BRACKET.sub(" ", text or "")
    t = re.sub(r"[A-Za-z0-9]+(?:::[A-Za-z0-9]+)+", " ", t)
    t = _ID_LIKE.sub(" ", t)
    t = re.sub(r"\b\d+_\d+\b|\b\d{4}-\d{2}-\d{2}\b|\b\d+[Dd]\b|\b\d+x\b", " ", t)
    out = []
    for m in _TOKEN.finditer(t):
        raw = m.group(0).strip()
        s = m.group("sci") or m.group("num")
        i, j = m.start(), m.end()
        prev = t[i - 1] if i > 0 else " "
        nxt = t[j] if j < len(t) else " "
        if prev.isalnum() and not s.startswith(("-", "+")):
            continue
        if nxt == "_":
            continue
        val = float(s.replace(",", ""))
        dec = len(s.split(".")[1].split("e")[0].split("E")[0]) if "." in s and not m.group("sci") else 0
        out.append({"raw": raw, "value": val, "decimals": dec, "pct": bool(m.group("pct")),
                    "sci": bool(m.group("sci")), "signed": s.startswith(("-", "+"))})
    return out


def flatten_numeric(obj, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if obj == obj and abs(obj) != float("inf"):
            out.append((prefix, float(obj)))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(flatten_numeric(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(flatten_numeric(v, f"{prefix}[{i}]"))
    return out


def _close(c: float, t: float, rel_tol: float) -> bool:
    return abs(c - t) <= rel_tol * max(abs(c), abs(t), 1e-12)


def _leaf_matches(num: dict, t: float, rel_tol: float) -> bool:
    c, dec = num["value"], num["decimals"]
    for cv in ([c] if num["signed"] else [c, -c]):
        if _close(cv, t, rel_tol) or (num["sci"] and _close(cv, t, 1e-2)):
            return True
        if dec >= 1 and abs(cv - t) <= 0.5 * 10 ** (-dec) + 1e-9:
            return True
        if num["pct"] and (_close(cv, t * 100.0, rel_tol) or _close(cv / 100.0, t, rel_tol)
                           or (dec >= 1 and abs(cv - t * 100.0) <= 0.5 * 10 ** (-dec) + 1e-9)):
            return True
    return False


def match_all(num: dict, leaves: list[tuple[str, float]], rel_tol: float = 1e-3) -> list[str]:
    """Every leaf path that the claimed number matches."""
    return [p for p, t in leaves if _leaf_matches(num, t, rel_tol)]


def match_number(num: dict, leaves: list[tuple[str, float]], rel_tol: float = 1e-3) -> str | None:
    m = match_all(num, leaves, rel_tol)
    return m[0] if m else None


def check_numbers(text: str, leaves: list[tuple[str, float]], rel_tol: float = 1e-3,
                  metric_of=None, mentioned: set[str] | None = None) -> dict:
    """Return {"numbers", "unmatched", "matches", "misbound"}.

    Metric binding: if the claim names metric(s) (`mentioned`) and a number matches only leaves that belong to
    other metrics (metric_of(path) -> name or None), it is "misbound" (a value swapped between metrics)."""
    nums = extract_numbers(text)
    unmatched, matches, misbound = [], {}, []
    for n in nums:
        hits = match_all(n, leaves, rel_tol)
        if not hits:
            unmatched.append(n["raw"])
            continue
        matches[n["raw"]] = hits[0]
        if mentioned and metric_of:
            owners = [metric_of(h) for h in hits]
            if all(o is not None and o not in mentioned for o in owners):
                misbound.append(n["raw"])
    return {"numbers": nums, "unmatched": unmatched, "matches": matches, "misbound": misbound}
