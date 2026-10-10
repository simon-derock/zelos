"""Literature grounding: a cited chunk must be retrieved AND must support the sentence.

Support = content-word overlap ratio >= theta_lex  OR  embedding cosine >= theta_vec.
Choice: the lexical test is deterministic, offline and cheap, so it runs first; the vector test is a
second chance for paraphrases and is used only when an embedding backend is importable and
ZELOS_GATE_VECTORS is not "0". The overlap ratio is |claim content words found in chunk| / |claim
content words| after removing stop words, citations and numbers, with light suffix stemming.
Inputs : claim text, chunk dict(chunk_id, text). Outputs: support_score(...) -> dict.
"""
from __future__ import annotations

import os
import re

_STOP = set("""a an the and or but if then than that this these those of in on at to for from by with without
as is are was were be been being it its into over under between among about above below after before
during can could may might should would will shall not no nor do does did done has have had having which
who whom whose what when where why how also such via per both either each other their there here more
most less least very using used use based show shows shown find found suggest suggests image images
retrieved evidence literature according study studies""".split())
_BRACKET = re.compile(r"\[[^\[\]]*\]")


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def content_words(text: str) -> set[str]:
    t = _BRACKET.sub(" ", text or "").lower()
    toks = re.findall(r"[a-z][a-z\-]{2,}", t)
    return {_stem(w.strip("-")) for w in toks if w not in _STOP}


def lexical_support(claim_text: str, chunk_text: str) -> float:
    cw = content_words(claim_text)
    if not cw:
        return 0.0
    return len(cw & content_words(chunk_text)) / len(cw)


def vector_support(claim_text: str, chunk_text: str) -> float | None:
    """Cosine of embeddings, or None if the backend is unavailable/disabled."""
    if os.environ.get("ZELOS_GATE_VECTORS", "1") == "0":
        return None
    try:
        from zelos.rag.embeddings import embed_texts

        d = embed_texts([chunk_text[:2000]], input_type="search_document")
        q = embed_texts([_BRACKET.sub(" ", claim_text)], input_type="search_query")
        return float((d[0] * q[0]).sum())
    except Exception:
        return None


def support_score(claim_text: str, chunk: dict, theta_lex: float = 0.30, theta_vec: float = 0.35) -> dict:
    lex = lexical_support(claim_text, chunk.get("text", ""))
    out = {"lex": round(lex, 3), "vec": None, "supported": lex >= theta_lex}
    if not out["supported"]:
        v = vector_support(claim_text, chunk.get("text", ""))
        if v is not None:
            out["vec"] = round(v, 3)
            out["supported"] = v >= theta_vec
    return out
