"""Corpus loading and chunking.

Inputs : corpus/chunks.jsonl, corpus/manifest.csv
Outputs: list[dict] chunks with keys chunk_id, pmcid, section, text, title, url, license.
"""
from __future__ import annotations

import json
from pathlib import Path

from zelos.common.config import ROOT, sha256_file

CORPUS_DIR = ROOT / "corpus"


def chunk_words(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    """Split text into word windows of ``size`` words with ``overlap`` words shared. Deterministic."""
    words = text.split()
    if not words:
        return []
    if size <= overlap:
        raise ValueError("size must exceed overlap")
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if out and len(piece) <= overlap:  # tail fully contained in previous chunk
            break
        out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def load_chunks(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else CORPUS_DIR / "chunks.jsonl"
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_sha256(path: str | Path | None = None) -> str:
    return sha256_file(Path(path) if path else CORPUS_DIR / "chunks.jsonl")
