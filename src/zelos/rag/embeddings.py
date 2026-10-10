"""Text embeddings with a record/replay cache.

Providers
  cohere : Cohere embed-v4.0 (v2 API, 1024-d float), batches of 96, retries with backoff.
  hash   : offline, deterministic sklearn HashingVectorizer (word 1-2 grams, 1024 features, L2).
Cache    : corpus/embed_cache.npz keyed by sha256(provider|model|input_type|text); Cohere vectors are
           stored as float16 and every returned Cohere vector passes through the same float16 round trip,
           so live and replayed runs agree bit for bit. Every live Cohere call is appended to
           corpus/cohere_ledger.jsonl (kind, n_texts, UTC time) so the call budget is auditable.
Inputs : list[str], input_type in {"search_document", "search_query"}
Outputs: float32 ndarray (n, dim), rows L2-normalised.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import UTC, datetime

import numpy as np

from zelos.common.config import ROOT, load_config, secret

CORPUS_DIR = ROOT / "corpus"
CACHE_PATH = CORPUS_DIR / "embed_cache.npz"
LEDGER_PATH = CORPUS_DIR / "cohere_ledger.jsonl"
DIM = 1024
_LOCK = threading.Lock()
_CACHE: dict[str, np.ndarray] | None = None
_DIRTY = False
LIVE_CALLS = {"embed": 0, "rerank": 0}  # Cohere calls made by this process


class EmbeddingUnavailable(RuntimeError):
    """Raised when provider='cohere' is requested but there is no key and the text is not cached."""


def _model(provider: str) -> str:
    if provider == "cohere":
        return str(load_config().get("embeddings", {}).get("model", "embed-v4.0"))
    return "hashing-word12-1024"


def cache_key(provider: str, input_type: str, text: str) -> str:
    return hashlib.sha256(f"{provider}|{_model(provider)}|{input_type}|{text}".encode()).hexdigest()


def _load_cache() -> dict[str, np.ndarray]:
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        _CACHE = _read_cache_file()
    return _CACHE


def _read_cache_file() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with np.load(CACHE_PATH, allow_pickle=False) as z:
            keys, vecs = z["keys"], z["vecs"]
            return {str(k): vecs[i] for i, k in enumerate(keys)}
    except Exception:  # truncated / concurrently written file: treat as empty, it is rebuilt on demand
        return {}


def _refresh_from_disk() -> None:
    """Merge entries written to the cache file by other processes (never drops in-memory entries)."""
    disk = _read_cache_file()
    with _LOCK:
        c = _load_cache()
        for k, v in disk.items():
            c.setdefault(k, v)


def flush_cache() -> None:
    """Persist the in-memory cache atomically (no-op if nothing new)."""
    global _DIRTY
    with _LOCK:
        if not _DIRTY or _CACHE is None:
            return
        keys = sorted(_CACHE)
        tmp = CACHE_PATH.with_name(f"embed_cache.{os.getpid()}.tmp.npz")
        np.savez_compressed(tmp, keys=np.array(keys), vecs=np.stack([_CACHE[k] for k in keys]).astype(np.float16))
        tmp.replace(CACHE_PATH)
        _DIRTY = False


def log_cohere_call(kind: str, n: int) -> None:
    LIVE_CALLS[kind] = LIVE_CALLS.get(kind, 0) + 1
    LEDGER_PATH.parent.mkdir(exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        f.write(json.dumps({"kind": kind, "n": n,
                            "utc": datetime.now(UTC).isoformat(timespec="seconds")}) + "\n")


def ledger_total() -> dict[str, int]:
    """Cumulative recorded Cohere calls per kind (from the committed ledger)."""
    out: dict[str, int] = {}
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text().splitlines():
            if line.strip():
                k = json.loads(line)["kind"]
                out[k] = out.get(k, 0) + 1
    return out


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def _hash_embed(texts: list[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(n_features=DIM, ngram_range=(1, 2), norm="l2", alternate_sign=False,
                           lowercase=True, token_pattern=r"(?u)\b\w+\b")
    return _l2(hv.transform(texts).toarray())


def _cohere_embed(texts: list[str], input_type: str, timeout: float = 60.0, attempts: int = 12) -> np.ndarray:
    import cohere

    key = secret("COHERE")
    if not key:
        raise EmbeddingUnavailable("COHERE key not set")
    co = cohere.ClientV2(api_key=key, timeout=timeout)
    model = _model("cohere")
    batch = int(load_config("rag").get("cohere_embed_batch", 96))
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        part = texts[i : i + batch]
        for attempt in range(attempts):
            try:
                r = co.embed(texts=part, model=model, input_type=input_type, embedding_types=["float"],
                             output_dimension=DIM)
                break
            except Exception as e:  # trial keys are limited to 100k tokens/min: wait for the window to reset
                if attempt == attempts - 1:
                    raise
                time.sleep(12 if ("429" in str(e) or "rate" in str(e).lower() or "TooMany" in type(e).__name__)
                           else 3)
        log_cohere_call("embed", len(part))
        out.append(np.asarray(r.embeddings.float_, dtype=np.float32))
    return np.concatenate(out, axis=0)


def _all_cached(texts: list[str], input_type: str, provider: str) -> bool:
    c = _load_cache()
    return all(cache_key(provider, input_type, t) in c for t in texts)


def all_cached(texts: list[str], input_type: str, provider: str = "cohere") -> bool:
    """True if every text already has a cached vector for this provider (no network access)."""
    return _all_cached(texts, input_type, provider)


def active_provider(texts: list[str] | None = None, input_type: str = "search_document") -> str:
    """'cohere' if a key is present or all given texts are cached under cohere, else 'hash'."""
    if secret("COHERE") and load_config()["embeddings"]["provider"] == "cohere":
        return "cohere"
    if texts and _all_cached(texts, input_type, "cohere"):
        return "cohere"
    return "hash"


def embed_texts(texts: list[str], input_type: str, provider: str | None = None, *, timeout: float = 60.0,
                attempts: int = 12) -> np.ndarray:
    """Embed texts -> float32 (n, 1024), L2-normalised rows. See module docstring for caching.

    timeout/attempts only matter for live Cohere calls; the retriever uses (10 s, 1 attempt) at query time."""
    global _DIRTY
    if input_type not in ("search_document", "search_query"):
        raise ValueError("input_type must be search_document or search_query")
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    provider = provider or active_provider(texts, input_type)
    if provider == "hash":
        return _hash_embed(texts)
    if provider != "cohere":
        raise ValueError(f"unknown provider {provider!r}")
    cache = _load_cache()
    keys = [cache_key("cohere", input_type, t) for t in texts]
    if any(k not in cache for k in keys):
        _refresh_from_disk()  # another process may have filled the cache meanwhile
    uniq: dict[str, str] = {}
    for t, k in zip(texts, keys, strict=True):
        if k not in cache:
            uniq.setdefault(k, t)
    if uniq:
        batch = int(load_config("rag").get("cohere_embed_batch", 96))
        items = list(uniq.items())
        for i in range(0, len(items), batch):
            part = items[i : i + batch]
            vecs = _l2(_cohere_embed([t for _, t in part], input_type, timeout, attempts)).astype(np.float16)
            with _LOCK:
                for (k, _), v in zip(part, vecs, strict=True):
                    cache[k] = v
                _DIRTY = True
            if (i // batch) % 2 == 1:
                flush_cache()  # persist progress so an interrupted build is not lost
        flush_cache()
    return _l2(np.stack([cache[k] for k in keys]).astype(np.float32))
