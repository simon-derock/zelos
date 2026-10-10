"""Hybrid retriever: BM25 + dense (Qdrant, cosine) fused with RRF, optional Cohere rerank-v3.5.

Inputs : corpus/chunks.jsonl (or a fixture), configs/default.yaml `retrieval:`, configs/rag.yaml
Outputs: hits = list of dict(chunk_id, text, title, url, license, pmcid, section, score, rank)
Modes  : bm25 | dense | hybrid | hybrid_rerank
Offline: with no Cohere key and no cache the retriever uses hash vectors and skips reranking; reranking
         and Cohere vectors are replayed from committed caches (corpus/) whenever they exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from zelos.common.config import ROOT, load_config, secret
from zelos.rag import embeddings as emb
from zelos.rag.corpus import load_chunks

RERANK_CACHE = ROOT / "corpus" / "rerank_cache.json"
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")
STOP = frozenset("""a an and are as at be been but by for from has have in into is it its of on or such that the their
then there these this to was were which with we our not can may also than more most other using used use via""".split())
_TOK = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    score: dict[str, float] = {}
    for ranking in rankings:
        for r, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    def __init__(self, chunks: list[dict], cfg: dict | None = None, provider: str | None = None):
        base = {**load_config()["retrieval"], **load_config("rag")}
        self.cfg = {**base, **(cfg or {})}
        self.chunks = chunks
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.ids = [c["chunk_id"] for c in chunks]
        self.rerank_model = str(load_config().get("rerank", {}).get("model", "rerank-v3.5"))
        self._lock = threading.Lock()
        self._rr_cache: dict[str, list] | None = None
        self.rerank_calls = 0
        self._last_rerank = 0.0
        self._hash_vecs: np.ndarray | None = None

        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

        texts = [c["text"] for c in chunks]
        # load() never touches the network: Cohere document vectors are used only if every chunk is cached,
        # otherwise the whole index falls back to the hash provider.
        self.provider = provider or ("cohere" if emb.all_cached(texts, "search_document", "cohere") else "hash")
        self._vecs = emb.embed_texts(texts, "search_document", provider=self.provider, timeout=10, attempts=1)
        self._qdrant = None
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, PointStruct, VectorParams

            self._qdrant = QdrantClient(location=":memory:")
            self._qdrant.create_collection("zelos_chunks",
                                           vectors_config=VectorParams(size=self._vecs.shape[1],
                                                                       distance=Distance.COSINE))
            step = 500
            for i in range(0, len(chunks), step):
                self._qdrant.upsert("zelos_chunks", points=[
                    PointStruct(id=j, vector=self._vecs[j].tolist()) for j in range(i, min(i + step, len(chunks)))])
        except Exception:
            self._qdrant = None  # numpy cosine fallback

    @classmethod
    def load(cls, cfg: dict | None = None, chunks_path: str | Path | None = None) -> HybridRetriever:
        return cls(load_chunks(chunks_path), cfg)

    # ---------------------------------------------------------------- channels
    def _bm25_rank(self, query: str, n: int) -> list[tuple[str, float]]:
        toks = tokenize(query)
        if not toks:
            return []
        s = self._bm25.get_scores(toks)
        order = np.argsort(-s, kind="stable")[:n]
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]

    def _hash_dense(self, query: str, n: int) -> list[tuple[str, float]]:
        """Offline fallback when the query vector is not cached and no key is set (separate hash vector space)."""
        with self._lock:
            if self._hash_vecs is None:
                self._hash_vecs = emb.embed_texts([c["text"] for c in self.chunks], "search_document",
                                                  provider="hash")
        q = emb.embed_texts([query], "search_query", provider="hash")[0]
        sims = self._hash_vecs @ q
        order = np.argsort(-sims, kind="stable")[:n]
        return [(self.ids[i], float(sims[i])) for i in order]

    def _dense_rank(self, query: str, n: int) -> tuple[list[tuple[str, float]], bool]:
        """Dense ranking and a flag telling whether the hash fallback was used."""
        try:
            q = emb.embed_texts([query], "search_query", provider=self.provider, timeout=10, attempts=1)[0]
        except Exception:  # no key, not cached, timeout, 429: silently use the offline hash channel
            if self.provider == "hash":
                raise
            return self._hash_dense(query, n), True
        if self._qdrant is not None:
            pts = self._qdrant.query_points("zelos_chunks", query=q.tolist(), limit=n).points
            return [(self.ids[int(p.id)], float(p.score)) for p in pts], False
        sims = self._vecs @ q
        order = np.argsort(-sims, kind="stable")[:n]
        return [(self.ids[i], float(sims[i])) for i in order], False

    # ---------------------------------------------------------------- rerank
    def _load_rr(self) -> dict[str, list]:
        if self._rr_cache is None:
            self._rr_cache = json.loads(RERANK_CACHE.read_text()) if RERANK_CACHE.exists() else {}
        return self._rr_cache

    def _rr_key(self, query: str, cand: list[str]) -> str:
        sig = "|".join(f"{c}:{hashlib.sha256(self.by_id[c]['text'].encode()).hexdigest()[:8]}" for c in cand)
        return hashlib.sha256(f"{self.rerank_model}|{query}|{sig}".encode()).hexdigest()

    def _rerank(self, query: str, cand: list[str]) -> list[tuple[str, float]] | None:
        """Relevance score for every candidate, or None when neither key nor cache is available."""
        key = self._rr_key(query, cand)
        with self._lock:
            cache = self._load_rr()
            if key in cache:
                return [(c, float(s)) for c, s in cache[key]]
        api = secret("COHERE")
        if not api or load_config()["rerank"]["provider"] != "cohere":
            return None
        import cohere

        co = cohere.ClientV2(api_key=api, timeout=float(self.cfg.get("rerank_timeout_s", 10)))
        gap = float(self.cfg.get("rerank_min_interval_s", 0))
        if gap > 0:  # optional client-side pacing for rate-limited (trial) keys
            with self._lock:
                wait = self._last_rerank + gap - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
                self._last_rerank = time.monotonic()
        r = None
        attempts = 1 + int(self.cfg.get("rerank_retries", 0))
        for attempt in range(attempts):
            try:
                r = co.rerank(model=self.rerank_model, query=query,
                              documents=[self.by_id[c]["text"] for c in cand], top_n=len(cand))
                break
            except Exception as e:
                if attempt == attempts - 1:
                    return None  # any error: fall back to RRF order
                time.sleep(15 if ("429" in str(e) or "rate" in str(e).lower()) else 2)
        if r is None:
            return None
        emb.log_cohere_call("rerank", len(cand))
        self.rerank_calls += 1
        scored = [(cand[x.index], float(x.relevance_score)) for x in r.results]
        with self._lock:
            cache[key] = [[c, s] for c, s in scored]
            try:  # merge entries other processes wrote meanwhile, then replace atomically
                for k2, v2 in json.loads(RERANK_CACHE.read_text()).items():
                    cache.setdefault(k2, v2)
            except Exception:
                pass
            tmp = RERANK_CACHE.with_name(f"rerank_cache.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache, sort_keys=True))
            tmp.replace(RERANK_CACHE)
        return scored

    # ---------------------------------------------------------------- search
    def _hit(self, cid: str, score: float, rank: int) -> dict[str, Any]:
        c = self.by_id[cid]
        return {"chunk_id": cid, "text": c["text"], "title": c["title"], "url": c["url"], "license": c["license"],
                "pmcid": c["pmcid"], "section": c.get("section", ""), "score": float(score), "rank": rank}

    def _search(self, query: str, k: int, mode: str) -> tuple[list[dict], dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = int(self.cfg["candidates"])
        info: dict[str, Any] = {"bm25_top": 0.0, "dense_top": 0.0, "rerank_top": None, "degraded": ""}
        bm = self._bm25_rank(query, n)
        de: list[tuple[str, float]] = []
        if mode != "bm25":
            de, fell_back = self._dense_rank(query, n)
            if fell_back:
                info["degraded"] = "dense_hash_fallback"
        info["bm25_top"] = bm[0][1] if bm else 0.0
        info["dense_top"] = de[0][1] if de else 0.0
        if mode == "bm25" or (mode == "dense" and not de):
            ranked = bm
        elif mode == "dense":
            ranked = de
        else:
            fused = rrf([[c for c, _ in bm], [c for c, _ in de]], int(self.cfg["rrf_k"]))
            ranked = fused
            if mode == "hybrid_rerank" and fused:
                pool = [c for c, _ in fused[: int(self.cfg["rerank_pool"])]]
                rr = self._rerank(query, pool)
                if rr is not None:
                    ranked = sorted(rr, key=lambda t: (-t[1], t[0]))
                    info["rerank_top"] = ranked[0][1]
                else:
                    info["degraded"] = (info["degraded"] + " rerank_unavailable").strip()
        hits = [self._hit(c, s, r) for r, (c, s) in enumerate(ranked[:k], start=1)]
        return hits, info

    def search(self, query: str, k: int = 5, mode: str = "hybrid_rerank") -> list[dict]:
        return self._search(query, k, mode)[0]

    def search_with_abstain(self, query: str, k: int = 5, mode: str = "hybrid_rerank") -> dict:
        """Hits plus an abstain decision.

        Rerank modes: abstain if the best rerank relevance < cfg['min_rerank'].
        Other modes (or rerank unavailable): abstain if best BM25 score < min_bm25 AND best dense cosine < min_dense.
        """
        hits, info = self._search(query, k, mode)
        if info["rerank_top"] is not None:
            thr = float(self.cfg["min_rerank"])
            ab = info["rerank_top"] < thr
            reason = f"top rerank relevance {info['rerank_top']:.3f} {'<' if ab else '>='} {thr}"
        else:
            tb, td = float(self.cfg["min_bm25"]), float(self.cfg["min_dense"])
            low_b, low_d = info["bm25_top"] < tb, info["dense_top"] < td
            # each mode only consults the channel(s) it actually uses; hybrid needs both to be weak
            ab = {"bm25": low_b, "dense": low_d}.get(mode, low_b and low_d)
            reason = (f"bm25 top {info['bm25_top']:.2f} (min {tb}), dense top {info['dense_top']:.3f} (min {td}): "
                      + ("below thresholds" if ab else "supported"))
        if not hits:
            ab, reason = True, "no candidates retrieved"
        if info["degraded"]:
            reason += f" [{info['degraded']}]"
        return {"hits": [] if ab else hits, "abstained": bool(ab), "reason": reason}
