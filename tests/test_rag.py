"""Offline tests for the RAG module (no network, no API keys)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from zelos.common.config import ROOT
from zelos.rag.corpus import chunk_words, load_chunks
from zelos.rag.embeddings import embed_texts
from zelos.rag.query_builder import GENERIC, queries_for_causes
from zelos.rag.retriever import HybridRetriever, rrf, tokenize

FIXTURE = ROOT / "tests" / "fixtures" / "rag_chunks.jsonl"


@pytest.fixture(autouse=True)
def _no_key(monkeypatch):
    monkeypatch.delenv("COHERE", raising=False)


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(load_chunks(FIXTURE), provider="hash")


def test_chunker_deterministic_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    a, b = chunk_words(text, 180, 30), chunk_words(text, 180, 30)
    assert a == b and len(a) >= 3
    assert a[0].split()[-30:] == a[1].split()[:30]
    assert all(len(c.split()) <= 180 for c in a)
    assert chunk_words("", 180, 30) == []


def test_chunk_ids_unique():
    spec = importlib.util.spec_from_file_location("build_corpus", ROOT / "scripts" / "build_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paper = {"rec": {"pmcid": "PMC1", "title": "T.", "license": "cc by"},
             "sections": [("Intro", " ".join(["alpha"] * 400)), ("Methods", " ".join(["beta"] * 400)),
                          ("Methods", " ".join(["gamma"] * 50)), ("Acknowledgments", " ".join(["thanks"] * 100))]}
    ch1, ch2 = mod.paper_chunks(paper, 180, 30), mod.paper_chunks(paper, 180, 30)
    assert ch1 == ch2
    ids = [c["chunk_id"] for c in ch1]
    assert len(ids) == len(set(ids)) and all(i.startswith("PMC1::s") for i in ids)
    assert not any("thanks" in c["text"] for c in ch1)  # non-content sections are skipped
    fixture_ids = [c["chunk_id"] for c in load_chunks(FIXTURE)]
    assert len(fixture_ids) == len(set(fixture_ids))


def test_rrf_toy():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63) and scores["d"] == pytest.approx(1 / 63)
    assert [d for d, _ in fused][:2] == ["a", "b"]  # tie broken by id, deterministic
    assert [d for d, _ in rrf([["x", "y"]], k=1)] == ["x", "y"]


def test_hash_embeddings_shape_norm_determinism():
    v = embed_texts(["air bubble in channel", "flat-field correction", "air bubble in channel"],
                    "search_document", provider="hash")
    assert v.shape == (3, 1024) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert np.array_equal(v[0], v[2])
    assert float(v[0] @ v[1]) < 0.5
    with pytest.raises(ValueError):
        embed_texts(["x"], "bogus", provider="hash")


def test_tokenize_stopwords():
    assert tokenize("The Air-Bubbles of 3 chips!") == ["air", "bubbles", "3", "chips"]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_retriever_offline_expected_chunk_in_top3(retriever, mode):
    hits = retriever.search("how do I remove air bubbles trapped in a microfluidic channel", k=3, mode=mode)
    assert "PMCT1::s0::0" in [h["chunk_id"] for h in hits]
    assert set(hits[0]) >= {"chunk_id", "text", "title", "url", "license", "pmcid", "score", "rank"}
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    hits = retriever.search("uneven illumination vignetting correction", k=3, mode=mode)
    assert hits[0]["chunk_id"] == "PMCT2::s0::0"


def test_abstain_on_nonsense_and_not_on_relevant(retriever):
    bad = retriever.search_with_abstain("zzqx wlorp ghtk blarf", k=5, mode="hybrid_rerank")
    assert bad["abstained"] and bad["hits"] == [] and bad["reason"]
    ok = retriever.search_with_abstain("air bubbles trapped in microfluidic channels block flow", k=5, mode="hybrid")
    assert not ok["abstained"] and ok["hits"]


def test_unknown_mode_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", mode="nope")


def test_query_builder_mapping():
    q = queries_for_causes(["BUBBLE", "UNEVEN_ILLUMINATION", "BUBBLE", "bubble"])
    assert len(q) == 2 and "bubble" in q[0].lower() and "illumination" in q[1].lower()
    assert queries_for_causes(["SOMETHING_NEW"]) == [GENERIC]
    assert queries_for_causes([]) == []


def test_default_corpus_files_consistent():
    manifest = Path(ROOT / "corpus" / "manifest.csv")
    if not manifest.exists():
        pytest.skip("corpus not built")
    import pandas as pd

    m = pd.read_csv(manifest)
    chunks = load_chunks()
    assert set(c["pmcid"] for c in chunks) == set(m.pmcid)
    assert m.n_chunks.sum() == len(chunks)
    assert set(m.license) <= {"cc by", "cc0", "CC BY", "CC0"}


def test_load_is_cache_only_and_falls_back_to_hash():
    import time

    t0 = time.time()
    r = HybridRetriever.load(chunks_path=FIXTURE)  # fixture texts are not in the Cohere cache
    assert r.provider == "hash" and time.time() - t0 < 15
    hits = r.search("remove air bubbles from channels", k=3)  # default mode degrades without a key
    assert hits and hits[0]["rank"] == 1
