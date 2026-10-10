"""E4: retrieval evaluation on the gold set (BM25 vs dense vs hybrid RRF vs hybrid+rerank) and abstention.

Inputs : eval/retrieval_gold.jsonl, corpus/chunks.jsonl, corpus caches (embeddings, rerank)
Outputs: results/e4.json, results/e4.md   (every number is computed here; nothing is typed by hand)
Usage  : python scripts/eval_retrieval.py [--offline]
         --offline drops the Cohere key so only committed caches are used (replay check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from zelos.common.config import ROOT, provenance, write_json
from zelos.rag import embeddings as emb
from zelos.rag.corpus import corpus_sha256
from zelos.rag.retriever import MODES, HybridRetriever

GOLD = ROOT / "eval" / "retrieval_gold.jsonl"
KS = (1, 3, 5)
DEPTH = 10


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def ndcg(rels: list[int], n_ideal: int, k: int = 5) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_ideal, k)))
    return dcg / idcg if idcg else 0.0


def level_metrics(rank_lists: list[list[int]], n_ideal: list[int],
                  ndcg_lists: list[list[int]] | None = None) -> dict:
    """rank_lists[i] = binary relevance of retrieved items (top DEPTH) for query i.

    Hit@k and MRR use rank_lists. nDCG@5 uses ndcg_lists when given (paper level: relevance of the
    de-duplicated paper ranking, so a paper is credited once), else rank_lists."""
    n = len(rank_lists)
    out: dict = {"n": n}
    for k in KS:
        hits = sum(1 for r in rank_lists if any(r[:k]))
        out[f"hit@{k}"] = hits / n
        out[f"hit@{k}_ci95"] = wilson(hits, n)
    out["mrr"] = float(np.mean([next((1 / (i + 1) for i, v in enumerate(r) if v), 0.0) for r in rank_lists]))
    nl = ndcg_lists if ndcg_lists is not None else rank_lists
    out["ndcg@5"] = float(np.mean([ndcg(r, ni) for r, ni in zip(nl, n_ideal, strict=True)]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if args.offline:
        os.environ.pop("COHERE", None)
    gold = [json.loads(line) for line in GOLD.read_text().splitlines() if line.strip()]
    ret_q = [g for g in gold if g["type"] in ("answerable", "reworded")]
    un_q = [g for g in gold if g["type"] == "unanswerable"]
    t_start = time.time()
    r = HybridRetriever.load({"rerank_min_interval_s": 6.2})  # pacing for the 10 calls/min trial limit
    # one batched Cohere call for every query vector (replayed from cache afterwards)
    emb.embed_texts([g["question"] for g in gold], "search_query", provider=r.provider)
    paper_ids = {c["pmcid"] for c in r.chunks}

    res: dict = {"modes": {}, "abstention": {}}
    for mode in MODES:
        paper_rl, paper_dd, chunk_rl, ideal_p, ideal_c, lat = [], [], [], [], [], []
        by_type: dict[str, list[int]] = {"answerable": [], "reworded": []}
        for g in ret_q:
            t0 = time.perf_counter()
            hits = r.search(g["question"], k=DEPTH, mode=mode)
            lat.append((time.perf_counter() - t0) * 1000)
            pr = [int(h["pmcid"] in g["gold_pmcids"]) for h in hits]
            cr = [int(h["chunk_id"] in g["gold_chunk_ids"]) for h in hits]
            pr += [0] * (DEPTH - len(pr))
            cr += [0] * (DEPTH - len(cr))
            paper_rl.append(pr)
            seen: set[str] = set()
            dd = []
            for h in hits:  # first occurrence of each paper only
                if h["pmcid"] not in seen:
                    seen.add(h["pmcid"])
                    dd.append(int(h["pmcid"] in g["gold_pmcids"]))
            paper_dd.append(dd + [0] * (DEPTH - len(dd)))
            chunk_rl.append(cr)
            ideal_p.append(len(g["gold_pmcids"]))
            ideal_c.append(len(g["gold_chunk_ids"]))
            by_type[g["type"]].append(int(any(pr[:5])))
        res["modes"][mode] = {
            "paper_level": level_metrics(paper_rl, ideal_p, paper_dd),
            "chunk_level": level_metrics(chunk_rl, ideal_c),
            "paper_hit@5_by_type": {t: {"n": len(v), "hit@5": float(np.mean(v))} for t, v in by_type.items()},
            "latency_ms_p50": float(np.median(lat)),
        }
        # abstention
        ab_un = [r.search_with_abstain(g["question"], k=5, mode=mode)["abstained"] for g in un_q]
        ab_an = [r.search_with_abstain(g["question"], k=5, mode=mode)["abstained"] for g in ret_q]
        top_un = [r._search(g["question"], 5, mode)[1] for g in un_q]
        top_an = [r._search(g["question"], 5, mode)[1] for g in ret_q]
        key = "rerank_top" if mode == "hybrid_rerank" and top_un and top_un[0]["rerank_top"] is not None else None
        res["abstention"][mode] = {
            "accuracy_on_unanswerable": float(np.mean(ab_un)), "n_unanswerable": len(un_q),
            "accuracy_on_unanswerable_ci95": wilson(int(sum(ab_un)), len(un_q)),
            "false_abstention_rate_on_answerable": float(np.mean(ab_an)), "n_answerable": len(ret_q),
            "false_abstention_rate_ci95": wilson(int(sum(ab_an)), len(ret_q)),
            "top_score_median_unanswerable": float(np.median([t[key] if key else max(t["bm25_top"], t["dense_top"])
                                                              for t in top_un])),
            "top_score_median_answerable": float(np.median([t[key] if key else max(t["bm25_top"], t["dense_top"])
                                                            for t in top_an])),
            "top_score_kind": "rerank_relevance" if key else "max(bm25_top, dense_top) raw",
        }

    n_papers = len(paper_ids)
    res["config"] = {"k_depth": DEPTH, "embedding_provider": r.provider,
                     "embedding_model": emb._model(r.provider), "rerank_model": r.rerank_model,
                     "rerank_available": res["abstention"]["hybrid_rerank"]["top_score_kind"] == "rerank_relevance",
                     "thresholds": {k: r.cfg[k] for k in ("min_rerank", "min_bm25", "min_dense", "candidates",
                                                          "rrf_k", "rerank_pool")},
                     "offline_flag": bool(args.offline)}
    res["corpus"] = {"n_papers": n_papers, "n_chunks": len(r.chunks), "chunks_sha256": corpus_sha256()}
    res["gold"] = {"path": "eval/retrieval_gold.jsonl", "sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
                   "n_answerable": sum(g["type"] == "answerable" for g in gold),
                   "n_reworded": sum(g["type"] == "reworded" for g in gold), "n_unanswerable": len(un_q)}
    res["cohere_calls"] = {"this_run": dict(emb.LIVE_CALLS), "cumulative_recorded_in_ledger": emb.ledger_total()}
    res["runtime_s"] = time.time() - t_start
    res["provenance"] = provenance({"corpus_sha256": res["corpus"]["chunks_sha256"]})
    write_json(ROOT / "results" / "e4.json", res)
    write_md(res)
    print(f"wrote results/e4.json  (provider={r.provider}, cohere calls this run={dict(emb.LIVE_CALLS)})")
    return 0


def _f(x: float) -> str:
    return f"{x:.3f}"


def write_md(res: dict) -> None:
    c = res["config"]
    lines = [
        "# E4 - retrieval and abstention", "",
        f"Corpus: {res['corpus']['n_papers']} papers, {res['corpus']['n_chunks']} chunks "
        f"(sha256 {res['corpus']['chunks_sha256'][:12]}). Gold: {res['gold']['n_answerable']} answerable + "
        f"{res['gold']['n_reworded']} reworded + {res['gold']['n_unanswerable']} unanswerable.",
        f"Embeddings: {c['embedding_provider']} ({c['embedding_model']}); rerank: {c['rerank_model']} "
        f"(used: {c['rerank_available']}). Cohere calls this run: {res['cohere_calls']['this_run']}; "
        f"ledger total: {res['cohere_calls']['cumulative_recorded_in_ledger']}.", "",
    ]
    for level, title in (("paper_level", "Paper level (primary)"), ("chunk_level", "Chunk level (secondary)")):
        n = res["modes"]["bm25"][level]["n"]
        lines += [f"## {title}, n={n} queries", "",
                  "| mode | Hit@1 [95% CI] | Hit@3 [95% CI] | Hit@5 [95% CI] | MRR | nDCG@5 | p50 latency ms |",
                  "|---|---|---|---|---|---|---|"]
        for m in MODES:
            d = res["modes"][m][level]

            def cell(k: int, d=d) -> str:
                lo, hi = d[f"hit@{k}_ci95"]
                return f"{_f(d[f'hit@{k}'])} [{_f(lo)}, {_f(hi)}]"

            lines.append(f"| {m} | {cell(1)} | {cell(3)} | {cell(5)} | {_f(d['mrr'])} | {_f(d['ndcg@5'])} | "
                         f"{res['modes'][m]['latency_ms_p50']:.1f} |")
        lines.append("")
    a = res["abstention"]
    lines += ["## Abstention", "", "| mode | accuracy on unanswerable (n) [95% CI] | false-abstention on answerable (n) "
              "[95% CI] |", "|---|---|---|"]
    for m in MODES:
        d = a[m]
        lines.append(f"| {m} | {_f(d['accuracy_on_unanswerable'])} ({d['n_unanswerable']}) "
                     f"[{_f(d['accuracy_on_unanswerable_ci95'][0])}, {_f(d['accuracy_on_unanswerable_ci95'][1])}] | "
                     f"{_f(d['false_abstention_rate_on_answerable'])} ({d['n_answerable']}) "
                     f"[{_f(d['false_abstention_rate_ci95'][0])}, {_f(d['false_abstention_rate_ci95'][1])}] |")
    lines += ["", "Paper level: relevance = the hit belongs to the gold paper; Hit@k and MRR use the ranked chunk list, "
              "nDCG@5 uses the de-duplicated paper ranking (ideal DCG = 1). Chunk level: relevance = exact gold "
              "chunk id.", "", "Thresholds are fixed a priori in configs/rag.yaml and were not tuned on this gold set. "
              "Gold questions were written by an automated assistant from chunk text and spot-checked; see "
              "eval/README_retrieval.md for limits.", ""]
    Path(ROOT / "results" / "e4.md").write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
