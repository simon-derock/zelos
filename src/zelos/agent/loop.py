"""Zelos agent loop: the LLM proposes, tools compute, the gate disposes.

Public API (all work offline with retriever=None and provider="template"):
  run_inspect(qc, retriever=None, provider="workers_ai") -> dict
  run_ask(question, retriever=None, provider="workers_ai") -> dict
  run_triage(qc_list, retriever=None, provider="workers_ai") -> dict
  run_certify(qc, retriever=None, provider="template") -> dict
Each returns card, ledger, ungated_draft, trace, usage (+ provider actually used). At most 2 LLM calls per
call: one explain call and one repair call. If the LLM is unavailable the loop falls back to the template.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter

from zelos.agent import llm, prompts, router, template
from zelos.common.config import load_config, provenance
from zelos.grounding import gate
from zelos.grounding import ledger as L
from zelos.grounding.extract import parse_response

DISCLAIMER = "Research tool. Not a medical device. No clinical or diagnostic claims."
GENERIC_QUERY = "brightfield microscopy image quality control organ-on-a-chip cell culture"


def result_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


class _Run:
    def __init__(self) -> None:
        self.trace: list[dict] = []
        self.usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.notes: list[str] = []

    def step(self, step: str, tool: str, args: dict, result, t0: float, tokens: int = 0) -> None:
        self.trace.append({"step": step, "tool": tool, "args": args, "result_hash": result_hash(result),
                           "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2), "tokens": tokens})

    def call(self, step: str, messages: list[dict]) -> dict:
        t0 = time.perf_counter()
        res = llm.chat(messages, json_mode=True, max_tokens=700,
                       temperature=load_config()["agent"].get("temperature", 0))
        u = res["usage"]
        self.usage["llm_calls"] += 1
        self.usage["prompt_tokens"] += u["prompt_tokens"]
        self.usage["completion_tokens"] += u["completion_tokens"]
        self.step(step, "llm.chat", {"model": res["model"], "cached": res["cached"]}, res["text"], t0,
                  u["prompt_tokens"] + u["completion_tokens"])
        return res


def _search(retriever, query: str, k: int, run: _Run) -> list[dict]:
    if retriever is None:
        return []
    t0 = time.perf_counter()
    try:
        hits = retriever.search(query, k=k)
    except Exception as e:  # noqa: BLE001 - retrieval is optional
        run.notes.append(f"retrieval failed: {type(e).__name__}")
        hits = []
    run.step("retrieve", "retrieve_evidence", {"query": query, "k": k}, [h.get("chunk_id") for h in hits], t0)
    return hits


def _merge(hit_lists: list[list[dict]], k: int) -> list[dict]:
    seen: dict[str, dict] = {}
    for hits in hit_lists:
        for h in hits:
            cid = h.get("chunk_id")
            if cid and (cid not in seen or h.get("score", 0) > seen[cid].get("score", 0)):
                seen[cid] = h
    return sorted(seen.values(), key=lambda h: -float(h.get("score", 0)))[:k]


def _evidence_for_qc(qc: dict, retriever, run: _Run, k: int = 5) -> list[dict]:
    if retriever is None:
        return []
    labels = [c["label"] for c in qc.get("causes", [])][:2]
    try:
        from zelos.rag.query_builder import queries_for_causes

        queries = list(queries_for_causes(labels)) if labels else []
    except Exception:  # noqa: BLE001
        queries = [f"{lab.replace('_', ' ').lower()} brightfield microscopy" for lab in labels]
    queries = queries[:2] or [GENERIC_QUERY]
    return _merge([_search(retriever, q, 3, run) for q in queries], k)


def _excerpt(text: str, n: int = 25) -> str:
    w = str(text or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def _evidence_card(evidence: list[dict], cited: set[str]) -> list[dict]:
    return [{"chunk_id": e["chunk_id"], "title": e.get("title", ""), "url": e.get("url", ""),
             "license": e.get("license", ""), "excerpt": _excerpt(e.get("text", "")), "cited": e["chunk_id"] in cited}
            for e in evidence]


def _gated(tool_outputs: list[dict], evidence: list[dict], provider: str, run: _Run, template_fn, msgs_fn,
           cfg: dict | None = None) -> dict:
    """Draft -> verify -> (<=1 repair) -> final. Returns draft, ledger, final claims, blocked, provider_used."""
    provider_used, abstain, reason = provider, False, ""
    if provider == "template":
        draft = template_fn()
    else:
        try:
            res = run.call("explain", msgs_fn())
            parsed = parse_response(res["text"])
            draft, abstain, reason = parsed["claims"], parsed["abstain"], parsed["reason"]
        except llm.LLMUnavailable as e:
            run.notes.append(f"LLM unavailable ({e}); used template explainer")
            provider_used, draft = "template", template_fn()
    if abstain:
        led = {"claims": [L.entry("c1", f"ABSTAIN: {reason}", "inference", [], "abstain", reason or "model abstained")],
               "n_supported": 0, "n_unsupported": 0, "grounded_rate": None}
        return {"draft": draft, "ledger": led, "final_claims": [], "blocked": [], "provider": provider_used,
                "abstained": True, "reason": reason}
    t0 = time.perf_counter()
    led = gate.verify(draft, tool_outputs, evidence, cfg)
    run.step("gate", "check_claims", {"n_claims": len(draft)}, led["claims"], t0)
    blocked: list[dict] = [e for e in led["claims"] if e["status"] == "unsupported"]
    final_claims = [{"text": e["text"], "type": e["type"], "evidence": e["evidence"]}
                    for e in led["claims"] if e["status"] == "supported"]
    max_calls = int(load_config()["agent"].get("max_llm_calls", 2))
    if blocked and provider_used != "template" and run.usage["llm_calls"] < max_calls:
        t0 = time.perf_counter()
        rep = gate.repair(draft, led, lambda m: run.call("repair", m), rounds=load_config()["grounding"]["repair_rounds"],
                          tool_outputs=tool_outputs, evidence=evidence, cfg=cfg)
        run.step("repair", "gate.repair", {"n_blocked": len(blocked)}, rep["ledger"]["claims"], t0)
        led, final_claims = rep["ledger"], rep["claims"]
        final_claims = [{"text": e["text"], "type": e["type"], "evidence": e["evidence"]}
                        for e in led["claims"] if e["status"] == "supported"]
        blocked = [e for e in led["claims"] if e["status"] == "unsupported"]
    uniq, seen = [], set()
    for b in blocked:
        if b["text"] not in seen:
            seen.add(b["text"])
            uniq.append(b)
    return {"draft": draft, "ledger": led, "final_claims": final_claims, "blocked": uniq, "provider": provider_used,
            "abstained": False, "reason": ""}


def _cited(claims: list[dict]) -> set[str]:
    return {x for c in claims for x in c.get("evidence", []) if "::" in x}


def _finish(mode: str, card: dict, g: dict, run: _Run, evidence: list[dict]) -> dict:
    cited = _cited(g["final_claims"])
    card["claims"] = [{"claim_id": e["claim_id"], "text": e["text"], "type": e["type"], "evidence": e["evidence"]}
                      for e in g["ledger"]["claims"] if e["status"] == "supported"]
    card["blocked_claims"] = [{"text": b["text"], "reason": b["reason"]} for b in g["blocked"]]
    card["evidence"] = _evidence_card(evidence, cited)
    card["disclaimer"] = DISCLAIMER
    if g["abstained"]:
        card["abstain"] = {"abstained": True, "reason": g["reason"]}
    return {"mode": mode, "provider": g["provider"], "card": card, "ledger": g["ledger"],
            "ungated_draft": g["draft"], "trace": run.trace, "usage": dict(run.usage), "notes": run.notes}


def run_inspect(qc: dict, retriever=None, provider: str = "workers_ai") -> dict:
    run = _Run()
    t0 = time.perf_counter()
    run.step("qc", "inspect_image", {"image_id": (qc.get("image") or {}).get("id")}, qc, t0)
    evidence = _evidence_for_qc(qc, retriever, run)
    g = _gated([qc], evidence, provider, run, lambda: template.explain_qc(qc, evidence),
               lambda: prompts.build_explain_messages(qc, evidence))
    primary = (qc.get("causes") or [None])[0]
    card = {"image_id": (qc.get("image") or {}).get("id"), "tool_call_id": qc.get("tool_call_id"),
            "level": qc.get("level"), "qc_score": qc.get("qc_score"), "density_state": qc.get("density_state"),
            "primary_cause": ({"label": primary["label"], "note": primary.get("note"), "action": primary.get("action"),
                               "metric": primary.get("metric"), "value": primary.get("value"), "z": primary.get("z")}
                              if primary else None)}
    return _finish("inspect", card, g, run, evidence)


def run_ask(question: str, retriever=None, provider: str = "workers_ai") -> dict:
    run = _Run()
    base = {"question": question}
    if router.route({"question": question}) == "refuse":
        led = {"claims": [L.entry("c1", "Request refused: outside the research scope of this tool", "inference", [],
                                  "refused", "diagnosis, clinical, fabrication or injection request")],
               "n_supported": 0, "n_unsupported": 0, "grounded_rate": None}
        g = {"draft": [], "ledger": led, "final_claims": [], "blocked": [], "provider": provider,
             "abstained": False, "reason": ""}
        out = _finish("ask", {**base, "refused": True}, g, run, [])
        return out
    evidence: list[dict] = []
    reason = "no retriever configured, so no evidence is available"
    abstained = retriever is None
    if retriever is not None:
        t0 = time.perf_counter()
        try:
            if hasattr(retriever, "search_with_abstain"):
                r = retriever.search_with_abstain(question, k=5)
                evidence, abstained, reason = list(r.get("hits", [])), bool(r.get("abstained")), r.get("reason", "")
            else:
                evidence = list(retriever.search(question, k=5))
                abstained = not evidence
                reason = "no passages retrieved"
        except Exception as e:  # noqa: BLE001
            abstained, reason = True, f"retrieval failed: {type(e).__name__}"
        run.step("retrieve", "retrieve_evidence", {"query": question, "k": 5}, [h.get("chunk_id") for h in evidence], t0)
    if abstained or not evidence:
        led = {"claims": [L.entry("c1", f"ABSTAIN: {reason}", "inference", [], "abstain", reason)],
               "n_supported": 0, "n_unsupported": 0, "grounded_rate": None}
        g = {"draft": [], "ledger": led, "final_claims": [], "blocked": [], "provider": provider,
             "abstained": True, "reason": reason}
        return _finish("ask", dict(base), g, run, [])
    g = _gated([], evidence, provider, run, lambda: template.explain_ask(question, evidence),
               lambda: prompts.build_ask_messages(question, evidence))
    return _finish("ask", dict(base), g, run, evidence)


def triage_stats(qc_list: list[dict]) -> dict:
    """Deterministic session statistics (a tool output with its own tool_call_id)."""
    ids = [(q.get("image") or {}).get("id", "?") for q in qc_list]
    counts = Counter(q.get("level") for q in qc_list)
    causes = Counter(c["label"] for q in qc_list for c in q.get("causes", []))
    scores = [float(q["qc_score"]) for q in qc_list if q.get("qc_score") is not None]
    reimage = []
    for q in sorted(qc_list, key=lambda q: -float(q.get("qc_score") or 0)):
        if q.get("level") in ("FAIL", "REVIEW"):
            top = (q.get("causes") or [{}])[0]
            reimage.append({"image_id": (q.get("image") or {}).get("id"), "level": q["level"],
                            "qc_score": q.get("qc_score"), "cause": top.get("label"), "action": top.get("action")})
    return {"tool_call_id": "triage-" + result_hash(ids)[:8], "n": len(qc_list),
            "counts": {k: counts.get(k, 0) for k in ("PASS", "REVIEW", "FAIL")},
            "mean_qc_score": round(sum(scores) / len(scores), 3) if scores else None,
            "dominant_causes": [{"label": k, "count": v} for k, v in causes.most_common(3)],
            "reimage": reimage}


def run_triage(qc_list: list[dict], retriever=None, provider: str = "workers_ai") -> dict:
    run = _Run()
    t0 = time.perf_counter()
    stats = triage_stats(qc_list)
    run.step("stats", "triage_folder", {"n": len(qc_list)}, stats, t0)
    evidence: list[dict] = []
    if retriever is not None and stats["dominant_causes"]:
        evidence = _evidence_for_qc({"causes": [{"label": stats["dominant_causes"][0]["label"]}]}, retriever, run)
    g = _gated([stats], evidence, provider, run, lambda: template.explain_triage(stats, evidence),
               lambda: prompts.build_triage_messages(stats, evidence))
    card = {"n": stats["n"], "counts": stats["counts"], "mean_qc_score": stats["mean_qc_score"],
            "dominant_causes": stats["dominant_causes"], "reimage": stats["reimage"], "tool_call_id": stats["tool_call_id"]}
    out = _finish("triage", card, g, run, evidence)
    out["stats"] = stats
    return out


def run_certify(qc: dict, retriever=None, provider: str = "template") -> dict:
    """Inspect, then emit a content-hashed certificate (hash of canonical JSON; not a cryptographic signature)."""
    res = run_inspect(qc, retriever, provider)
    body = {"schema": "zelos-certificate/1", "image": qc.get("image"), "level": qc.get("level"),
            "qc_score": qc.get("qc_score"), "causes": [c["label"] for c in qc.get("causes", [])],
            "qc_tool_call_id": qc.get("tool_call_id"), "ledger_sha256": result_hash(res["ledger"]),
            "n_supported": res["ledger"]["n_supported"], "n_blocked": len(res["card"]["blocked_claims"]),
            "evidence_ids": [e["chunk_id"] for e in res["card"]["evidence"] if e["cited"]],
            "disclaimer": DISCLAIMER, "provenance": provenance()}
    cert = {**body, "certificate_sha256": result_hash({k: v for k, v in body.items() if k != "provenance"})}
    return {**res, "mode": "certify", "certificate": cert}


def run_request(request: dict, retriever=None, provider: str = "workers_ai") -> dict:
    """Route a request dict and dispatch. Refusals return a ledger entry with status 'refused'."""
    kind = router.route(request)
    if kind == "refuse":
        return run_ask(str(request.get("question") or request.get("text") or "refused request"), retriever, provider)
    if kind == "inspect":
        return run_inspect(request.get("qc") or request["image"], retriever, provider)
    if kind == "certify":
        return run_certify(request["qc"], retriever, provider)
    if kind == "triage":
        return run_triage(request.get("qc_list") or request.get("images") or [], retriever, provider)
    return run_ask(str(request.get("question") or request.get("text") or ""), retriever, provider)
