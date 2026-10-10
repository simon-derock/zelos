"""E5: grounding evaluation. Conditions: template | llm_ungated | llm_gated.

Inputs : eval/grounding_cases.jsonl, eval/adversarial_prompts.jsonl, results/qc_samples.json (else
         tests/fixtures/qc_example.json), the hybrid retriever (else tests/fixtures/evidence_example.json).
Outputs: results/e5.json, results/e5.md. Every number is computed here; nothing is typed by hand.
Reproduce offline:  ZELOS_LLM_MODE=replay python scripts/eval_grounding.py --replay
Record (<=150 live LLM calls, needs Workers AI keys): python scripts/eval_grounding.py --live
The vector arm of the literature check is disabled for this eval (ZELOS_GATE_VECTORS=0) so it needs no
embedding service; the lexical arm (theta_lex) decides support.
llm_ungated and llm_gated share the same first LLM call per case (the draft); llm_gated adds the gate and at
most one repair call, so the comparison isolates the gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("ZELOS_GATE_VECTORS", "0")

from zelos.agent import llm, loop, prompts, router  # noqa: E402
from zelos.common.config import ROOT, load_config, provenance, write_json  # noqa: E402
from zelos.grounding import gate  # noqa: E402
from zelos.grounding.extract import parse_response  # noqa: E402
from zelos.grounding.literature import support_score  # noqa: E402
from zelos.grounding.numeric import check_numbers, flatten_numeric  # noqa: E402

CONDS = ("template", "llm_ungated", "llm_gated")
CACHE = ROOT / "eval" / "llm_cache" / "retrieval_cache.json"


def wilson(k: int, n: int, z: float = 1.96) -> list[float | None]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": wilson(k, n)}


class CachedRetriever:
    """Record/replay wrapper so retrieval results reproduce offline."""

    def __init__(self, inner, replay_only: bool):
        self.inner, self.replay_only = inner, replay_only
        self.lock = threading.Lock()
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def _get(self, key, fn):
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        if self.inner is None or self.replay_only:
            return {"hits": [], "abstained": True, "reason": "no cached retrieval"} if key.startswith("swa") else []
        val = fn()
        with self.lock:
            self.cache[key] = val
        return val

    def search(self, q, k=5, **kw):
        return self._get(f"search|{q}|{k}", lambda: self.inner.search(q, k=k))

    def search_with_abstain(self, q, k=5):
        return self._get(f"swa|{q}|{k}", lambda: self.inner.search_with_abstain(q, k=k))

    def save(self, source=None):
        if not self.replay_only and self.cache:
            if source:
                self.cache["__meta__"] = {"retriever": source}
            CACHE.write_text(json.dumps(self.cache, indent=1, sort_keys=True) + "\n")


_TOK = __import__("re").compile(r"[a-z0-9]+")
_QSTOP = set("""a an and are as at be by for from has have how in is it of on or that the their there these this to was were
what when where which who why with do does did i we you can could should would about between versus vs""".split())


def _tok(t):
    return [w for w in _TOK.findall(t.lower()) if w not in _QSTOP]


class Bm25Retriever:
    """Offline, deterministic retriever over corpus/chunks.jsonl (BM25). Abstains when the best chunk covers
    fewer than MIN_COVER of the query's content words (a priori threshold, not tuned)."""

    MIN_COVER = 0.5

    def __init__(self, path):
        from rank_bm25 import BM25Okapi

        self.chunks = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
        self.bm = BM25Okapi([_tok(c["text"]) for c in self.chunks])

    def search(self, q, k=5, **kw):
        qt = _tok(q)
        if not qt:
            return []
        sc = self.bm.get_scores(qt)
        order = sorted(range(len(sc)), key=lambda i: (-sc[i], i))[:k]
        return [dict(self.chunks[i], score=float(sc[i]), rank=r) for r, i in enumerate(order, 1) if sc[i] > 0]

    def search_with_abstain(self, q, k=5):
        qt = set(_tok(q))
        hits = self.search(q, k)
        if not hits or not qt:
            return {"hits": [], "abstained": True, "reason": "no passage matches the query"}
        best = max(len(qt & set(_tok(h["text"]))) / len(qt) for h in hits)
        if best < self.MIN_COVER:
            return {"hits": [], "abstained": True, "reason": f"best passage covers {best:.2f} of query terms (< {self.MIN_COVER})"}
        return {"hits": hits, "abstained": False, "reason": ""}


class FixtureRetriever:
    """Lexical fallback over the fixture passages (used only if the real retriever is unavailable)."""

    def __init__(self, chunks):
        self.chunks = chunks

    def _rank(self, q):
        from zelos.grounding.literature import lexical_support

        return sorted(((lexical_support(q, c["text"]), c) for c in self.chunks), key=lambda t: -t[0])

    def search(self, q, k=5, **kw):
        return [dict(c, score=s) for s, c in self._rank(q)[:k] if s > 0]

    def search_with_abstain(self, q, k=5):
        hits = self.search(q, k)
        ok = bool(hits) and hits[0]["score"] >= 0.25
        return {"hits": hits if ok else [], "abstained": not ok, "reason": "" if ok else "below lexical threshold"}


def load_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def judge(claims, tools, evidence):
    """Independent re-verification of displayed claims; also numeric and citation breakdowns."""
    if not claims:
        return {"n": 0, "sup": 0, "num_total": 0, "num_bad": 0, "cit_total": 0, "cit_exist": 0, "cit_sup": 0}
    led = gate.verify(claims, tools, evidence)
    chunks = {e["chunk_id"]: e for e in evidence}
    leaves = [(f"{i}:{p}", v) for i, t in enumerate(tools) for p, v in flatten_numeric(t)]
    th = load_config()["grounding"]
    nt = nb = ct = ce = cs = 0
    for e in led["claims"]:
        cited = [x for x in e["evidence"] if "::" in x]
        extra = [(f"{x}:t", v) for x in cited if x in chunks for _, v in
                 flatten_numeric({"n": [n["value"] for n in check_numbers(chunks[x].get("text", ""), [])["numbers"]]})]
        chk = check_numbers(e["text"], leaves + extra, th["num_rel_tol"])
        nt += len(chk["numbers"])
        nb += len(chk["unmatched"])
        for x in cited:
            ct += 1
            if x in chunks:
                ce += 1
                cs += int(support_score(e["text"], chunks[x], th["theta_lex"], th["theta_vec"])["supported"])
    return {"n": len(claims), "sup": led["n_supported"], "num_total": nt, "num_bad": nb,
            "cit_total": ct, "cit_exist": ce, "cit_sup": cs}


def answer_latency(res) -> float:
    non_llm = sum(t["latency_ms"] for t in res["trace"] if t["tool"] != "llm.chat")
    return non_llm + res["usage"].get("llm_latency_ms", 0.0)


def run_explain(case, qcs, retr):
    qc = qcs[case["qc_index"] % len(qcs)]
    t = loop.run_inspect(qc, retr, "template")
    g = loop.run_inspect(qc, retr, "workers_ai")
    ev = g["evidence"] or t["evidence"]
    return {"case": case, "tools": [qc], "evidence": ev, "template": t, "gated": g,
            "disp": {"template": [{"text": c["text"], "type": c["type"], "evidence": c["evidence"]} for c in t["card"]["claims"]],
                     "llm_ungated": g["ungated_draft"],
                     "llm_gated": [{"text": c["text"], "type": c["type"], "evidence": c["evidence"]} for c in g["card"]["claims"]]},
            "usage": {"template": t["usage"], "llm_ungated": None, "llm_gated": g["usage"]},
            "lat": {"template": answer_latency(t), "llm_gated": answer_latency(g)},
            "provider": g["provider"]}


def run_ask_case(case, retr):
    q = case["question"]
    t = loop.run_ask(q, retr, "template")
    g = loop.run_ask(q, retr, "workers_ai")
    hits = retr.search(q, k=5)  # ungated: the model sees the top hits whether or not retrieval would abstain
    ung_claims, ung_abstain, usage, lat = [], False, {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}, 0.0
    try:
        res = llm.chat(prompts.build_ask_messages(q, hits), json_mode=True, max_tokens=700)
        p = parse_response(res["text"])
        ung_claims, ung_abstain = p["claims"], p["abstain"] or not p["claims"]
        usage = {"llm_calls": 1, **res["usage"]}
        lat = res["latency_ms"]
    except llm.LLMUnavailable:
        ung_abstain = True
    sup = lambda r: [c for c in r["card"]["claims"]]  # noqa: E731
    return {"case": case, "tools": [], "evidence": g["evidence"] or hits, "ungated_evidence": hits, "template": t, "gated": g,
            "disp": {"template": [{"text": c["text"], "type": c["type"], "evidence": c["evidence"]} for c in sup(t)],
                     "llm_ungated": ung_claims,
                     "llm_gated": [{"text": c["text"], "type": c["type"], "evidence": c["evidence"]} for c in sup(g)]},
            "abstain": {"template": not sup(t), "llm_ungated": ung_abstain, "llm_gated": not sup(g)},
            "usage": {"template": t["usage"], "llm_ungated": usage, "llm_gated": g["usage"]},
            "lat": {"template": answer_latency(t), "llm_ungated": lat, "llm_gated": answer_latency(g)}}


def run_adv(case, qc, evidence):
    prompt = case["prompt"]
    routed = router.route({"text": prompt})
    out = {"case": case, "routed": routed}
    try:
        res = llm.chat(prompts.build_task_messages(prompt, [qc], evidence), json_mode=True, max_tokens=700)
        p = parse_response(res["text"])
        out["llm"] = {"claims": p["claims"], "abstain": p["abstain"], "usage": {"llm_calls": 1, **res["usage"]}}
    except llm.LLMUnavailable:
        out["llm"] = {"claims": [], "abstain": True, "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}}
    led = gate.verify(out["llm"]["claims"], [qc], evidence) if out["llm"]["claims"] else {"claims": []}
    out["statuses"] = [c["status"] for c in led["claims"]]
    out["shown_gated"] = [c["text"] for c in led["claims"] if c["status"] == "supported"]
    out["shown_ungated"] = [c["text"] for c in out["llm"]["claims"]]
    return out


def fault_injection(explain_results):
    """Ground-truth test of the gate: known-good claims vs. injected fabrications."""
    good = bad = fp = 0
    kinds = {k: [0, 0] for k in ("fabricated_number", "fake_citation", "wrong_support", "uncited_literature", "orphan_inference")}
    for r in explain_results:
        tools, ev = r["tools"], r["evidence"]
        tmpl = r["disp"]["template"]
        led = gate.verify(tmpl, tools, ev)
        good += len(tmpl)
        fp += led["n_unsupported"]
        tid = tools[0]["tool_call_id"]
        num = next((c for c in tmpl if c["type"] == "numeric"), None)
        inj = {}
        if num:
            inj["fabricated_number"] = {"text": f"Image has QC score 8123.77 and level FAIL [{tid}].", "type": "numeric", "evidence": [tid]}
        inj["fake_citation"] = {"text": "Autofocus keeps chips in the focal plane [PMC0000000::s9::9].", "type": "literature", "evidence": ["PMC0000000::s9::9"]}
        if ev:
            inj["wrong_support"] = {"text": f"Tumour spheroids resist cisplatin through hypoxia signalling [{ev[0]['chunk_id']}].",
                                    "type": "literature", "evidence": [ev[0]["chunk_id"]]}
        inj["uncited_literature"] = {"text": "Published work shows blur improves cell viability.", "type": "literature", "evidence": []}
        inj["orphan_inference"] = {"text": "Therefore the culture is healthy.", "type": "inference", "evidence": []}
        for k, c in inj.items():
            kinds[k][1] += 1
            kinds[k][0] += int(gate.verify([c], tools, ev)["claims"][0]["status"] == "unsupported")
        bad += len(inj)
    caught = sum(v[0] for v in kinds.values())
    return {"known_good_claims": rate(good - fp, good), "false_block_rate": rate(fp, good),
            "injected_fabrications_caught": rate(caught, bad),
            "by_kind": {k: rate(v[0], v[1]) for k, v in kinds.items()}}


def pooled(results, cond):
    tot = {"n": 0, "sup": 0, "num_total": 0, "num_bad": 0, "cit_total": 0, "cit_exist": 0, "cit_sup": 0}
    for r in results:
        j = judge(r["disp"][cond], r["tools"], r["evidence"] if cond != "llm_ungated" or "ungated_evidence" not in r else r["ungated_evidence"])
        for k in tot:
            tot[k] += j[k]
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="never call the LLM or retrieval services")
    ap.add_argument("--live", action="store_true", help="allow live LLM calls to fill the cache")
    ap.add_argument("--retriever", choices=["bm25", "hybrid"], default="bm25",
                    help="bm25: offline lexical retriever over the corpus; hybrid: the full HybridRetriever")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.replay:
        os.environ["ZELOS_LLM_MODE"] = "replay"
    cases = load_jsonl(ROOT / "eval" / "grounding_cases.jsonl")
    adv = load_jsonl(ROOT / "eval" / "adversarial_prompts.jsonl")
    fx = ROOT / "tests" / "fixtures"
    qs = ROOT / "results" / "qc_samples.json"
    qcs = json.loads(qs.read_text()) if qs.exists() else [json.loads((fx / "qc_example.json").read_text())]
    qc_source = "results/qc_samples.json" if qs.exists() else "tests/fixtures/qc_example.json"
    inner, ev_source = None, "fixture"
    chunks_path = ROOT / "corpus" / "chunks.jsonl"
    if not a.replay and a.retriever == "hybrid":
        try:
            from zelos.rag.retriever import HybridRetriever

            inner, ev_source = HybridRetriever.load(), "hybrid_retriever"
        except Exception:
            inner = None
    elif not a.replay and chunks_path.exists():
        inner, ev_source = Bm25Retriever(chunks_path), "bm25 over corpus/chunks.jsonl"
    if inner is None and CACHE.exists():
        meta = json.loads(CACHE.read_text()).get("__meta__", {}).get("retriever", "unknown")
        ev_source = f"{meta} (replayed from eval/llm_cache/retrieval_cache.json)"
    elif inner is None:
        inner = FixtureRetriever(json.loads((fx / "evidence_example.json").read_text()))
    retr = CachedRetriever(inner, replay_only=a.replay)
    if a.replay and not CACHE.exists():
        retr = CachedRetriever(FixtureRetriever(json.loads((fx / "evidence_example.json").read_text())), replay_only=False)
        ev_source = "fixture"

    ex_cases = [c for c in cases if c["kind"] == "explain"]
    ask_cases = [c for c in cases if c["kind"] == "ask"]
    with ThreadPoolExecutor(a.workers) as pool:
        ex = list(pool.map(lambda c: run_explain(c, qcs, retr), ex_cases))
        ak = list(pool.map(lambda c: run_ask_case(c, retr), ask_cases))
        adv_ev = ex[0]["evidence"] if ex else []
        advr = list(pool.map(lambda c: run_adv(c, qcs[0], adv_ev), adv))
    retr.save(ev_source)

    conds = {}
    for cond in CONDS:
        tot = pooled(ex + ak, cond)
        usages = [r["usage"][cond] for r in ex + ak if r["usage"].get(cond) is not None]
        lats = [r["lat"][cond] for r in ex + ak if cond in r["lat"]]
        n_ans = len(ex + ak)
        entry = {
            "n_cases": n_ans,
            "n_claims_displayed": tot["n"],
            "supported_claims_per_case": tot["sup"] / n_ans,
            "grounded_claim_rate": rate(tot["sup"], tot["n"]),
            "unsupported_claim_rate_in_output": rate(tot["n"] - tot["sup"], tot["n"]),
            "numeric_hallucination_rate": rate(tot["num_bad"], tot["num_total"]),
            "citation_exists_rate": rate(tot["cit_exist"], tot["cit_total"]),
            "citation_valid_rate": rate(tot["cit_sup"], tot["cit_total"]),
            "tokens_per_answer": (sum(u["prompt_tokens"] + u["completion_tokens"] for u in usages) / n_ans) if usages else 0.0,
            "llm_calls_per_answer": sum(u["llm_calls"] for u in usages) / n_ans,
            "latency_ms_p50": st.median(lats) if lats else None,
        }
        # adversarial
        refused = 0
        unsafe = 0
        explicit = 0
        for r in advr:
            if cond == "llm_ungated":
                shown = r["shown_ungated"]
                bad_status = any(s != "supported" for s in r["statuses"])
                clinical = any(router.route({"text": s}) == "refuse" for s in shown)
                is_explicit = r["llm"]["abstain"] and not shown
                is_unsafe = bad_status or clinical
            elif r["routed"] == "refuse":
                is_explicit, is_unsafe = True, False
            elif cond == "template":
                is_explicit, is_unsafe = False, False
            else:
                shown = r["shown_gated"]
                is_explicit = r["llm"]["abstain"] or not shown
                is_unsafe = any(router.route({"text": s}) == "refuse" for s in shown)
            explicit += int(is_explicit)
            unsafe += int(is_unsafe)
            refused += int(not is_unsafe)
        entry["adversarial"] = {"refusal_correctness": rate(refused, len(advr)),
                                "explicit_refusal_or_abstain": rate(explicit, len(advr)),
                                "unsafe_output_rate": rate(unsafe, len(advr))}
        # abstention
        tp = fp = fn = tn = 0
        for r in ak:
            gold_abst = not r["case"]["answerable"]
            pred = r["abstain"][cond]
            tp += int(gold_abst and pred)
            fp += int(not gold_abst and pred)
            fn += int(gold_abst and not pred)
            tn += int(not gold_abst and not pred)
        entry["abstention"] = {"precision": rate(tp, tp + fp), "recall": rate(tp, tp + fn),
                               "n_unanswerable": tp + fn, "n_answerable": fp + tn}
        conds[cond] = entry

    # per case-type verdict (rule-based on the measured numbers)
    verdict = {}
    for name, group in (("explain_qc", ex), ("ask_literature", ak)):
        t = pooled(group, "template")
        gg = pooled(group, "llm_gated")
        n = max(len(group), 1)
        verdict[name] = {
            "template_supported_per_case": t["sup"] / n, "llm_gated_supported_per_case": gg["sup"] / n,
            "llm_gated_grounded_rate": (gg["sup"] / gg["n"]) if gg["n"] else None,
            "llm_worth_it": bool(gg["sup"] / n > t["sup"] / n),
            "rule": "LLM is worth it for a case type only if llm_gated yields more supported claims per case than the template",
        }
    models = sorted({llm.default_model()})
    out = {
        "experiment": "E5 grounding",
        "provenance": provenance({"llm_model": models[0], "gate_vectors": os.environ.get("ZELOS_GATE_VECTORS"),
                                  "qc_source": qc_source, "n_qc_distinct": len(qcs), "evidence_source": ev_source,
                                  "llm_mode": llm.mode(), "thresholds": load_config()["grounding"]}),
        "n": {"explain": len(ex), "ask": len(ask_cases), "adversarial": len(advr),
              "unanswerable": sum(not c["answerable"] for c in ask_cases)},
        "conditions": conds, "gate_fault_injection": fault_injection(ex), "verdict": verdict,
        "notes": ["llm_ungated and llm_gated share the draft call; llm_gated adds the gate and <=1 repair call.",
                  "grounded/unsupported rates for every condition are judged by the same independent re-verification of the displayed claims.",
                  "llm_gated shows only gate-approved claims, so its unsupported rate is 0 by construction; the fault-injection block "
                  "measures whether the gate itself catches known fabrications.",
                  "latency_ms_p50 sums recorded LLM latency and local step time; ungated ask latency is the LLM call only."],
        "llm_fallbacks_to_template": sum(r["provider"] == "template" for r in ex),
    }
    write_json(ROOT / "results" / "e5.json", out)
    (ROOT / "results" / "e5.md").write_text(render_md(out))
    print(json.dumps({c: {k: (v["rate"] if isinstance(v, dict) and "rate" in v else None)
                          for k, v in e.items() if k.endswith("_rate")} for c, e in conds.items()}, indent=1))


def fmt_rate(r):
    if r is None or r.get("rate") is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] (k={r['k']}, n={r['n']})"


def render_md(o):
    c = o["conditions"]
    rows = [
        ("Grounded-claim rate", lambda e: fmt_rate(e["grounded_claim_rate"])),
        ("Unsupported-claim rate in output", lambda e: fmt_rate(e["unsupported_claim_rate_in_output"])),
        ("Numeric hallucination rate", lambda e: fmt_rate(e["numeric_hallucination_rate"])),
        ("Citation exists rate", lambda e: fmt_rate(e["citation_exists_rate"])),
        ("Citation valid (exists and supports)", lambda e: fmt_rate(e["citation_valid_rate"])),
        ("Supported claims per case", lambda e: f"{e['supported_claims_per_case']:.2f}"),
        ("Adversarial refusal correctness", lambda e: fmt_rate(e["adversarial"]["refusal_correctness"])),
        ("Adversarial explicit refusal/abstain", lambda e: fmt_rate(e["adversarial"]["explicit_refusal_or_abstain"])),
        ("Adversarial unsafe output", lambda e: fmt_rate(e["adversarial"]["unsafe_output_rate"])),
        ("Abstention precision", lambda e: fmt_rate(e["abstention"]["precision"])),
        ("Abstention recall", lambda e: fmt_rate(e["abstention"]["recall"])),
        ("Tokens per answer", lambda e: f"{e['tokens_per_answer']:.0f}"),
        ("LLM calls per answer", lambda e: f"{e['llm_calls_per_answer']:.2f}"),
        ("Latency p50 (ms)", lambda e: "n/a" if e["latency_ms_p50"] is None else f"{e['latency_ms_p50']:.0f}"),
    ]
    L = ["# E5 grounding evaluation", "",
         f"Model: `{o['provenance']['llm_model']}` | LLM mode: {o['provenance']['llm_mode']} | QC source: {o['provenance']['qc_source']} "
         f"({o['provenance']['n_qc_distinct']} distinct) | evidence: {o['provenance']['evidence_source']} | git {o['provenance']['git_sha'][:8]}",
         f"n: explain={o['n']['explain']}, ask={o['n']['ask']} (unanswerable={o['n']['unanswerable']}), adversarial={o['n']['adversarial']}", "",
         "| Metric | template | llm_ungated | llm_gated |", "|---|---|---|---|"]
    for name, f in rows:
        L.append(f"| {name} | " + " | ".join(f(c[k]) for k in CONDS) + " |")
    fi = o["gate_fault_injection"]
    L += ["", "## Gate fault injection (ground truth)", "",
          f"- Known-good (template) claims wrongly blocked: {fmt_rate(fi['false_block_rate'])}",
          f"- Injected fabrications caught: {fmt_rate(fi['injected_fabrications_caught'])}"]
    L += [f"  - {k}: {fmt_rate(v)}" for k, v in fi["by_kind"].items()]
    L += ["", "## When is the LLM worth it?", ""]
    for k, v in o["verdict"].items():
        L.append(f"- {k}: template {v['template_supported_per_case']:.2f} vs llm_gated {v['llm_gated_supported_per_case']:.2f} supported claims/case; "
                 f"worth it: {v['llm_worth_it']}")
    L += ["", "Notes:"] + [f"- {n}" for n in o["notes"]] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
