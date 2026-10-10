"""E5 model sweep: which LLM is worth using behind the grounding gate?

Inputs : the E5 harness (scripts/eval_grounding.py): same cases (eval/grounding_cases.jsonl), prompts
         (zelos.agent.prompts), retriever (offline Bm25Retriever, replayed from eval/llm_cache/retrieval_cache.json
         when a query was cached) and gate (zelos.grounding.gate). Only the model behind zelos.agent.llm changes.
Outputs: results/e5_sweep.json, results/e5_sweep.md. Every number is computed here; nothing is typed by hand.
Reproduce offline:  python scripts/eval_llm_sweep.py --replay      (uses eval/llm_cache; needs no keys)
Record live:        python scripts/eval_llm_sweep.py               (Workers AI + Cohere keys in .env)

Live-run safety: the first --probe-n cases of a model run live, the observed neurons per case are extrapolated
to the remaining cases, and the full set runs only if the projection fits the remaining neuron budget. A
quota error stops that model and is recorded as quota_exhausted (no cell is ever filled by hand). Cohere calls
are capped with llm.set_call_budget.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ZELOS_GATE_VECTORS", "0")

import eval_grounding as eg  # noqa: E402  (the E5 harness; imported, not duplicated)

from zelos.agent import llm  # noqa: E402
from zelos.common.config import ROOT, load_config, provenance, secret, write_json  # noqa: E402
from zelos.grounding.extract import parse_response  # noqa: E402

NEURON_USD = 0.011 / 1000.0  # Workers AI: $0.011 per 1,000 neurons; free allowance 10,000 neurons/day

# label -> run profile. `extra` / `max_tokens` are what a reasoning model needs to answer inside the same strict-JSON
# prompt (probed once: gemma-4 and gpt-oss return content=null and finish_reason=length at the production
# max_tokens=700 with default reasoning). Non-reasoning models run with the production settings unchanged.
MODELS = [
    {"id": "llama-3.1-8b-fast", "provider": "workers_ai", "model": "@cf/meta/llama-3.1-8b-instruct-fast",
     "max_tokens": None, "extra": None,
     "price": {"in": 0.045, "out": 0.384, "neurons_in": 4119, "neurons_out": 34868,
               "source": "Workers AI pricing page, row @cf/meta/llama-3.1-8b-instruct-fp8-fast "
                         "(the -fast id is not returned by the models search catalog; same price assumed)"}},
    {"id": "gemma-4-26b-a4b", "provider": "workers_ai", "model": "@cf/google/gemma-4-26b-a4b-it",
     "max_tokens": None, "extra": {"chat_template_kwargs": {"enable_thinking": False}},
     "price": {"in": 0.10, "out": 0.30, "neurons_in": 9091, "neurons_out": 27273, "source": "Workers AI models catalog"}},
    {"id": "gpt-oss-120b", "provider": "workers_ai", "model": "@cf/openai/gpt-oss-120b",
     "max_tokens": 1500, "extra": {"reasoning_effort": "low"},
     "price": {"in": 0.35, "out": 0.75, "neurons_in": 31818, "neurons_out": 68182, "source": "Workers AI models catalog"}},
    {"id": "qwen3.8-27b", "provider": "workers_ai", "model": "@cf/qwen/qwen3.8-27b",
     "max_tokens": None, "extra": {"chat_template_kwargs": {"enable_thinking": False}},
     "price": {"in": 0.45, "out": 3.20, "neurons_in": 40909, "neurons_out": 290909, "source": "Workers AI models catalog"}},
    {"id": "command-a-03-2025", "provider": "cohere", "model": "command-a-03-2025",
     "max_tokens": None, "extra": None,
     "price": {"in": None, "out": None, "neurons_in": None, "neurons_out": None,
               "source": "Cohere trial key (no per-token charge; production price not used)"}},
]


# ---- statistics ---------------------------------------------------------------------------------------------------
def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# ---- call recording shim --------------------------------------------------------------------------------------------
class Recorder:
    """Wraps llm.chat to record every call (also cache replays) per thread, so per-answer format/latency can be
    attributed without touching the agent loop."""

    def __init__(self):
        self.tl = threading.local()
        self.orig = llm.chat
        self.lock = threading.Lock()
        self.live_neurons = 0.0
        self.live_calls = 0

    def install(self):
        llm.chat = self._chat

    def uninstall(self):
        llm.chat = self.orig

    def begin(self):
        self.tl.calls = []

    def calls(self) -> list[dict]:
        return list(getattr(self.tl, "calls", []))

    def _chat(self, messages, **kw):
        rec = {"cached": None, "format": None, "empty": False, "truncated": False, "error": None, "neurons": None}
        try:
            res = self.orig(messages, **kw)
        except llm.LLMQuotaExceeded as e:
            rec["error"] = "quota_exhausted"
            self.tl.calls.append(rec)
            raise e
        except llm.LLMUnavailable as e:
            rec["error"] = type(e).__name__
            self.tl.calls.append(rec)
            raise
        text = res.get("text") or ""
        meta = res.get("meta") or {}
        p = parse_response(text)
        rec.update(cached=bool(res.get("cached")), format=p["format"], empty=not text.strip(),
                   truncated=meta.get("finish_reason") == "length", neurons=meta.get("neurons"),
                   prompt_tokens=res["usage"]["prompt_tokens"], completion_tokens=res["usage"]["completion_tokens"])
        self.tl.calls.append(rec)
        if not rec["cached"]:
            with self.lock:
                self.live_calls += 1
                self.live_neurons += float(rec["neurons"] or 0.0)
        return res


# ---- per-model run ---------------------------------------------------------------------------------------------------
class StopModel(Exception):
    pass


def run_cases(rec: Recorder, prof: dict, cases: list[dict], kind: str, qcs, retr, workers: int) -> tuple[list[dict], dict]:
    """Run cases (explain|ask) for one model profile. Returns (per-case results, run info)."""
    info = {"quota_exhausted": False, "unavailable": 0, "stopped_at": None}
    done: list[dict] = []

    def one(case):
        rec.begin()
        r = eg.run_explain(case, qcs, retr) if kind == "explain" else eg.run_ask_case(case, retr)
        r["calls"] = rec.calls()
        return r

    with llm.use(prof["provider"], prof["model"], prof["max_tokens"], prof["extra"]):
        with ThreadPoolExecutor(workers) as pool:
            for r in pool.map(one, cases):
                errs = {c["error"] for c in r["calls"] if c["error"]}
                if "quota_exhausted" in errs:
                    info["quota_exhausted"] = True
                    info["stopped_at"] = r["case"]["id"]
                    continue  # discard: the answer fell back to the template, it is not a model result
                if errs:
                    info["unavailable"] += 1
                    continue
                done.append(r)
    return done, info


def summarize(rs: list[dict], price: dict | None, kind: str) -> dict:
    """All metrics for one model and one case type, computed from per-case harness results."""
    n = len(rs)
    if n == 0:
        return {"n_cases": 0}
    draft = eg.pooled(rs, "llm_ungated")
    gated = eg.pooled(rs, "llm_gated")
    calls = [c for r in rs for c in r["calls"]]
    n_calls = len(calls)
    fmt_bad = sum(1 for c in calls if c["format"] not in ("json", "abstain"))
    toks_in = [r["usage"]["llm_gated"]["prompt_tokens"] for r in rs]
    toks_out = [r["usage"]["llm_gated"]["completion_tokens"] for r in rs]
    lat = [r["lat"]["llm_gated"] for r in rs]
    calls_per = [r["usage"]["llm_gated"]["llm_calls"] for r in rs]
    out = {
        "n_cases": n,
        "draft": {
            "n_claims": draft["n"],
            "unsupported_claim_rate": eg.rate(draft["n"] - draft["sup"], draft["n"]),
            "valid_citation_rate": eg.rate(draft["cit_sup"], draft["cit_total"]),
            "numeric_hallucination_rate": eg.rate(draft["num_bad"], draft["num_total"]),
            "supported_claims_per_case": draft["sup"] / n,
        },
        "gated": {
            "n_claims": gated["n"],
            "unsupported_claim_rate": eg.rate(gated["n"] - gated["sup"], gated["n"]),
            "valid_citation_rate": eg.rate(gated["cit_sup"], gated["cit_total"]),
            "numeric_hallucination_rate": eg.rate(gated["num_bad"], gated["num_total"]),
            "supported_claims_per_case": gated["sup"] / n,
            "supported_claims_per_case_sd": (sum((x - gated["sup"] / n) ** 2 for x in
                                                 [eg.judge(r["disp"]["llm_gated"], r["tools"], r["evidence"])["sup"] for r in rs]) / n) ** 0.5,
            "blocked_claims_per_case": (draft["n"] - draft["sup"]) / n,
        },
        "json_format_failure_rate": eg.rate(fmt_bad, n_calls),
        "truncated_responses": sum(c["truncated"] for c in calls),
        "empty_responses": sum(c["empty"] for c in calls),
        "llm_calls_recorded": n_calls,
        "tokens_per_answer": {"prompt": mean(toks_in), "completion": mean(toks_out), "total": mean(toks_in) + mean(toks_out)},
        "llm_calls_per_answer": mean(calls_per),
        "cases_answered_by_template_fallback": sum(r["provider"] == "template" for r in rs) if kind == "explain" else None,
        "cases_reaching_llm": sum(1 for r in rs if r["usage"]["llm_gated"]["llm_calls"] > 0),
        "latency_ms": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95)},
    }
    if price and price.get("in") is not None:
        usd = [(a * price["in"] + b * price["out"]) / 1e6 for a, b in zip(toks_in, toks_out, strict=True)]
        neu = [(a * price["neurons_in"] + b * price["neurons_out"]) / 1e6 for a, b in zip(toks_in, toks_out, strict=True)]
        out["cost"] = {"usd_per_100_answers": 100 * mean(usd), "neurons_per_100_answers": 100 * mean(neu),
                       "basis": "mean prompt/completion tokens per answer x catalog per-M-token price"}
    else:
        out["cost"] = {"usd_per_100_answers": None, "neurons_per_100_answers": None, "basis": price["source"] if price else "n/a"}
    return out


def verdict(models: dict, group: str, n_full: int) -> dict:
    """'best supported-claims per case at zero unsupported after the gate; cheapest within 5% of best'."""
    cands = []
    for mid, m in models.items():
        s = (m.get("groups") or {}).get(group)
        if not s or s.get("n_cases", 0) < n_full:
            continue
        g = s["gated"]
        cands.append({"id": mid, "per_case": g["supported_claims_per_case"],
                      "unsupported_k": g["unsupported_claim_rate"]["k"], "usd": s["cost"]["usd_per_100_answers"],
                      "n": s["n_cases"]})
    zero = [c for c in cands if c["unsupported_k"] == 0]
    if not zero:
        return {"group": group, "n_full": n_full, "best": None, "cheapest_within_5pct": None, "candidates": cands,
                "text": f"{group}: no model with zero unsupported claims after the gate on all {n_full} cases"}
    best = max(zero, key=lambda c: (c["per_case"], -(c["usd"] if c["usd"] is not None else 1e9)))
    near = [c for c in zero if c["per_case"] >= 0.95 * best["per_case"]]
    priced = [c for c in near if c["usd"] is not None]
    cheap = min(priced, key=lambda c: c["usd"]) if priced else None
    text = (f"{group}: best supported claims per case at zero unsupported after the gate = {best['id']} "
            f"({best['per_case']:.2f}/case, n={best['n']}).")
    if cheap:
        text += (f" Cheapest within 5% of best = {cheap['id']} ({cheap['per_case']:.2f}/case, "
                 f"${cheap['usd']:.4f} per 100 answers).")
    else:
        text += " No priced model within 5% of best."
    return {"group": group, "n_full": n_full, "best": best["id"], "cheapest_within_5pct": cheap["id"] if cheap else None,
            "candidates": cands, "text": text}


# ---- main -------------------------------------------------------------------------------------------------------------
def build_retriever(replay: bool):
    fx = ROOT / "tests" / "fixtures"
    chunks = ROOT / "corpus" / "chunks.jsonl"
    inner = eg.Bm25Retriever(chunks) if chunks.exists() else eg.FixtureRetriever(
        json.loads((fx / "evidence_example.json").read_text()))
    # Cached retrieval first, so prompts are byte-identical to the recorded E5 run; live BM25 only fills misses.
    return eg.CachedRetriever(inner, replay_only=False)


def check_catalog_prices() -> dict:
    """Compare the embedded prices with the live Workers AI catalog (best effort; never raises)."""
    try:
        import requests

        a, t = secret("CLOUDFLARE_ACCOUNT_ID"), secret("CLOUDFLARE_API_TOKEN")
        if not (a and t):
            return {"checked": False, "reason": "no credentials"}
        found = {}
        for page in (1, 2):
            r = requests.get(f"https://api.cloudflare.com/client/v4/accounts/{a}/ai/models/search",
                             params={"per_page": 100, "page": page}, headers={"Authorization": f"Bearer {t}"}, timeout=60)
            for m in r.json().get("result", []):
                for p in m.get("properties", []):
                    if p.get("property_id") == "price" and isinstance(p.get("value"), list):
                        found[m["name"]] = {x["unit"]: x["price"] for x in p["value"]}
        res = {}
        for m in MODELS:
            if m["provider"] != "workers_ai":
                continue
            c = found.get(m["model"])
            if c is None:
                res[m["model"]] = "not in catalog"
            else:
                ok = abs(c.get("per M input tokens", -1) - m["price"]["in"]) < 1e-9 and \
                    abs(c.get("per M output tokens", -1) - m["price"]["out"]) < 1e-9
                res[m["model"]] = "matches catalog" if ok else f"MISMATCH catalog={c}"
        return {"checked": True, "result": res}
    except Exception as e:  # noqa: BLE001
        return {"checked": False, "reason": type(e).__name__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", action="store_true", help="cache only: never call an LLM")
    ap.add_argument("--models", default="", help="comma-separated ids to run (default: all)")
    ap.add_argument("--probe-n", type=int, default=3)
    ap.add_argument("--neuron-budget", type=float, default=8500.0,
                    help="max Workers AI neurons this run may spend (free allowance is ~10,000/day)")
    ap.add_argument("--cohere-calls", type=int, default=120)
    ap.add_argument("--ask", action="store_true", help="also run the ask-literature cases when the budget allows")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="e5_sweep")
    a = ap.parse_args()
    if a.replay:
        os.environ["ZELOS_LLM_MODE"] = "replay"
    llm.set_call_budget("cohere", a.cohere_calls)

    cases = eg.load_jsonl(ROOT / "eval" / "grounding_cases.jsonl")
    ex_cases = [c for c in cases if c["kind"] == "explain"]
    ask_cases = [c for c in cases if c["kind"] == "ask"]
    qs = ROOT / "results" / "qc_samples.json"
    qcs = json.loads(qs.read_text()) if qs.exists() else [json.loads((ROOT / "tests/fixtures/qc_example.json").read_text())]
    retr = build_retriever(a.replay)
    rec = Recorder()
    rec.install()
    want = {x for x in a.models.split(",") if x}
    prices_check = {"checked": False, "reason": "replay"} if a.replay else check_catalog_prices()

    out_models: dict[str, dict] = {}
    t_start = time.time()

    def run_group(prof: dict, entry: dict, gname: str, kind: str, gcases: list[dict]) -> None:
        if entry["status"] != "ok":
            return
        first, rest = gcases[: a.probe_n], gcases[a.probe_n:]
        n0 = rec.live_neurons
        done, info = run_cases(rec, prof, first, kind, qcs, retr, a.workers)
        probe_neu = rec.live_neurons - n0
        probe_live = [r for r in done if any(not c["cached"] for c in r["calls"])]
        if info["quota_exhausted"]:
            entry["status"] = "quota_exhausted"
            entry["notes"].append(f"{gname}: provider quota error at case {info['stopped_at']}; "
                                  f"{len(done)} case(s) completed before it")
            if done:
                entry["groups"][gname] = summarize(done, prof["price"], kind)
            return
        if not done and (info["unavailable"] or a.replay):
            entry["status"] = "no_cached_responses" if a.replay else "unavailable"
            entry["notes"].append(f"{gname}: no usable responses ({info['unavailable']} case(s) unavailable)")
            return
        # extrapolate the remaining cases from the probe (ask cases: 1.5x, they also make one ungated call)
        per_case = probe_neu / len(probe_live) if probe_live else 0.0
        projected = per_case * len(rest) * (1.0 if kind == "explain" else 1.5)
        remaining = a.neuron_budget - rec.live_neurons
        entry.setdefault("probe", {})[gname] = {
            "cases": len(first), "live_cases": len(probe_live), "neurons_probe": probe_neu,
            "neurons_per_live_case": per_case, "projected_neurons_rest": projected,
            "neuron_budget_remaining_before_rest": remaining}
        if prof["provider"] == "workers_ai" and projected > remaining:
            entry["status"] = "probe_only"
            entry["notes"].append(f"{gname}: projected {projected:.0f} neurons for the remaining {len(rest)} cases "
                                  f"exceeds the {remaining:.0f} left in the budget; only the {len(done)} probe cases ran")
            entry["groups"][gname] = summarize(done, prof["price"], kind)
            return
        more, info2 = run_cases(rec, prof, rest, kind, qcs, retr, a.workers)
        done += more
        if info2["quota_exhausted"]:
            entry["status"] = "quota_exhausted"
            entry["notes"].append(f"{gname}: provider quota error at case {info2['stopped_at']}; "
                                  f"{len(done)} case(s) completed before it")
        dropped = info["unavailable"] + info2["unavailable"]
        if dropped:
            entry["notes"].append(f"{gname}: {dropped} case(s) dropped (LLM unavailable / cache miss)")
        entry["groups"][gname] = summarize(done, prof["price"], kind)

    profiles = [p for p in MODELS if not want or p["id"] in want]
    for prof in profiles:
        out_models[prof["id"]] = {"profile": {k: prof[k] for k in ("provider", "model", "max_tokens", "extra")},
                                  "price": prof["price"], "status": "ok", "groups": {}, "notes": []}
    phases = [("explain_qc", "explain", ex_cases)] + ([("ask_literature", "ask", ask_cases)] if a.ask else [])
    try:
        for gname, kind, gcases in phases:  # all models finish explain-QC before any model spends on ask cases
            for prof in profiles:
                entry = out_models[prof["id"]]
                live0, neu0 = rec.live_calls, rec.live_neurons
                run_group(prof, entry, gname, kind, gcases)
                led = entry.setdefault("ledger", {"live_responses": 0, "neurons_billed_live": 0.0})
                led["live_responses"] += rec.live_calls - live0
                led["neurons_billed_live"] += rec.live_neurons - neu0
    finally:
        rec.uninstall()
    for prof in profiles:
        http = [r for r in llm.LEDGER if r["provider"] == prof["provider"] and r["model"] == prof["model"]]
        out_models[prof["id"]]["ledger"].update({
            "http_attempts": len(http),
            "http_status": {str(s): sum(1 for r in http if r["status"] == s)
                            for s in sorted({r["status"] for r in http if r["status"] is not None})}})

    n_ex = len(ex_cases)
    verdicts = [verdict(out_models, "explain_qc", n_ex)]
    if a.ask:
        verdicts.append(verdict(out_models, "ask_literature", len(ask_cases)))
    result = {
        "experiment": "E5 model sweep",
        "provenance": provenance({"gate_vectors": os.environ.get("ZELOS_GATE_VECTORS"), "llm_mode": llm.mode(),
                                  "retriever": "Bm25Retriever over corpus/chunks.jsonl, cached queries replayed",
                                  "thresholds": load_config()["grounding"], "agent": load_config()["agent"],
                                  "neuron_budget": a.neuron_budget, "probe_n": a.probe_n}),
        "n": {"explain_qc": n_ex, "ask_literature": len(ask_cases) if a.ask else None},
        "models": out_models,
        "verdict": verdicts,
        "price_check": prices_check,
        "call_ledger": {
            "live_responses_this_run": rec.live_calls,
            "http_attempts_this_run": len(llm.LEDGER),
            "http_attempts_by_provider": {p: llm.calls_made(p) for p in llm.PROVIDERS},
            "workers_ai_neurons_billed_this_run": rec.live_neurons,
            "cohere_call_cap": a.cohere_calls,
            "wall_seconds": round(time.time() - t_start, 1),
            "note": "ledger covers this process only; manual probes made before the sweep are not included",
        },
        "notes": [
            "Same cases, prompts, retriever and gate for every model; only the model (and, for reasoning models, "
            "max_tokens and the reasoning-off/low switch) differs. 'draft' is the first model output before the gate; "
            "'gated' is what the user sees after the gate and at most one repair call.",
            "Unsupported/citation/numeric rates are judged by independent re-verification of the claims (eg.judge), as in E5.",
            "JSON-format failure = a model reply that is neither strict JSON claims nor an ABSTAIN (it is parsed as free text).",
            "Latency is the recorded provider latency of the live call plus local step time; replayed cache entries keep their recorded latency.",
        ],
    }
    write_json(ROOT / "results" / f"{a.out}.json", result)
    (ROOT / "results" / f"{a.out}.md").write_text(render_md(result))
    for v in verdicts:
        print(v["text"])
    return 0


# ---- rendering -----------------------------------------------------------------------------------------------------------
def fr(r) -> str:
    if r is None or r.get("rate") is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['rate']:.3f} [{lo:.3f}, {hi:.3f}] ({r['k']}/{r['n']})"


def f2(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def render_md(o: dict) -> str:
    L = ["# E5 model sweep: which LLM behind the gate", "",
         f"git {o['provenance']['git_sha'][:8]} | LLM mode {o['provenance']['llm_mode']} | "
         f"explain-QC n={o['n']['explain_qc']}" + (f", ask-literature n={o['n']['ask_literature']}" if o["n"]["ask_literature"] else ""),
         ""]
    L += ["## Models and run status", "", "| Model | Provider | Profile | Status | Cases | Live responses | Neurons billed (live) |",
          "|---|---|---|---|---|---|---|"]
    for m in o["models"].values():
        p = m["profile"]
        prof = f"max_tokens={p['max_tokens'] or 'production (700)'}" + (f", extra={json.dumps(p['extra'])}" if p["extra"] else "")
        cases = ", ".join(f"{g}={s['n_cases']}" for g, s in m["groups"].items()) or "0"
        L.append(f"| `{p['model']}` | {p['provider']} | {prof} | {m['status']} | {cases} | "
                 f"{m['ledger']['live_responses']} | {m['ledger']['neurons_billed_live']:.0f} |")
    for gname, title in (("explain_qc", "Explain-QC cases"), ("ask_literature", "Ask-literature cases")):
        ms = [(mid, m) for mid, m in o["models"].items() if gname in m["groups"] and m["groups"][gname]["n_cases"]]
        if not ms:
            continue
        L += ["", f"## {title}", "", "| Metric | " + " | ".join(f"{mid} (n={m['groups'][gname]['n_cases']})" for mid, m in ms) + " |",
              "|---|" + "---|" * len(ms)]
        rows = [
            ("Unsupported-claim rate BEFORE gate (draft)", lambda s: fr(s["draft"]["unsupported_claim_rate"])),
            ("Unsupported-claim rate after gate", lambda s: fr(s["gated"]["unsupported_claim_rate"])),
            ("Valid-citation rate, draft", lambda s: fr(s["draft"]["valid_citation_rate"])),
            ("Valid-citation rate, after gate", lambda s: fr(s["gated"]["valid_citation_rate"])),
            ("Numeric-hallucination rate, draft", lambda s: fr(s["draft"]["numeric_hallucination_rate"])),
            ("Numeric-hallucination rate, after gate", lambda s: fr(s["gated"]["numeric_hallucination_rate"])),
            ("Supported claims per case, draft", lambda s: f2(s["draft"]["supported_claims_per_case"])),
            ("Supported claims per case, after gate (sd)",
             lambda s: f"{f2(s['gated']['supported_claims_per_case'])} ({f2(s['gated']['supported_claims_per_case_sd'])})"),
            ("Claims blocked per case", lambda s: f2(s["gated"]["blocked_claims_per_case"])),
            ("JSON-format failure rate (per LLM call)", lambda s: fr(s["json_format_failure_rate"])),
            ("Truncated / empty responses", lambda s: f"{s['truncated_responses']} / {s['empty_responses']}"),
            ("Tokens per answer (prompt + completion)",
             lambda s: f"{s['tokens_per_answer']['total']:.0f} ({s['tokens_per_answer']['prompt']:.0f} + {s['tokens_per_answer']['completion']:.0f})"),
            ("LLM calls per answer", lambda s: f2(s["llm_calls_per_answer"])),
            ("Answers that reached the LLM", lambda s: f"{s['cases_reaching_llm']}/{s['n_cases']}"),
            ("Latency p50 / p95 (ms)",
             lambda s: "n/a" if s["latency_ms"]["p50"] is None else f"{s['latency_ms']['p50']:.0f} / {s['latency_ms']['p95']:.0f}"),
            ("Cost per 100 answers (USD)",
             lambda s: "trial key" if s["cost"]["usd_per_100_answers"] is None else f"${s['cost']['usd_per_100_answers']:.4f}"),
            ("Neurons per 100 answers",
             lambda s: "n/a" if s["cost"]["neurons_per_100_answers"] is None else f"{s['cost']['neurons_per_100_answers']:.0f}"),
        ]
        for name, f in rows:
            L.append(f"| {name} | " + " | ".join(f(m["groups"][gname]) for _, m in ms) + " |")
    L += ["", "## Verdict (computed)", ""] + [f"- {v['text']}" for v in o["verdict"]]
    L += ["", "## Prices and call ledger", ""]
    for mid, m in o["models"].items():
        pr = m["price"]
        ptxt = "trial key" if pr["in"] is None else f"${pr['in']}/M in, ${pr['out']}/M out"
        L.append(f"- {mid}: {ptxt}. Source: {pr['source']}.")
    pc = o["price_check"]
    L.append(f"- Catalog price check: {json.dumps(pc.get('result') or pc.get('reason'))}")
    cl = o["call_ledger"]
    L.append(f"- Live responses this run: {cl['live_responses_this_run']}; HTTP attempts by provider: "
             f"{json.dumps(cl['http_attempts_by_provider'])}; Workers AI neurons billed this run: "
             f"{cl['workers_ai_neurons_billed_this_run']:.0f}; Cohere cap: {cl['cohere_call_cap']}.")
    for mid, m in o["models"].items():
        for n in m["notes"]:
            L.append(f"- {mid}: {n}")
    L += ["", "Notes:"] + [f"- {n}" for n in o["notes"]] + [""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
