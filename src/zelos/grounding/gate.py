"""Grounding gate: verify claims against tool outputs and retrieved evidence, repair once, block the rest.

verify(claims, tool_outputs, evidence, cfg=None) -> {"claims": [ledger entries], "n_supported",
    "n_unsupported", "grounded_rate"}
repair(draft_claims, ledger, llm_fn, rounds=1, ...) asks the LLM once to fix or drop unsupported claims.

Rules (a priori, thresholds from configs/default.yaml `grounding`):
  * every cited id must exist: tool ids in tool_outputs, chunk ids in the retrieved evidence, or an
    earlier supported claim id (c1, c2, ...); unknown ids -> unsupported
  * every number must match a tool value (numeric.match_number) or, in a literature claim, a number in a
    cited chunk
  * a literature claim must cite >=1 retrieved chunk that supports it (lexical OR vector)
  * an inference must reference >=1 tool id or supported claim
"""
from __future__ import annotations

import re

from zelos.common.config import load_config
from zelos.grounding import ledger as L
from zelos.grounding.extract import citations_in, parse_response
from zelos.grounding.literature import support_score
from zelos.grounding.numeric import check_numbers, flatten_numeric

_CLAIM_REF = re.compile(r"^c\d+$")


def _grounding_cfg(cfg: dict | None) -> dict:
    g = dict(load_config()["grounding"])
    if cfg:
        g.update(cfg.get("grounding", cfg))
    return g


_METRIC_ROOTS = {"metrics", "z", "zbad", "percentiles"}
_CAUSE_LEAF = re.compile(r"^(.*causes\[\d+\])\.(value|z)$")


def _metric_names(tool_outputs: list[dict]) -> set[str]:
    names: set[str] = set()
    for t in tool_outputs:
        for root in _METRIC_ROOTS:
            if isinstance(t.get(root), dict):
                names |= set(t[root])
        names |= {c["metric"] for c in t.get("causes", []) if isinstance(c, dict) and c.get("metric")}
    return names


def _make_metric_of(tool_outputs: list[dict], tids: list[str]):
    """path (as 'tid:path') -> metric name it belongs to, or None for global values (qc_score, shape, ...)."""
    cause_metric = {}
    for tid, t in zip(tids, tool_outputs, strict=True):
        for i, c in enumerate(t.get("causes", [])):
            if isinstance(c, dict) and c.get("metric"):
                cause_metric[f"{tid}:causes[{i}]"] = c["metric"]

    def metric_of(path: str):
        tid, _, p = path.partition(":")
        parts = p.split(".")
        if len(parts) > 1 and parts[0] in _METRIC_ROOTS:
            return parts[1]
        m = _CAUSE_LEAF.match(f"{tid}:{p}")
        if m:
            return cause_metric.get(m.group(1))
        return None
    return metric_of


def tool_ids_of(tool_outputs: list[dict]) -> list[str]:
    return [t.get("tool_call_id") or f"tool-{i}" for i, t in enumerate(tool_outputs)]


def verify(claims: list[dict], tool_outputs: list[dict], evidence: list[dict], cfg: dict | None = None) -> dict:
    g = _grounding_cfg(cfg)
    tids = tool_ids_of(tool_outputs)
    leaves_by_tool = {tid: flatten_numeric(t) for tid, t in zip(tids, tool_outputs, strict=True)}
    all_leaves = [(f"{tid}:{p}", v) for tid, ls in leaves_by_tool.items() for p, v in ls]
    chunks = {e["chunk_id"]: e for e in evidence if e.get("chunk_id")}
    metric_names = _metric_names(tool_outputs)
    metric_of = _make_metric_of(tool_outputs, tids)
    entries: list[dict] = []
    supported_ids: set[str] = set()

    for i, c in enumerate(claims, start=1):
        cid = f"c{i}"
        text = str(c.get("text", ""))
        ev = [str(x) for x in (c.get("evidence") or [])]
        for x in citations_in(text):
            if x not in ev:
                ev.append(x)
        typ = c.get("type") if c.get("type") in {"numeric", "literature", "inference"} else "inference"
        status, reason = "supported", "ok"
        tool_refs = [x for x in ev if x in tids]
        chunk_refs = [x for x in ev if x in chunks]
        claim_refs = [x for x in ev if _CLAIM_REF.match(x) and x in supported_ids]
        unknown = [x for x in ev if x not in tids and x not in chunks and x not in claim_refs]

        if not text.strip():
            status, reason = "unsupported", "empty claim"
        elif unknown:
            status, reason = "unsupported", f"cites id not in tool results or retrieved evidence: {unknown[0]}"
        else:
            extra_leaves = list(all_leaves)
            for ch in chunk_refs:
                extra_leaves += [(f"{ch}:text", v) for _, v in flatten_numeric(_chunk_numbers(chunks[ch]))]
            low = text.lower()
            mentioned = {m for m in metric_names if m.lower() in low or m.lower().replace("_", " ") in low}
            chk = check_numbers(text, extra_leaves, g["num_rel_tol"], metric_of, mentioned)
            if len(re.findall(r"[A-Za-z]{2,}", re.sub(r"\[[^\]]*\]", " ", text))) < 2:
                status, reason = "unsupported", "claim too short to verify (fewer than 2 words)"
            elif chk["unmatched"]:
                status, reason = "unsupported", f"number(s) not in tool outputs: {', '.join(chk['unmatched'][:3])}"
            elif chk["misbound"]:
                status, reason = "unsupported", f"number {chk['misbound'][0]} belongs to a different metric than the one named"
            elif typ == "literature" or chunk_refs:
                if not chunk_refs:
                    status, reason = "unsupported", "literature claim without a retrieved citation"
                else:
                    best = max((support_score(text, chunks[ch], g["theta_lex"], g["theta_vec"]) | {"chunk": ch}
                                for ch in chunk_refs), key=lambda s: (s["supported"], s["lex"]))
                    if not best["supported"]:
                        status, reason = ("unsupported",
                                          f"cited passage does not support the claim (lexical overlap {best['lex']})")
            elif typ == "inference" and not (tool_refs or claim_refs):
                status, reason = "unsupported", "inference does not reference a tool result or supported claim"
            elif typ == "numeric" and not chk["numbers"] and not (tool_refs or claim_refs):
                status, reason = "unsupported", "numeric claim has no number and no tool reference"
        if status == "supported":
            supported_ids.add(cid)
        entries.append(L.entry(cid, text, typ, ev, status, reason))
    return {"claims": entries, **L.summarize(entries)}


def _chunk_numbers(chunk: dict) -> dict:
    from zelos.grounding.numeric import extract_numbers

    return {"n": [n["value"] for n in extract_numbers(chunk.get("text", ""))]}


def _compact_tools(tool_outputs: list[dict]) -> str:
    import json

    return json.dumps(tool_outputs, separators=(",", ":"), default=str)[:6000]


def repair(draft_claims: list[dict], ledger: dict, llm_fn, rounds: int = 1, *, tool_outputs: list[dict] | None = None,
           evidence: list[dict] | None = None, cfg: dict | None = None) -> dict:
    """One bounded repair pass. llm_fn(messages) -> chat() dict. Returns
    {"claims": final claim list (supported kept + repaired), "ledger": verify() dict over final claims,
     "blocked": [entries still unsupported], "llm_calls": int, "usage": {...}}.
    Unsupported draft claims that were not repaired are kept as blocked entries (status unsupported)."""
    from zelos.agent import prompts

    tool_outputs, evidence = tool_outputs or [], evidence or []
    bad = [e for e in ledger["claims"] if e["status"] == "unsupported"]
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    calls = 0
    keep = [{"text": e["text"], "type": e["type"], "evidence": e["evidence"]}
            for e in ledger["claims"] if e["status"] == "supported"]
    if not bad or rounds < 1:
        return {"claims": keep, "ledger": ledger, "blocked": bad, "llm_calls": 0, "usage": usage}
    msgs = prompts.build_repair_messages(tool_outputs, evidence, keep, bad)
    try:
        res = llm_fn(msgs)
    except Exception:  # LLM unavailable: block what failed
        res = None
    repaired: list[dict] = []
    if res:
        calls = 1
        usage = {"prompt_tokens": res["usage"]["prompt_tokens"], "completion_tokens": res["usage"]["completion_tokens"]}
        repaired = parse_response(res["text"])["claims"]
    final_claims = keep + repaired
    final = verify(final_claims, tool_outputs, evidence, cfg)
    # anything still unsupported is blocked; first-pass failures are retained in the ledger as blocked history
    first_pass = [dict(e, claim_id=f"d{n}", reason=f"blocked at draft: {e['reason']}", repaired=bool(repaired))
                  for n, e in enumerate(bad, start=1)]
    still_bad = [e for e in final["claims"] if e["status"] == "unsupported"]
    out_ledger = {"claims": final["claims"] + first_pass, "n_supported": final["n_supported"],
                  "n_unsupported": final["n_unsupported"] + len(first_pass),
                  "grounded_rate": final["grounded_rate"], "n_draft_unsupported": len(bad)}
    return {"claims": final_claims, "ledger": out_ledger, "blocked": still_bad + first_pass,
            "llm_calls": calls, "usage": usage}
