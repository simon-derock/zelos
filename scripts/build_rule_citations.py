"""Attach a verified literature basis to every cause -> action rule (or report it unsupported).

For each label in configs/rule_claims.yaml: BM25 over corpus/chunks.jsonl -> top 8 chunks -> keep the chunk
with the highest support under the grounding gate (content-word overlap >= theta_lex) -> store an excerpt of at
most 25 words. Output: results/rule_citations.json. Nothing is typed by hand; unsupported rules stay unsupported.
Run: python scripts/build_rule_citations.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zelos.common.config import load_config, provenance, write_json  # noqa: E402
from zelos.grounding.literature import content_words, lexical_support  # noqa: E402


def best_sentence(claim: str, text: str, n_words: int = 25) -> str:
    cw = content_words(claim)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]
    if not sents:
        return " ".join(text.split()[:n_words])
    s = max(sents, key=lambda x: len(cw & content_words(x)))
    w = s.split()
    return " ".join(w[:n_words]) + (" ..." if len(w) > n_words else "")


def has_groups(sentence: str, groups: list[list[str]]) -> bool:
    t = sentence.lower()
    return all(any(term.lower() in t for term in g) for g in groups)


def main() -> None:
    # The gate's general threshold (0.30) accepted off-topic sentences for rule bases (inspected by reading every
    # match), so rule bases use a stricter 0.50. This is a POST-HOC choice made after seeing the matches; disclosed.
    theta = max(0.50, load_config()["grounding"]["theta_lex"])
    rules = yaml.safe_load((ROOT / "configs/rule_claims.yaml").read_text())
    chunks = [json.loads(x) for x in (ROOT / "corpus/chunks.jsonl").read_text().splitlines() if x.strip()]
    out = {}
    for label, spec in rules.items():
        claim, groups = spec["claim"], spec["require"]
        best = (0.0, None, None)
        for c in chunks:
            for sent in re.split(r"(?<=[.!?])\s+", c["text"]):
                if len(sent.split()) < 6 or not has_groups(sent, groups):
                    continue
                s = lexical_support(claim, sent)
                if s > best[0]:
                    best = (s, c, sent)
        s, c, sent = best
        ok = c is not None and s >= theta
        w = sent.split() if sent else []
        out[label] = {"claim": claim, "require": groups, "status": "supported" if ok else "unsupported", "support": round(s, 3),
                      "chunk_id": c["chunk_id"] if ok else None, "title": c.get("title") if ok else None,
                      "url": c.get("url") if ok else None, "license": c.get("license") if ok else None,
                      "excerpt": (" ".join(w[:25]) + (" ..." if len(w) > 25 else "")) if ok else None}
    n_ok = sum(v["status"] == "supported" for v in out.values())
    write_json(ROOT / "results/rule_citations.json", {"rules": out, "n_supported": n_ok, "n_rules": len(out), "theta_lex": theta,
                                                       "method": "sentence-level: all require-groups present AND lexical overlap >= theta over every corpus sentence",
                                                       "disclosure": "theta raised from the gate default 0.30 to 0.50 after reading the matches (post hoc)",
                                                       "provenance": provenance()})
    for k, v in out.items():
        print(f"{k:20s} {v['status']:11s} {v['support']:.2f} {v['chunk_id']}\n    {v['excerpt']}")
    print(f"{n_ok}/{len(out)} rules have a verified literature basis")


if __name__ == "__main__":
    main()
