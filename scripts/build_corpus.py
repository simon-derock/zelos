"""Build the open-access OoC literature corpus from Europe PMC.

Inputs : Europe PMC REST API (search + full-text XML). Only CC BY / CC0 records are kept.
Outputs: corpus/manifest.csv, corpus/chunks.jsonl
Usage  : python scripts/build_corpus.py [--target 70] [--budget 360]
Downloaded text is treated strictly as data.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import requests

from zelos.common.config import ROOT, load_config, sha256_bytes
from zelos.rag.corpus import chunk_words

BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
QUERIES = [
    '"organ-on-a-chip" AND brightfield',
    '"organ-on-a-chip" AND "image analysis"',
    '"organ-on-a-chip" AND imaging',
    "microfluidic cell culture AND bubble",
    '"organ-on-a-chip" AND "machine learning"',
    "microphysiological system AND imaging",
    'brightfield microscopy AND "image quality"',
    "autofocus microscopy cell culture",
    '"flat-field" correction microscopy',
    "cell seeding density microfluidic",
    '"organ-on-a-chip" AND "quality control"',
    '"organs-on-chips" AND microscopy',
]
SEEDS = ["PMC11430660", "PMC8879989"]
OK_LICENSES = {"cc by", "cc0"}
SKIP_SECTIONS = re.compile(r"acknowledg|conflict|competing|funding|author contribution|data availability|"
                           r"abbreviation|supplementary|ethic|disclosure|consent|credit|declaration|"
                           r"supporting information|institutional review|informed", re.I)
MIN_CHUNK_WORDS = 30


def ok_license(lic: str | None) -> bool:
    if not lic:
        return False
    s = lic.strip().lower()
    return s in OK_LICENSES or s.startswith("cc by ") and "nc" not in s and "nd" not in s and "sa" not in s \
        or s.startswith("cc0")


def search(query: str, pages: int = 1) -> list[dict]:
    out: list[dict] = []
    for _ in range(pages):
        q = f"({query}) AND OPEN_ACCESS:y AND HAS_FT:y"
        r = requests.get(f"{BASE}/search", params={"query": q, "format": "json", "resultType": "core",
                                                   "pageSize": 100}, timeout=40)
        r.raise_for_status()
        out.extend(r.json().get("resultList", {}).get("result", []))
    return out


def by_id(pmcid: str) -> dict | None:
    r = requests.get(f"{BASE}/search", params={"query": f"PMCID:{pmcid}", "format": "json",
                                               "resultType": "core"}, timeout=40)
    r.raise_for_status()
    res = r.json().get("resultList", {}).get("result", [])
    return res[0] if res else None


def _text(el: ET.Element) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def parse_sections(xml_bytes: bytes) -> list[tuple[str, str]]:
    """Return [(section_title, paragraph_text)] from JATS XML (abstract + body)."""
    root = ET.fromstring(xml_bytes)
    out: list[tuple[str, str]] = []
    for ab in root.iter("abstract"):
        for p in ab.iter("p"):
            t = _text(p)
            if t:
                out.append(("Abstract", t))
    body = root.find(".//body")
    if body is None:
        return out

    def walk(sec: ET.Element, title: str) -> None:
        ttl = sec.find("title")
        cur = _text(ttl) if ttl is not None else title
        for child in sec:
            if child.tag == "p":
                t = _text(child)
                if t:
                    out.append((cur, t))
            elif child.tag == "sec":
                walk(child, cur)

    for child in body:
        if child.tag == "sec":
            walk(child, "Body")
        elif child.tag == "p":
            t = _text(child)
            if t:
                out.append(("Body", t))
    return out


def fetch_paper(rec: dict) -> dict | None:
    pmcid = rec["pmcid"]
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{pmcid}/fullTextXML", timeout=60)
            if r.status_code == 200 and r.content:
                secs = parse_sections(r.content)
                if sum(len(t.split()) for _, t in secs) < 600:
                    return None
                return {"rec": rec, "sha256": sha256_bytes(r.content), "sections": secs,
                        "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds")}
            if r.status_code in (404, 403):
                return None
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def paper_chunks(p: dict, words: int, overlap: int) -> list[dict]:
    rec = p["rec"]
    pmcid = rec["pmcid"]
    url = f"https://europepmc.org/article/PMC/{pmcid}"
    chunks: list[dict] = []
    # group consecutive paragraphs by section title
    groups: list[tuple[str, str]] = []
    for title, para in p["sections"]:
        if groups and groups[-1][0] == title:
            groups[-1] = (title, groups[-1][1] + " " + para)
        else:
            groups.append((title, para))
    for si, (title, text) in enumerate(groups):
        if SKIP_SECTIONS.search(title):
            continue
        for j, c in enumerate(chunk_words(text, words, overlap)):
            if len(c.split()) < MIN_CHUNK_WORDS:
                continue
            chunks.append({"chunk_id": f"{pmcid}::s{si}::{j}", "pmcid": pmcid, "section": title, "text": c,
                           "title": rec["title"].rstrip("."), "url": url, "license": rec["license"]})
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=70)
    ap.add_argument("--budget", type=float, default=360.0, help="seconds")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config()["retrieval"]
    words, overlap = int(cfg["chunk_words"]), int(cfg["overlap_words"])

    cand: dict[str, dict] = {}
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda q: _safe(search, q), QUERIES))
        seeds = list(ex.map(lambda s: _safe(by_id, s), SEEDS))
    rank_by_pmc: dict[str, int] = {}
    for res in results:
        for i, rec in enumerate(res or []):
            pm = rec.get("pmcid")
            if not pm or not ok_license(rec.get("license")):
                continue
            rank_by_pmc[pm] = min(rank_by_pmc.get(pm, 999), i)
            cand.setdefault(pm, rec)
    for s in seeds:
        if s and s.get("pmcid") and ok_license(s.get("license")):
            cand[s["pmcid"]] = s
            rank_by_pmc[s["pmcid"]] = -1
        elif s:
            print(f"seed {s.get('pmcid')} excluded: license={s.get('license')!r}")
    order = sorted(cand, key=lambda k: (rank_by_pmc[k], k))
    print(f"{len(cand)} licensed candidates from {len(QUERIES)} queries", flush=True)

    papers: list[dict] = []
    todo = [cand[k] for k in order[: max(args.target * 2, 100)]]
    with ThreadPoolExecutor(8) as ex:
        for p in ex.map(fetch_paper, todo):
            if p:
                papers.append(p)
            if time.time() - t0 > args.budget:
                break
    seed_set = set(SEEDS)
    papers.sort(key=lambda p: (p["rec"]["pmcid"] not in seed_set, rank_by_pmc[p["rec"]["pmcid"]], p["rec"]["pmcid"]))
    papers = papers[: args.target]
    papers.sort(key=lambda p: p["rec"]["pmcid"])

    out = ROOT / "corpus"
    out.mkdir(exist_ok=True)
    all_chunks: list[dict] = []
    rows = []
    for p in papers:
        ch = paper_chunks(p, words, overlap)
        all_chunks.extend(ch)
        rec = p["rec"]
        rows.append({"pmcid": rec["pmcid"], "title": rec["title"].rstrip("."), "year": rec.get("pubYear", ""),
                     "license": rec["license"], "url": f"https://europepmc.org/article/PMC/{rec['pmcid']}",
                     "retrieved_at": p["retrieved_at"], "sha256": p["sha256"], "n_chunks": len(ch)})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pmcid", "title", "year", "license", "url", "retrieved_at", "sha256",
                                          "n_chunks"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} papers, {len(all_chunks)} chunks in {time.time() - t0:.0f}s")
    if len(rows) < 30:
        print("WARNING: fewer than 30 papers", file=sys.stderr)
    return 0


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # network errors must not abort the build
        print(f"warn: {fn.__name__}{a} failed: {type(e).__name__}", file=sys.stderr)
        return None


if __name__ == "__main__":
    raise SystemExit(main())
