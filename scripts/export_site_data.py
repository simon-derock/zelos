"""Export real Zelos pipeline outputs for the static web console (web/public/data/site.json + thumbnails).

Every sample shown on the site is produced here by the actual pipeline: vision QC -> agent (Workers AI
explainer, gated) -> claim ledger. Corrupted variants are generated with src/zelos/vision/corrupt.py and
labelled as such. QC uses a session batch reference (median/MAD of the other images from the same
experiment session, no labels), the triage mode a lab would run on a day's images.

Run: python scripts/export_site_data.py [--provider workers_ai|template]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zelos.agent.loop import run_inspect
from zelos.common.config import ROOT, load_config, provenance, write_json
from zelos.vision.corrupt import corrupt
from zelos.vision.metrics import BAD_DIRECTION, to_gray01
from zelos.vision.pipeline import inspect
from zelos.vision.scoring import fit_reference

OUT = ROOT / "web" / "public" / "data"
DATA = ROOT / "data" / "raw"


def thumb(img: np.ndarray, path: Path) -> None:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = 960 / max(h, w)
    small = cv2.resize(g, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), small, [cv2.IMWRITE_WEBP_QUALITY, 82])


def site_card(card: dict) -> dict:
    """Map the agent card onto the web console contract (strings for cause/action/claims)."""
    pc = card.get("primary_cause") or {}
    def txt(c):
        if not isinstance(c, dict):
            return str(c)
        ev = " ".join(f"[{e}]" for e in c.get("evidence", []))
        return f"{c.get('text', '')} {ev}".strip()
    return {**card, "primary_cause": pc.get("label") if isinstance(pc, dict) else pc,
            "action": pc.get("action", "") if isinstance(pc, dict) else card.get("action", ""),
            "claims": [txt(c) for c in card.get("claims", [])],
            "blocked": [txt(c) for c in card.get("blocked_claims", card.get("blocked", []))]}


def get_retriever(timeout_s: float = 20.0):
    """Hybrid retriever if it loads within timeout_s, else the offline BM25 retriever used by E5."""
    import sys
    import threading

    box: dict = {}

    def _load() -> None:
        try:
            from zelos.rag.retriever import HybridRetriever

            box["r"] = HybridRetriever.load()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_load, daemon=True)
    t.start()
    t.join(timeout_s)
    if "r" in box:
        return box["r"], "hybrid"
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_grounding import Bm25Retriever

    return Bm25Retriever(ROOT / "corpus" / "chunks.jsonl"), "bm25 (offline fallback)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="workers_ai")
    a = ap.parse_args()
    cfg = load_config()
    real = pd.read_pickle(ROOT / "cache" / "features_all.pkl")
    disk = pd.read_csv(ROOT / "data" / "raw" / "manifest.csv")
    on_disk = {r.file_id: r.local_path for r in disk.itertuples()}
    retriever, rmode = get_retriever()
    print("retriever:", rmode)

    rng = np.random.default_rng(cfg["seed"])
    real = real.assign(local_path=real.file_id.map(on_disk))
    avail = real[real.local_path.notna()]
    picks: list[tuple[pd.Series, str | None]] = []
    for lab in ("good", "bad"):
        sub = avail[avail.label == lab]
        for cl in sorted(sub.cell_line.unique())[:4]:
            row = sub[sub.cell_line == cl].sample(1, random_state=cfg["seed"]).iloc[0]
            picks.append((row, None))
    base = avail[avail.label == "good"].sample(4, random_state=cfg["seed"] + 1)
    for (_, row), kind in zip(base.iterrows(), ["defocus", "bubble", "overexpose", "illumination"], strict=True):
        picks.append((row, kind))

    samples = []
    for row, kind in picks:
        sess = real[real.session.astype(str) == str(row.session)]
        if len(sess) < 8:
            sess = real[real.cell_line == row.cell_line]
        ref = fit_reference(sess[list(BAD_DIRECTION)])
        img = cv2.imread(str(DATA / row.local_path), cv2.IMREAD_UNCHANGED)
        sid = str(row.file_id)
        variant = "original"
        if kind:
            g, _ = to_gray01(img)
            img = (corrupt(g, kind, 5, rng) * 255).astype(np.uint8)
            sid = f"{row.file_id}-{kind}"
            variant = f"corrupted:{kind}:5"
        qc = inspect(img, cell_line=row.cell_line, image_id=sid, reference=ref, mode="batch", culture=True)
        qc["reference"] = f"batch: imaging session {row.session} (n={len(sess)}, label-agnostic)"
        res = run_inspect(qc, retriever=retriever, provider=a.provider)
        thumb(img, OUT / "thumbs" / f"{sid}.webp")
        samples.append({"id": sid, "cell_line": row.cell_line, "time_bin": str(row.time_bin),
                        "expert_label": row.label, "thumb": f"data/thumbs/{sid}.webp", "variant": variant,
                        "qc": qc, "card": site_card(res.get("card") or {}),
                        **{k: res.get(k) for k in ("ledger", "ungated_draft", "trace", "usage", "provider")}})
        print(sid, variant, qc["level"], qc["qc_score"], (res.get("usage") or {}).get("llm_calls"))

    def jload(name):
        f = ROOT / "results" / name
        return json.loads(f.read_text()) if f.exists() else None

    results = jload("results.json") or {}
    site = {
        "generated": provenance(),
        "dataset": {"name": "Organ-on-a-Chip (OOC) Image Dataset", "citation": "Movčana et al., Data 9(2):28, 2024",
                    "doi": "10.5281/zenodo.10203721", "license": "CC BY 4.0 (Zenodo) / CC BY-SA (article page)",
                    "n_images_used": int(len(real)), "cell_lines": sorted(real.cell_line.unique().tolist())},
        "samples": samples,
        "results": {k: (jload(f"{k}.json") or results.get(k, {})) for k in ("e1", "e2", "e4", "e5", "e9")},
        "vlm_baseline": jload("vlm_baseline.json"),
        "llm_sweep": jload("e5_sweep.json"),
        "rule_citations": jload("rule_citations.json"),
        "meta": {"version": "1.1", "branch": "next", "submitted_version": "v1.0-submission"},
        "findings": results.get("findings", []),
        "retriever": rmode,
    }
    write_json(OUT / "site.json", site)
    print("wrote", OUT / "site.json", len(samples), "samples")


if __name__ == "__main__":
    main()
