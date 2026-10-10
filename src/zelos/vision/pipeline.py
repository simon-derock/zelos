"""Single-image QC entry point used by the agent, API, MCP server and CLI.

inspect(path|bytes|array, cell_line=None, image_id=None, reference=None, mode=None, culture=False) -> QC dict
  tool_call_id, image{id, sha256, shape, dtype, cell_line}, metrics, z, zbad, percentiles, qc_score, level,
  causes[{label, note, metric, value, z, action, route?}], route, density_state, reference, thresholds, latency_ms
  (+ culture_state when culture=True and the optional DINOv2 stack is installed)

Reference modes
  stored : between-session-aware reference fitted on development sessions (results/reference_v11.json)
  batch  : median/MAD of the images of the same imaging session (pass `reference=`); selected automatically
Thresholds come from results/calibration.json (90th/99th percentile of expert-good scores on calibration
sessions); without it the pre-registered z = 3 / 5 are used. Causes use signature centroids
(results/signatures.json) with the rule table as fallback.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from zelos.common.config import CONFIGS, ROOT, load_config, sha256_bytes
from zelos.vision.attribution import attribute, attribute_v2, load_signatures
from zelos.vision.metrics import compute_metrics
from zelos.vision.scoring import score_image

REFERENCE_PATH = ROOT / "results" / "reference.json"
REFERENCE_V11 = ROOT / "results" / "reference_v11.json"
CALIBRATION = ROOT / "results" / "calibration.json"


@lru_cache
def load_reference(path: str | None = None) -> dict[str, Any]:
    if path is None and REFERENCE_V11.exists():
        return json.loads(REFERENCE_V11.read_text())
    return json.loads(Path(path or REFERENCE_PATH).read_text())


@lru_cache
def _calibration() -> dict[str, Any] | None:
    return json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else None


@lru_cache
def _rule_basis() -> dict[str, Any]:
    f = ROOT / "results" / "rule_citations.json"
    return json.loads(f.read_text())["rules"] if f.exists() else {}


@lru_cache
def _routes() -> dict[str, str]:
    return yaml.safe_load((CONFIGS / "routes.yaml").read_text())


def read_image(path: str | Path) -> tuple[np.ndarray, bytes]:
    raw = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img, raw


def _gray_u8(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if g.dtype == np.uint16:
        g = (g / 257).astype(np.uint8)
    elif g.dtype != np.uint8:
        g = (np.clip(g, 0, 1) * 255).astype(np.uint8)
    return g


def inspect(src: str | Path | np.ndarray | bytes, cell_line: str | None = None, image_id: str | None = None,
            reference: dict[str, Any] | None = None, mode: str | None = None, culture: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    if isinstance(src, (str, Path)):
        img, raw = read_image(src)
        image_id = image_id or Path(src).stem
    elif isinstance(src, bytes):
        raw = src
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("unreadable image bytes")
    else:
        img = src
        raw = np.ascontiguousarray(img).tobytes()
    digest = sha256_bytes(raw)
    cfg = load_config()
    mode = mode or ("batch" if reference is not None else "stored")
    cal = _calibration()
    key = {"batch": "batch_session", "stored": "between_session (v1.1)"}.get(mode)
    cutoffs = None
    if cal and key in cal["variants"]:
        v = cal["variants"][key]
        review_z, fail_z, th_src = v["review"], v["fail"], "calibrated"
        cutoffs = v.get("cutoffs")
    else:
        review_z, fail_z, th_src = cfg["qc"]["review_z"], cfg["qc"]["fail_z"], "pre-registered"
    m = compute_metrics(img, cfg["vision"]["max_side"])
    ref = reference or load_reference()
    if cutoffs:
        s = score_image(m, ref, cell_line, 1e9, 1e9)  # z values only; level from the calibrated ratio below
        zb = s["zbad"]
        ratio = max(zb.get(k, 0.0) / c for k, c in cutoffs.items())
        s["qc_zmax"], s["qc_score"] = s["qc_score"], ratio
        s["level"] = "FAIL" if ratio > fail_z else "REVIEW" if ratio > review_z else "PASS"
    else:
        s = score_image(m, ref, cell_line, review_z, fail_z)
    causes: list[dict[str, Any]] = []
    attribution = "none"
    if s["level"] != "PASS":
        sig = load_signatures(variant="batch" if mode == "batch" else "stored")
        if sig and sig.get("labels"):
            causes, attribution = attribute_v2(m, s["zbad"], sig), "signature-centroid"
        if not causes:
            causes, attribution = attribute(m, s["zbad"]), "rule-table"
    routes = _routes()
    basis = _rule_basis()
    for c in causes:
        c["route"] = routes.get(c["label"], "REVIEW")
        b = basis.get(c["label"])
        c["basis"] = ({k: b[k] for k in ("chunk_id", "title", "url", "license", "excerpt")} if b and b["status"] == "supported"
                      else None)
    route = "USE" if s["level"] == "PASS" else (causes[0]["route"] if causes else "REVIEW")
    out = {
        "tool_call_id": f"qc-{digest[:8]}",
        "image": {"id": image_id or digest[:12], "sha256": digest, "shape": list(img.shape[:2]),
                  "dtype": str(img.dtype), "cell_line": cell_line},
        "metrics": {k: round(v, 6) for k, v in m.items()},
        "z": {k: round(v, 3) for k, v in s["z"].items()},
        "zbad": {k: round(v, 3) for k, v in s["zbad"].items()},
        "percentiles": {k: round(v, 1) for k, v in s["percentiles"].items()},
        "qc_score": round(s["qc_score"], 3),
        "qc_zmax": round(s.get("qc_zmax", s["qc_score"]), 3),
        "level": s["level"],
        "route": route,
        "causes": causes,
        "attribution": attribution,
        "density_state": s["density_state"],
        "reference": f"{mode}: {s['reference']}",
        "thresholds": {"review": round(review_z, 3), "fail": round(fail_z, 3), "source": th_src},
    }
    if culture:
        from zelos.vision.culture import culture_state

        out["culture_state"] = culture_state(_gray_u8(img))
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
