"""Single-image QC entry point used by the agent, API, MCP server and CLI.

inspect(path_or_array, cell_line=None, image_id=None) -> QC dict (schema in docs / README):
  tool_call_id, image{id, sha256, shape, dtype, cell_line}, metrics, z, zbad, percentiles,
  qc_score, level, causes[{label, note, metric, value, z, action}], density_state, latency_ms
The reference statistics come from results/reference.json (built by `evaluate.py` from the dev split).
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from zelos.common.config import ROOT, load_config, sha256_bytes
from zelos.vision.attribution import attribute
from zelos.vision.metrics import compute_metrics
from zelos.vision.scoring import score_image

REFERENCE_PATH = ROOT / "results" / "reference.json"


@lru_cache
def load_reference(path: str | None = None) -> dict[str, Any]:
    return json.loads(Path(path or REFERENCE_PATH).read_text())


def read_image(path: str | Path) -> tuple[np.ndarray, bytes]:
    raw = Path(path).read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img, raw


def inspect(src: str | Path | np.ndarray | bytes, cell_line: str | None = None, image_id: str | None = None,
            reference: dict[str, Any] | None = None) -> dict[str, Any]:
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
    m = compute_metrics(img, cfg["vision"]["max_side"])
    ref = reference or load_reference()
    s = score_image(m, ref, cell_line, cfg["qc"]["review_z"], cfg["qc"]["fail_z"])
    causes = attribute(m, s["zbad"]) if s["level"] != "PASS" else []
    return {
        "tool_call_id": f"qc-{digest[:8]}",
        "image": {"id": image_id or digest[:12], "sha256": digest, "shape": list(img.shape[:2]),
                  "dtype": str(img.dtype), "cell_line": cell_line},
        "metrics": {k: round(v, 6) for k, v in m.items()},
        "z": {k: round(v, 3) for k, v in s["z"].items()},
        "zbad": {k: round(v, 3) for k, v in s["zbad"].items()},
        "percentiles": {k: round(v, 1) for k, v in s["percentiles"].items()},
        "qc_score": round(s["qc_score"], 3),
        "level": s["level"],
        "causes": causes,
        "density_state": s["density_state"],
        "reference": s["reference"],
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
