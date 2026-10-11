"""Artifact attribution from a fixed, a-priori rule table (configs/artifact_rules.yaml) and
cause -> lab action table (configs/actions.yaml).

attribute(metrics, zbad, rules, actions) -> list of causes sorted by triggering z (first = primary):
  {"label", "note", "metric", "value", "z", "action"}
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from zelos.common.config import CONFIGS


@lru_cache
def load_rules() -> tuple[dict[str, Any], dict[str, str]]:
    rules = yaml.safe_load((CONFIGS / "artifact_rules.yaml").read_text())["labels"]
    actions = yaml.safe_load((CONFIGS / "actions.yaml").read_text())
    return rules, actions


def attribute(metrics: dict[str, float], zbad: dict[str, float], rules: dict | None = None,
              actions: dict | None = None) -> list[dict[str, Any]]:
    if rules is None or actions is None:
        rules, actions = load_rules()
    out = []
    for label, spec in rules.items():
        best = None
        for t in spec["triggers"]:
            zz = zbad.get(t["metric"], 0.0)
            if zz > t["z"] and (best is None or zz > best[1]):
                best = (t["metric"], zz)
        if best:
            src = best[0].replace("mean_intensity_low", "mean_intensity")
            out.append({"label": label, "note": spec["note"], "metric": best[0], "value": float(metrics[src]),
                        "z": round(float(best[1]), 3), "action": actions.get(label, "")})
    return sorted(out, key=lambda c: -c["z"])


# ---------------------------------------------------------------------------------------------
# v2: signature attribution. Each cause label has a centroid in log1p(one-sided z) space, estimated as the
# mean over corruptions injected into DEV-session good images (descriptive statistic; no gradient updates).
# At inference the cause is the nearest centroid by cosine similarity. Rules (v1) remain as fallback.
# ---------------------------------------------------------------------------------------------
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from zelos.vision.metrics import QUALITY_METRICS  # noqa: E402

SIG_KEYS = [*QUALITY_METRICS, "mean_intensity_low"]


def _vec(zbad: dict[str, float]) -> np.ndarray:
    return np.log1p(np.array([max(0.0, float(zbad.get(k, 0.0))) for k in SIG_KEYS]))


def fit_signatures(examples: dict[str, list[dict[str, float]]]) -> dict[str, Any]:
    """examples: label -> list of zbad dicts (one per injected corruption)."""
    sig = {}
    for label, zs in examples.items():
        if not zs:
            continue
        m = np.mean([_vec(z) for z in zs], axis=0)
        n = float(np.linalg.norm(m))
        sig[label] = {"centroid": (m / n).tolist() if n > 0 else m.tolist(), "n": len(zs)}
    return {"keys": SIG_KEYS, "labels": sig}


def attribute_v2(metrics: dict[str, float], zbad: dict[str, float], signatures: dict[str, Any],
                 rules: dict | None = None, actions: dict | None = None, top: int = 2, rel: float = 0.9) -> list[dict[str, Any]]:
    if rules is None or actions is None:
        rules, actions = load_rules()
    v = _vec(zbad)
    nv = float(np.linalg.norm(v))
    if nv == 0:
        return []
    v = v / nv
    sims = {lab: float(np.dot(v, np.asarray(s["centroid"]))) for lab, s in signatures["labels"].items()}
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    out = []
    for lab, sim in ranked[:top]:
        if out and sim < rel * ranked[0][1]:
            break
        cen = np.asarray(signatures["labels"][lab]["centroid"])
        j = int(np.argmax(v * cen))
        key = SIG_KEYS[j]
        src = key.replace("mean_intensity_low", "mean_intensity")
        out.append({"label": lab, "note": rules.get(lab, {}).get("note", lab), "metric": key, "value": float(metrics.get(src, 0.0)),
                    "z": round(float(zbad.get(key, 0.0)), 3), "similarity": round(sim, 3),
                    "action": actions.get(lab, "")})
    return out


def load_signatures(path: str | Path | None = None, variant: str = "batch") -> dict[str, Any] | None:
    from zelos.common.config import ROOT

    p = Path(path) if path else ROOT / "results" / "signatures.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(variant) or d
