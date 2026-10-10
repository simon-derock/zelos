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
