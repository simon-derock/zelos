"""Map QC cause labels to literature search queries.

Inputs : list of cause labels (e.g. "BUBBLE"), configs/query_templates.yaml
Outputs: list of query strings, one per distinct label, order preserved.
"""
from __future__ import annotations

from functools import lru_cache

import yaml

from zelos.common.config import CONFIGS

GENERIC = "organ-on-a-chip brightfield microscopy image quality control artifacts culture monitoring"


@lru_cache
def _templates() -> dict[str, str]:
    return yaml.safe_load((CONFIGS / "query_templates.yaml").read_text()) or {}


def queries_for_causes(labels: list[str]) -> list[str]:
    """One query per distinct label; unknown labels map to the generic OoC imaging QC query."""
    tpl = _templates()
    out: list[str] = []
    for lab in labels:
        q = tpl.get(str(lab).strip().upper(), GENERIC)
        if q not in out:
            out.append(q)
    return out
