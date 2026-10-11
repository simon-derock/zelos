"""Reference statistics and robust scoring (descriptive statistics only; nothing is trained).

fit_reference(df, by)  -> {"global": {metric: [median, scale]}, "groups": {g: {...}}, "n": ...}
score_image(m, ref, g) -> {"z": signed robust z, "zbad": one-sided z in the bad direction,
                           "qc_score": max zbad over quality metrics, "level": PASS|REVIEW|FAIL,
                           "percentiles": approx. reference percentile per metric}
Robust z = 0.6745 * (x - median) / MAD, with MAD floored so constant metrics cannot explode.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from zelos.vision.metrics import BAD_DIRECTION, QUALITY_METRICS

# Absolute floors for the robust scale (in metric units) — a-priori, for metrics that are ~0 on clean images.
ABS_FLOOR = {
    "clip_low_frac": 0.002,
    "clip_high_frac": 0.002,
    "debris_count": 1.0,
    "debris_area_frac": 0.0005,
    "bubble_score": 0.002,
}


def _stats(col: pd.Series, name: str) -> list[float]:
    x = col.to_numpy(dtype=float)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) / 0.6745
    scale = max(mad, 0.02 * abs(med), ABS_FLOOR.get(name, 1e-9))
    q = np.percentile(x, np.arange(0, 101, 5)).tolist()
    return [med, scale, *q]


def fit_reference(metrics_df: pd.DataFrame, by: str | None = None, min_group: int = 30) -> dict[str, Any]:
    cols = list(BAD_DIRECTION)
    ref: dict[str, Any] = {"global": {c: _stats(metrics_df[c], c) for c in cols}, "groups": {}, "n": len(metrics_df)}
    if by and by in metrics_df:
        for g, sub in metrics_df.groupby(by):
            if len(sub) >= min_group:
                ref["groups"][str(g)] = {c: _stats(sub[c], c) for c in cols}
    return ref


def fit_reference_sessions(metrics_df: pd.DataFrame, session_col: str = "session", min_session: int = 8) -> dict[str, Any]:
    """Reference whose scale includes between-session spread (variance components).

    scale^2 = within-session MAD^2 (residuals after removing each session's median)
            + between-session MAD^2 (spread of session medians); center = median of session medians.
    Descriptive statistics only. Fixes the over-confident scale of a pooled MAD when sessions differ
    in acquisition settings.
    """
    cols = list(BAD_DIRECTION)
    sess = metrics_df[session_col].astype(str)
    big = sess.map(sess.value_counts()) >= min_session
    d = metrics_df[big]
    table: dict[str, list[float]] = {}
    within_t: dict[str, float] = {}
    between_t: dict[str, float] = {}
    for c in cols:
        x = d[c].to_numpy(dtype=float)
        med_by = d.groupby(sess[big])[c].transform("median").to_numpy(dtype=float)
        resid = x - med_by
        w = float(np.median(np.abs(resid - np.median(resid)))) / 0.6745
        sm = d.groupby(sess[big])[c].median().to_numpy(dtype=float)
        center = float(np.median(sm))
        b = float(np.median(np.abs(sm - center))) / 0.6745
        scale = max(math.sqrt(w * w + b * b), 0.02 * abs(center), ABS_FLOOR.get(c, 1e-9))
        q = np.percentile(x, np.arange(0, 101, 5)).tolist()
        table[c] = [center, scale, *q]
        within_t[c], between_t[c] = w, b
    return {"global": table, "groups": {}, "n": int(len(d)), "within": within_t, "between": between_t,
            "kind": "between-session-aware"}


def _percentile(x: float, q: list[float]) -> float:
    return float(np.interp(x, q, np.arange(0, 101, 5))) if q[-1] > q[0] else 50.0


def score_image(
    metrics: dict[str, float], ref: dict[str, Any], group: str | None = None, review_z: float = 3.0, fail_z: float = 5.0
) -> dict[str, Any]:
    table = ref["groups"].get(str(group)) if group is not None else None
    table = table or ref["global"]
    z, zbad, pct = {}, {}, {}
    for k, d in BAD_DIRECTION.items():
        med, scale, *q = table[k]
        zz = (metrics[k] - med) / scale
        z[k] = float(zz)
        zbad[k] = float(max(0.0, zz * d)) if d != 0 else float(abs(zz))
        pct[k] = _percentile(metrics[k], q)
    zbad["mean_intensity_low"] = float(max(0.0, -z["mean_intensity"]))
    qc = max(zbad[k] for k in QUALITY_METRICS)
    qc = 0.0 if math.isnan(qc) else qc
    level = "FAIL" if qc > fail_z else "REVIEW" if qc > review_z else "PASS"
    dens = z["foreground_frac"]
    density = "SPARSE" if dens < -3 else "CONFLUENT" if dens > 3 else "NORMAL"
    return {"z": z, "zbad": zbad, "qc_score": float(qc), "level": level, "percentiles": pct, "density_state": density,
            "reference": "group:" + str(group) if table is not ref["global"] else "global"}
