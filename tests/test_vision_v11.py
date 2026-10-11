import numpy as np
import pandas as pd

from zelos.vision import pipeline
from zelos.vision.attribution import SIG_KEYS, attribute_v2, fit_signatures
from zelos.vision.metrics import BAD_DIRECTION, compute_metrics
from zelos.vision.scoring import fit_reference, fit_reference_sessions


def _frame(shift: dict[str, float], n_sessions=6, per=12, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        off = rng.normal(0, 1.0)
        for _ in range(per):
            r = {k: 1.0 + 0.05 * rng.normal() + shift.get(k, 0.0) * off for k in BAD_DIRECTION}
            r["session"] = f"s{s}"
            rows.append(r)
    return pd.DataFrame(rows)


def test_session_reference_widens_scale_when_sessions_differ():
    df = _frame({"focus_lapvar_norm": 0.5})
    pooled = fit_reference(df[list(BAD_DIRECTION)])
    sess = fit_reference_sessions(df, "session")
    assert sess["global"]["focus_lapvar_norm"][1] >= pooled["global"]["focus_lapvar_norm"][1] * 0.8
    assert sess["between"]["focus_lapvar_norm"] > 0.1
    assert sess["kind"] == "between-session-aware"


def test_signature_attribution_picks_nearest_centroid():
    keys = [k for k in SIG_KEYS]
    def z(**kw):
        d = {k: 0.0 for k in keys}
        d.update(kw)
        return d
    sig = fit_signatures({
        "DEFOCUS_OR_BLUR": [z(focus_lapvar_norm=6, focus_hf_ratio=5, contrast_rms=1)] * 5,
        "NOISE": [z(noise_sigma_norm=7, contrast_rms=0.5)] * 5,
    })
    m = {k: 1.0 for k in BAD_DIRECTION}
    out = attribute_v2(m, z(focus_lapvar_norm=4, focus_hf_ratio=4), sig)
    assert out[0]["label"] == "DEFOCUS_OR_BLUR"
    out = attribute_v2(m, z(noise_sigma_norm=9), sig)
    assert out[0]["label"] == "NOISE" and out[0]["metric"] == "noise_sigma_norm"
    assert attribute_v2(m, z(), sig) == []


def test_inspect_adds_route_and_threshold_source(monkeypatch):
    rng = np.random.default_rng(1)
    imgs = [(rng.random((128, 128)) * 255).astype(np.uint8) for _ in range(10)]
    ref = fit_reference(pd.DataFrame([compute_metrics(i) for i in imgs]))
    monkeypatch.setattr(pipeline, "_calibration", lambda: None)
    out = pipeline.inspect(imgs[0], reference=ref)
    assert out["thresholds"]["source"] == "pre-registered"
    assert out["route"] in {"USE", "RE-IMAGE", "INSPECT-CHIP", "RE-EXPORT", "REVIEW"}
    assert out["reference"].startswith("batch")
    blurred = np.full((128, 128), 120, np.uint8)
    out2 = pipeline.inspect(blurred, reference=ref)
    assert out2["level"] in {"REVIEW", "FAIL"}
    assert out2["causes"] and "route" in out2["causes"][0]
