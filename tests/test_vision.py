import numpy as np
import pandas as pd
import pytest

from zelos.vision.attribution import attribute
from zelos.vision.corrupt import KIND_TO_LABEL, KINDS, corrupt
from zelos.vision.metrics import compute_metrics
from zelos.vision.scoring import fit_reference, score_image


def synth(seed: int, n: int = 384) -> np.ndarray:
    """Cell-like brightfield texture: many small dark/bright blobs on a flat background."""
    rng = np.random.default_rng(seed)
    import cv2

    img = np.full((n, n), 0.55, np.float32)
    for _ in range(260):
        y, x = rng.integers(0, n, 2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(3, 9)), float(rng.uniform(0.3, 0.8)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 0.006, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


@pytest.fixture(scope="module")
def ref():
    rows = [compute_metrics(synth(i)) for i in range(12)]
    return fit_reference(pd.DataFrame(rows))


def test_blur_lowers_focus():
    a = compute_metrics(synth(1))
    b = compute_metrics(corrupt(synth(1), "gaussian_blur", 4, np.random.default_rng(0)))
    assert b["focus_lapvar_norm"] < a["focus_lapvar_norm"]
    assert b["focus_hf_ratio"] < a["focus_hf_ratio"]


def test_noise_raises_noise_sigma():
    a = compute_metrics(synth(2))
    b = compute_metrics(corrupt(synth(2), "gaussian_noise", 3, np.random.default_rng(0)))
    assert b["noise_sigma_norm"] > a["noise_sigma_norm"]


def test_vignette_raises_illum_flatness():
    a = compute_metrics(synth(3))
    b = compute_metrics(corrupt(synth(3), "illumination", 5, np.random.default_rng(0)))
    assert b["illum_flatness"] > a["illum_flatness"]


def test_clean_mostly_pass(ref):
    levels = [score_image(compute_metrics(synth(100 + i)), ref)["level"] for i in range(6)]
    assert levels.count("PASS") >= 4


def test_heavy_corruption_fails(ref):
    x = corrupt(synth(5), "gaussian_blur", 5, np.random.default_rng(0))
    assert score_image(compute_metrics(x), ref)["level"] == "FAIL"


@pytest.mark.parametrize("kind", ["gaussian_blur", "gaussian_noise", "illumination", "overexpose", "jpeg"])
def test_attribution_primary_label(ref, kind):
    x = corrupt(synth(7), kind, 5, np.random.default_rng(1))
    m = compute_metrics(x)
    s = score_image(m, ref)
    causes = attribute(m, s["zbad"])
    assert causes and KIND_TO_LABEL[kind] in [c["label"] for c in causes[:2]]


def test_corrupt_deterministic():
    for k in KINDS:
        a = corrupt(synth(9), k, 3, np.random.default_rng(42))
        b = corrupt(synth(9), k, 3, np.random.default_rng(42))
        assert np.array_equal(a, b)
