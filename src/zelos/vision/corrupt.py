"""Controlled corruptions with known ground truth for the E1 benchmark.

corrupt(img01, kind, severity in 1..5, rng) -> corrupted float32 image in [0,1].
KIND_TO_LABEL maps each corruption to the cause label Zelos should attribute.
"""
from __future__ import annotations

import cv2
import numpy as np

KIND_TO_LABEL = {
    "gaussian_blur": "DEFOCUS_OR_BLUR",
    "defocus": "DEFOCUS_OR_BLUR",
    "motion_blur": "DEFOCUS_OR_BLUR",
    "gaussian_noise": "NOISE",
    "illumination": "UNEVEN_ILLUMINATION",
    "contrast": "LOW_CONTRAST",
    "overexpose": "OVER_EXPOSED",
    "underexpose": "UNDER_EXPOSED",
    "debris": "DEBRIS",
    "bubble": "BUBBLE",
    "jpeg": "COMPRESSION",
}
KINDS = list(KIND_TO_LABEL)


def corrupt(img: np.ndarray, kind: str, severity: int, rng: np.random.Generator) -> np.ndarray:
    s = int(severity) - 1
    x = img.astype(np.float32).copy()
    h, w = x.shape
    if kind == "gaussian_blur":
        x = cv2.GaussianBlur(x, (0, 0), [1.0, 1.8, 2.8, 4.0, 6.0][s])
    elif kind == "defocus":
        r = [2, 3, 5, 7, 10][s]
        k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(k, (r, r), r, 1.0, -1)
        x = cv2.filter2D(x, -1, k / k.sum(), borderType=cv2.BORDER_REFLECT)
    elif kind == "motion_blur":
        n = [5, 9, 13, 19, 27][s]
        k = np.zeros((n, n), np.float32)
        k[n // 2, :] = 1.0
        m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        k = cv2.warpAffine(k, m, (n, n))
        x = cv2.filter2D(x, -1, k / max(k.sum(), 1e-6), borderType=cv2.BORDER_REFLECT)
    elif kind == "gaussian_noise":
        x = x + rng.normal(0, [0.02, 0.035, 0.05, 0.075, 0.11][s], x.shape).astype(np.float32)
    elif kind == "illumination":
        a = [0.25, 0.4, 0.55, 0.7, 0.85][s]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = rng.uniform(0, 2 * np.pi)
        ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
        r2 = ((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2
        field = 1 + a * ramp - a * 0.8 * r2
        x = x * field
    elif kind == "contrast":
        c = [0.6, 0.45, 0.32, 0.22, 0.12][s]
        x = (x - x.mean()) * c + x.mean()
    elif kind == "overexpose":
        x = x * [1.4, 1.7, 2.0, 2.5, 3.2][s]
    elif kind == "underexpose":
        x = x * [0.55, 0.4, 0.28, 0.18, 0.1][s] - [0.0, 0.01, 0.02, 0.03, 0.05][s]
    elif kind == "debris":
        n = [6, 12, 25, 45, 80][s]
        for _ in range(n):
            cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
            r = int(rng.integers(2, 6))
            cv2.circle(x, (cx, cy), r, float(max(0.0, x.mean() - 0.35)), -1)
    elif kind == "bubble":
        n = [1, 1, 2, 2, 3][s]
        rad = [45, 65, 75, 90, 110][s]
        for _ in range(n):
            cy, cx = int(rng.integers(rad, h - rad)), int(rng.integers(rad, w - rad))
            inner = float(min(1.0, x.mean() + 0.15))
            cv2.circle(x, (cx, cy), rad, float(max(0.0, x.mean() - 0.3)), -1)
            cv2.circle(x, (cx, cy), rad - 5, inner, -1)
    elif kind == "jpeg":
        q = [50, 30, 18, 10, 5][s]
        u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    else:
        raise ValueError(kind)
    return np.clip(x, 0, 1).astype(np.float32)
