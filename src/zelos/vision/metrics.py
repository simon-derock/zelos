"""Training-free QC metric suite for brightfield OoC images.

Input : 2-D grayscale image (uint8/uint16/float) — RGB is converted with luminance weights.
Output: dict[str, float] of physically interpretable, unit-less ("relative") measurements.

Every metric is classical image statistics; nothing is learned. Formulas:
- focus_lapvar_norm   var(Laplacian(I)) / var(I)                 (contrast-invariant sharpness)
- focus_tenengrad_norm mean(Gx^2+Gy^2) / var(I)                  (Sobel gradient energy)
- focus_hf_ratio      FFT power above 0.25*Nyquist / total power (DC excluded)
- contrast_rms        std(I) on [0,1] scale
- contrast_p99_p1     p99(I) - p1(I)
- mean_intensity      mean(I)
- clip_low_frac / clip_high_frac   fraction of pixels at the dtype min / max
- illum_flatness      (max-min)/mean of a degree-2 polynomial fit to a 64x64 area-downsampled image
- noise_sigma_norm    Immerkaer (1996) fast noise estimate / std(I)
- debris_count, debris_area_frac   small dark compact blobs found by black-hat + 5-sigma threshold
- bubble_score        area fraction of the largest round, texture-free region (local std below 10% of median)
- foreground_frac     fraction of pixels whose local std exceeds the Otsu threshold (confluence proxy)
- blob_density_per_mpx DoG local maxima per megapixel of the analysed image
- texture_lbp_entropy entropy of the uniform LBP histogram (8 neighbours, radius 1)
- jpeg_blockiness     mean |step| across 8x8 block boundaries / mean |step| elsewhere
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_EPS = 1e-12


def to_gray01(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (float32 image in [0,1], dtype max) for uint8/uint16/float inputs."""
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        mx = 255.0
    elif img.dtype == np.uint16:
        mx = 65535.0
    else:
        mx = 1.0
    return (img.astype(np.float32) / mx), mx


def downscale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w) / max_side
    if s <= 1:
        return img
    return cv2.resize(img, (round(w / s), round(h / s)), interpolation=cv2.INTER_AREA)


def _local_std(img: np.ndarray, k: int = 15) -> np.ndarray:
    m = cv2.blur(img, (k, k))
    m2 = cv2.blur(img * img, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def _otsu(x: np.ndarray) -> float:
    hist, edges = np.histogram(x, bins=256)
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, _EPS)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, _EPS)
    between = w0 * w1 * (m0 - m1) ** 2
    return float(centers[int(np.argmax(between))])


def focus_hf_ratio(img: np.ndarray, cutoff: float = 0.25) -> float:
    h, w = img.shape
    c = min(h, w, 512)
    y0, x0 = (h - c) // 2, (w - c) // 2
    crop = img[y0 : y0 + c, x0 : x0 + c]
    crop = (crop - crop.mean()) * np.hanning(c)[:, None] * np.hanning(c)[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(crop))) ** 2
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - c / 2, xx - c / 2) / (c / 2)
    p[c // 2, c // 2] = 0
    return float(p[r > cutoff].sum() / max(p.sum(), _EPS))


def illum_flatness(img: np.ndarray) -> float:
    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64)
    yy, xx = np.mgrid[0:64, 0:64] / 63.0
    A = np.stack([np.ones_like(xx), xx, yy, xx * xx, yy * yy, xx * yy], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, small.ravel(), rcond=None)
    fit = A @ coef
    return float((fit.max() - fit.min()) / max(fit.mean(), _EPS))


def noise_sigma(img: np.ndarray) -> float:
    """Immerkaer, 'Fast noise variance estimation', CVIU 1996."""
    m = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = img.shape
    conv = cv2.filter2D(img, -1, m, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def debris(img: np.ndarray) -> tuple[float, float]:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bh = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, k)
    thr = float(np.median(bh) + 5 * 1.4826 * np.median(np.abs(bh - np.median(bh))) + 0.05)
    mask = (bh > thr).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    ww = stats[1:, cv2.CC_STAT_WIDTH] if n > 1 else np.array([])
    hh = stats[1:, cv2.CC_STAT_HEIGHT] if n > 1 else np.array([])
    fill = area / np.maximum(ww * hh, 1)
    keep = (area >= 6) & (area <= 900) & (fill > 0.45)
    return float(keep.sum()), float(area[keep].sum() / img.size)


def bubble_score(img: np.ndarray) -> float:
    ls = _local_std(img, 9)
    flat = (ls < 0.10 * max(float(np.median(ls)), 1e-4)).astype(np.uint8)
    flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=8)
    best = 0.0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 0.002 * img.size:
            continue
        comp = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = cv2.arcLength(cnts[0], True)
        circ = 4 * math.pi * a / max(per * per, _EPS)
        if circ > 0.6:
            best = max(best, a / img.size)
    return float(best)


def blob_density(img: np.ndarray) -> float:
    g1 = cv2.GaussianBlur(img, (0, 0), 2)
    g2 = cv2.GaussianBlur(img, (0, 0), 4)
    dog = np.abs(g1 - g2)
    mx = cv2.dilate(dog, np.ones((7, 7), np.uint8))
    thr = dog.mean() + 2 * dog.std()
    n = int(((dog == mx) & (dog > thr)).sum())
    return n / (img.size / 1e6)


def lbp_entropy(img: np.ndarray) -> float:
    small = downscale(img, 256)
    c = small[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for b, (dy, dx) in enumerate(offs):
        nb = small[1 + dy : small.shape[0] - 1 + dy, 1 + dx : small.shape[1] - 1 + dx]
        code |= ((nb >= c).astype(np.uint8) << b)
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def jpeg_blockiness(img: np.ndarray) -> float:
    dx = np.abs(np.diff(img, axis=1))
    cols = np.arange(dx.shape[1])
    on = dx[:, (cols % 8) == 7].mean()
    off = dx[:, (cols % 8) != 7].mean()
    dy = np.abs(np.diff(img, axis=0))
    rows = np.arange(dy.shape[0])
    on2 = dy[(rows % 8) == 7, :].mean()
    off2 = dy[(rows % 8) != 7, :].mean()
    return float((on + on2) / max(off + off2, _EPS))


def compute_metrics(img: np.ndarray, max_side: int = 1024) -> dict[str, float]:
    """Compute every QC metric for one image. See module docstring for formulas."""
    g, mx = to_gray01(img)
    raw = img if img.ndim == 2 else cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    if raw.dtype.kind in "ui":
        clip_low = float((raw == 0).mean())
        clip_high = float((raw == int(mx)).mean())
    else:
        clip_low = float((g <= 0.0).mean())
        clip_high = float((g >= 1.0).mean())
    blk = jpeg_blockiness(g)
    s = downscale(g, max_side)
    var = float(s.var()) + _EPS
    lap = cv2.Laplacian(s, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)
    ls = _local_std(s, 15)
    deb_n, deb_a = debris(s)
    sd = math.sqrt(var)
    return {
        "focus_lapvar_norm": float(lap.var() / var),
        "focus_tenengrad_norm": float((gx * gx + gy * gy).mean() / var),
        "focus_hf_ratio": focus_hf_ratio(s),
        "contrast_rms": sd,
        "contrast_p99_p1": float(np.percentile(s, 99) - np.percentile(s, 1)),
        "mean_intensity": float(s.mean()),
        "clip_low_frac": clip_low,
        "clip_high_frac": clip_high,
        "illum_flatness": illum_flatness(s),
        "noise_sigma_norm": noise_sigma(s) / sd,
        "debris_count": deb_n,
        "debris_area_frac": deb_a,
        "bubble_score": bubble_score(s),
        "foreground_frac": float((ls > _otsu(ls)).mean()),
        "blob_density_per_mpx": blob_density(s),
        "texture_lbp_entropy": lbp_entropy(s),
        "jpeg_blockiness": blk,
    }


# Direction in which each metric indicates a *problem* (+1: high is bad, -1: low is bad, 0: two-sided/info).
BAD_DIRECTION: dict[str, int] = {
    "focus_lapvar_norm": -1,
    "focus_tenengrad_norm": -1,
    "focus_hf_ratio": -1,
    "contrast_rms": -1,
    "contrast_p99_p1": -1,
    "mean_intensity": 0,
    "clip_low_frac": +1,
    "clip_high_frac": +1,
    "illum_flatness": +1,
    "noise_sigma_norm": +1,
    "debris_count": +1,
    "debris_area_frac": +1,
    "bubble_score": +1,
    "foreground_frac": 0,
    "blob_density_per_mpx": 0,
    "texture_lbp_entropy": 0,
    "jpeg_blockiness": +1,
}
QUALITY_METRICS = [k for k, v in BAD_DIRECTION.items() if v != 0]
