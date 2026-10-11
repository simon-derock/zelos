"""Culture-state signal: frozen DINOv2-small embedding + non-parametric kNN against the labelled image bank.

This is the second certificate output. It answers "does this culture resemble expert-accepted or
expert-rejected cultures?", which is a different question from "is the image technically sound?"
(see E2: the artifact score does not predict the expert label, kNN on embeddings does).
Nothing is trained: DINOv2 (Apache-2.0) is frozen, kNN is a lookup. torch/transformers are optional
imports; without them (e.g. the light Space image) `culture_state` returns None.

culture_state(gray_uint8) -> {"p_bad": float, "state": "TYPICAL_GOOD|UNCERTAIN|TYPICAL_BAD", "k": 15, "bank": n} | None
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from zelos.common.config import ROOT

BANK = ROOT / "results" / "culture_bank.npz"
H, W = 448, 588
DINO_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"  # facebook/dinov2-small (Apache-2.0), pinned commit


@lru_cache
def _bank():
    if not BANK.exists():
        return None
    z = np.load(BANK, allow_pickle=False)
    e = z["cls"].astype(np.float32)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return e, z["bad"].astype(np.float32)


@lru_cache
def _model():
    try:
        import torch
        from transformers import AutoModel

        torch.set_num_threads(2)
        return AutoModel.from_pretrained("facebook/dinov2-small", revision=DINO_REV).eval()
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def embed(gray_u8: np.ndarray) -> np.ndarray | None:
    m = _model()
    if m is None:
        return None
    import cv2
    import torch

    x = cv2.resize(gray_u8, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=2)
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    with torch.no_grad():
        h = m(pixel_values=torch.from_numpy(x.transpose(2, 0, 1)[None])).last_hidden_state[0]
    return h[0].numpy()


def culture_state(gray_u8: np.ndarray, k: int = 15, lo: float = 0.3, hi: float = 0.7) -> dict[str, Any] | None:
    b = _bank()
    if b is None:
        return None
    v = embed(gray_u8)
    if v is None:
        return None
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    sim = b[0] @ v
    idx = np.argsort(-sim)[:k]
    p = float(b[1][idx].mean())
    state = "TYPICAL_BAD" if p >= hi else "TYPICAL_GOOD" if p <= lo else "UNCERTAIN"
    return {"p_bad": round(p, 3), "state": state, "k": k, "bank": int(len(b[1]))}
