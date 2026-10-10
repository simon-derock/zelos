import cv2
import numpy as np
from fastapi.testclient import TestClient

from zelos.api.main import app
from zelos.vision.metrics import compute_metrics


def _png() -> bytes:
    rng = np.random.default_rng(0)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health():
    assert TestClient(app).get("/api/health").json()["status"] == "ok"


def test_inspect_endpoint(tmp_path, monkeypatch):
    import pandas as pd

    from zelos.vision import pipeline
    from zelos.vision.scoring import fit_reference

    ref = fit_reference(pd.DataFrame([compute_metrics((np.random.default_rng(i).random((256, 256)) * 255).astype(np.uint8)) for i in range(5)]))
    monkeypatch.setattr(pipeline, "load_reference", lambda path=None: ref)
    r = TestClient(app).post("/api/inspect", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["level"] in {"PASS", "REVIEW", "FAIL"} and body["tool_call_id"].startswith("qc-")


def test_bad_upload():
    r = TestClient(app).post("/api/inspect", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
