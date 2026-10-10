"""Public-demo hardening: CORS, rate limiting, upload validation, session triage. No network."""
import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from zelos.api import main as api


def _field(seed: int, blur: float = 0.0, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur((rng.random((size, size)) * 255).astype(np.uint8), (0, 0), 1.2 + 0.1 * (seed % 3))
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img


def _enc(img: np.ndarray, ext: str = ".png") -> bytes:
    return cv2.imencode(ext, img)[1].tobytes()


def _client(**env: str) -> TestClient:
    base = {"ZELOS_ENV": "dev", "ZELOS_PRELOAD": "0", "ZELOS_RATE_PER_MIN": "1000"}
    return TestClient(api.create_app({**base, **env}))


def _files(n: int, field: str = "files", outlier_blur: float = 0.0):
    out = []
    for i in range(n):
        blur = outlier_blur if (outlier_blur and i == n - 1) else 0.0
        out.append((field, (f"img{i}.png", _enc(_field(i, blur)), "image/png")))
    return out


# ---------------------------------------------------------------- health
def test_health_reports_version_and_retriever_state():
    body = _client().get("/api/health").json()
    assert body["status"] == "ok" and body["version"] and body["retriever_loaded"] is False
    assert body["limits"]["triage_images"] == [4, 12]


# ---------------------------------------------------------------- CORS
def test_cors_defaults_to_production_origins():
    c = _client(ZELOS_ENV="prod")
    ok = c.get("/api/health", headers={"Origin": "https://zelos.philipsimonderock.com"})
    assert ok.headers.get("access-control-allow-origin") == "https://zelos.philipsimonderock.com"
    ok2 = c.get("/api/health", headers={"Origin": "https://papaya-figolla-a9dd1f.netlify.app"})
    assert ok2.headers.get("access-control-allow-origin") == "https://papaya-figolla-a9dd1f.netlify.app"
    bad = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_cors_wildcard_only_in_dev_and_env_override():
    assert _client(ZELOS_ENV="dev").get("/api/health", headers={"Origin": "https://x.example"}).headers[
        "access-control-allow-origin"] == "*"
    c = _client(ZELOS_ENV="prod", ZELOS_CORS_ORIGINS="https://a.example, https://b.example")
    assert c.get("/api/health", headers={"Origin": "https://b.example"}).headers[
        "access-control-allow-origin"] == "https://b.example"
    assert "access-control-allow-origin" not in c.get("/api/health", headers={"Origin": "https://zelos.philipsimonderock.com"}).headers


def test_cors_preflight_and_localhost_in_prod():
    c = _client(ZELOS_ENV="prod")
    r = c.options("/api/explain", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:5173"


# ---------------------------------------------------------------- rate limit
def test_rate_limit_returns_429_with_retry_after():
    c = _client(ZELOS_RATE_PER_MIN="3")
    png = _enc(_field(0))
    codes = [c.post("/api/inspect", files={"file": ("a.png", png, "image/png")}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    r = c.post("/api/inspect", files={"file": ("a.png", png, "image/png")})
    assert int(r.headers["retry-after"]) >= 1
    assert c.get("/api/health").status_code == 200  # health is never limited


def test_sliding_window_expires_and_isolates_clients():
    w = api.SlidingWindow(2, window=10)
    assert w.hit("a", now=0) is None and w.hit("a", now=1) is None
    assert w.hit("a", now=2) == pytest.approx(8.0)
    assert w.hit("b", now=2) is None
    assert w.hit("a", now=10.5) is None  # first hit expired


def test_rate_limit_keys_on_forwarded_client_behind_proxy():
    c = _client(ZELOS_RATE_PER_MIN="1", ZELOS_PROXY_HOPS="1", ZELOS_RATE_GLOBAL_PER_MIN="100")
    png = _enc(_field(0))

    def go(ip):
        return c.post("/api/inspect", files={"file": ("a.png", png, "image/png")},
                      headers={"X-Forwarded-For": f"9.9.9.9, {ip}"}).status_code
    assert go("1.1.1.1") == 200 and go("1.1.1.1") == 429 and go("2.2.2.2") == 200


# ---------------------------------------------------------------- upload validation
def test_rejects_non_images_and_unsupported_types():
    c = _client()
    for blob in (b"not an image", b"GIF89a" + b"\x00" * 64, b"%PDF-1.7 " + b"\x00" * 64, b""):
        r = c.post("/api/inspect", files={"file": ("x.png", blob, "image/png")})
        assert r.status_code == 400, blob[:6]


def test_rejects_truncated_png_header():
    r = _client().post("/api/inspect", files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 6, "image/png")})
    assert r.status_code == 400


def test_rejects_decompression_bomb_header():
    png = bytearray(_enc(_field(0)))
    png[16:24] = struct.pack(">II", 30000, 30000)  # 900 MP declared
    r = _client().post("/api/inspect", files={"file": ("bomb.png", bytes(png), "image/png")})
    assert r.status_code == 400 and "megapixel" in r.json()["detail"]


def test_rejects_oversize_upload(monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 10_000)
    big = _enc(np.random.default_rng(1).integers(0, 255, (400, 400), dtype=np.uint8))
    assert len(big) > 10_000
    r = _client().post("/api/inspect", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


def test_accepts_png_jpeg_tiff():
    c = _client()
    img = _field(3)
    for ext, mime in ((".png", "image/png"), (".jpg", "image/jpeg"), (".tif", "image/tiff")):
        r = c.post("/api/inspect", files={"file": (f"x{ext}", _enc(img, ext), mime)})
        assert r.status_code == 200, (ext, r.text)
        assert r.json()["image"]["shape"] == [256, 256]


def test_header_parsers_agree_with_decoder():
    img = np.zeros((90, 140), np.uint8)
    for ext in (".png", ".jpg", ".tif"):
        assert api.image_dims(_enc(img, ext)) == (140, 90)


def test_filename_is_sanitised():
    r = _client().post("/api/inspect", files={"file": ("../../etc/pa\x00ss.png", _enc(_field(0)), "image/png")})
    assert r.status_code == 200 and "/" not in r.json()["image"]["id"]


# ---------------------------------------------------------------- explain / triage
def test_explain_still_returns_agent_result_and_qc():
    r = _client().post("/api/explain?provider=template", files={"file": ("a.png", _enc(_field(0)), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert {"card", "ledger", "trace", "usage", "evidence", "qc"} <= set(body)
    assert body["qc"]["tool_call_id"] == body["card"]["tool_call_id"]


def test_explain_rejects_unknown_provider():
    r = _client().post("/api/explain?provider=nope", files={"file": ("a.png", _enc(_field(0)), "image/png")})
    assert r.status_code == 400


def test_triage_needs_four_to_twelve_images():
    c = _client()
    assert c.post("/api/triage", files=_files(3)).status_code == 400
    assert c.post("/api/triage", files=_files(13)).status_code == 400
    assert c.post("/api/triage", files=_files(1)).status_code == 400


def test_triage_returns_batch_relative_verdicts():
    c = _client()
    r = c.post("/api/triage", files=_files(6, outlier_blur=6.0))
    assert r.status_code == 200
    rows = r.json()
    assert [x["id"] for x in rows] == [f"img{i}" for i in range(6)]
    for x in rows:
        assert {"id", "level", "qc_score", "causes", "density_state", "tool_call_id"} <= set(x)
        assert x["level"] in {"PASS", "REVIEW", "FAIL"} and x["tool_call_id"].startswith("qc-")
        assert x["reference"] == "batch:n=6"
    assert rows[-1]["qc_score"] > max(x["qc_score"] for x in rows[:-1])  # the blurred frame stands out


def test_triage_dedupes_ids_and_validates_each_file():
    c = _client()
    same = [("files", ("a.png", _enc(_field(i)), "image/png")) for i in range(4)]
    ids = [x["id"] for x in c.post("/api/triage", files=same).json()]
    assert ids == ["a", "a-2", "a-3", "a-4"]
    bad = _files(5)
    bad[2] = ("files", ("junk.png", b"nope", "image/png"))
    r = c.post("/api/triage", files=bad)
    assert r.status_code == 400 and "junk" in r.json()["detail"]


def test_triage_costs_one_rate_unit_per_image():
    c = _client(ZELOS_RATE_PER_MIN="10")
    assert c.post("/api/triage", files=_files(6)).status_code == 200
    assert c.post("/api/triage", files=_files(6)).status_code == 429


def test_body_limit_aborts_before_parsing(monkeypatch):
    monkeypatch.setattr(api, "MAX_BYTES", 5_000)
    monkeypatch.setattr(api, "FORM_OVERHEAD", 0)
    blob = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50_000
    r = _client().post("/api/inspect", files={"file": ("big.png", blob, "image/png")})
    assert r.status_code == 413


def test_root_describes_service():
    body = _client().get("/").json()
    assert body["name"] == "Zelos" and body["health"] == "/api/health"
