"""Stream all 3,072 OoC images from the Zenodo zip and keep only features (images are never stored).

For every member: fetch with a CRC-verified byte-range request, compute the 17 physics QC metrics and a
frozen DINOv2-small embedding (CLS + mean patch token, 448x588 input), then discard the pixels.

Output: cache/features_all.pkl  (DataFrame: file_id, session, label, cell_line, time_bin, split_orig, 17 metrics,
        cls_0..383, mean_0..383).  Resumable: re-running only fetches what is missing.
Run   : python scripts/extract_features.py [--workers 10]
Nothing is trained; DINOv2 (Meta AI, Apache-2.0) is used frozen as a feature extractor.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
import sys
import time
import zipfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "src"))
from remotezip import HTTPRangeFile  # noqa: E402

from zelos.vision.metrics import compute_metrics  # noqa: E402

ZIP_URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
ZIP_SIZE = 6710767405
OUT = ROOT / "cache" / "features_all.pkl"
H, W = 448, 588
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

_model = None


def parse(name: str) -> dict:
    p = name.split("/")
    stem = Path(p[5]).stem
    return {"zip_path": name, "split_orig": p[1], "label": p[2], "cell_line": p[3].replace("cell_type_", ""),
            "time_bin": p[4].replace("_days", ""), "file_id": stem, "session": stem.split("_")[0]}


def fetch(entry: tuple) -> bytes:
    name, hdr, csize, ctype, crc = entry
    sess = requests.Session()
    end = min(hdr + 30 + len(name.encode()) + 4096 + csize, ZIP_SIZE - 1)
    err = None
    for _ in range(6):
        try:
            r = sess.get(ZIP_URL, headers={"Range": f"bytes={hdr}-{end}"}, timeout=300)
            if r.status_code != 206:
                raise OSError(f"status {r.status_code}")
            buf = r.content
            assert struct.unpack("<I", buf[:4])[0] == 0x04034B50
            n, m = struct.unpack("<HH", buf[26:30])
            raw = buf[30 + n + m : 30 + n + m + csize]
            data = zlib.decompress(raw, -15) if ctype == 8 else raw
            if zlib.crc32(data) & 0xFFFFFFFF != crc:
                raise OSError("crc")
            return data
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2)
    raise OSError(f"fetch failed {name}: {err}")


def dino(img_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _model
    import torch

    if _model is None:
        torch.set_num_threads(1)
        from transformers import AutoModel

        _model = AutoModel.from_pretrained("facebook/dinov2-small").eval()
    x = cv2.resize(img_gray, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = np.repeat(x[..., None], 3, axis=2)
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        h = _model(pixel_values=torch.from_numpy(x)).last_hidden_state[0]
    return h[0].numpy(), h[1:].mean(0).numpy()


def job(item: tuple) -> dict:
    meta, entry = item
    data = fetch(entry)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    m = compute_metrics(img, 1024)
    cls, mean = dino(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    row = {k: v for k, v in meta.items() if k != "zip_path"}
    row.update({"crc32": f"{entry[4]:08x}", "h": img.shape[0], "w": img.shape[1]})
    row.update(m)
    row.update({f"cls_{i}": float(v) for i, v in enumerate(cls)})
    row.update({f"mean_{i}": float(v) for i, v in enumerate(mean)})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.parent.mkdir(exist_ok=True)
    z = zipfile.ZipFile(HTTPRangeFile(ZIP_URL, size=ZIP_SIZE))
    infos = [i for i in z.infolist() if not i.is_dir()]
    done = pd.read_pickle(OUT) if OUT.exists() else pd.DataFrame()
    have = set(done["crc32"]) if len(done) else set()
    todo = [(parse(i.filename), (i.filename, i.header_offset, i.compress_size, i.compress_type, i.CRC))
            for i in infos if f"{i.CRC:08x}" not in have]
    print(f"{len(infos)} images, {len(have)} cached, {len(todo)} to extract", flush=True)
    rows = done.to_dict("records")
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(job, todo, chunksize=1), 1):
            rows.append(r)
            if k % 50 == 0 or k == len(todo):
                pd.DataFrame(rows).to_pickle(OUT)
                el = time.time() - t0
                print(f"  {k}/{len(todo)}  {el/60:.1f} min  eta {(len(todo)-k)*el/k/60:.1f} min", flush=True)
    pd.DataFrame(rows).to_pickle(OUT)
    print("done", len(rows), "rows ->", OUT)


if __name__ == "__main__":
    main()
