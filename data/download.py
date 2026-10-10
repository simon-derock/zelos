"""Download the OOC image dataset (Movčana et al., Data 2024; Zenodo 10.5281/zenodo.10203721).

The full archive is 6.7 GB. By default this script fetches a *stratified, seeded subset*
directly out of the remote zip with HTTP byte-range requests (no full download), verifying
each member's CRC-32 against the zip central directory.

Inputs : --n (subset size, default 480), --all (fetch every image), --seed, --workers
Outputs: data/raw/OOC_datasheet.xlsx, data/raw/images/<split>/<class>/<cell>/<time>/<id>.png,
         data/raw/manifest.csv (one row per downloaded image, with md5/CRC and labels)

Usage:  python data/download.py --n 480
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remotezip import HTTPRangeFile  # noqa: E402

RECORD = "10203721"
ZIP_URL = f"https://zenodo.org/api/records/{RECORD}/files/OOC_image_dataset.zip/content"
XLSX_URL = f"https://zenodo.org/api/records/{RECORD}/files/OOC_datasheet.xlsx/content"
ZIP_SIZE = 6710767405
ZIP_MD5 = "8f7e058996203d48eb03b2d86c0a2e4d"
XLSX_MD5 = "a3d4875e3da4da83bac45c0ce727cb40"
ROOT = Path(__file__).resolve().parent / "raw"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_member(name: str) -> dict:
    """'OOC_image_dataset/test/bad/cell_type_A549/0-1_days/221123_1.png' -> fields."""
    parts = name.split("/")
    return {
        "zip_path": name,
        "split_orig": parts[1],
        "label": parts[2],
        "cell_line": parts[3].replace("cell_type_", ""),
        "time_bin": parts[4].replace("_days", ""),
        "file_id": Path(parts[5]).stem,
        "session": Path(parts[5]).stem.split("_")[0],
    }


def fetch_member(info: zipfile.ZipInfo, out: Path, session: requests.Session) -> None:
    """Fetch one deflated/stored member with a single range request and verify CRC."""
    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 4096 + info.compress_size
    for attempt in range(6):
        try:
            r = session.get(ZIP_URL, headers={"Range": f"bytes={start}-{min(end, ZIP_SIZE - 1)}"}, timeout=300)
            if r.status_code != 206:
                raise IOError(f"status {r.status_code}")
            buf = r.content
            sig, *_ = struct.unpack("<I", buf[:4])
            assert sig == 0x04034B50, "bad local header"
            n, m = struct.unpack("<HH", buf[26:30])
            raw = buf[30 + n + m : 30 + n + m + info.compress_size]
            data = zlib.decompress(raw, -15) if info.compress_type == 8 else raw
            if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
                raise IOError("CRC mismatch")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.rename(out)
            return
        except Exception as e:  # retry on any network/parse error
            err = e
    raise IOError(f"failed {info.filename}: {err}")


def stratified_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n rows balanced over (label, cell_line); keeps original test images preferentially
    proportional. Deterministic given the seed."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(["label", "cell_line"]))
    per = max(1, n // len(groups))
    picks = []
    for _, g in groups:
        k = min(len(g), per)
        picks.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    out = pd.concat(picks)
    rest = df.drop(out.index)
    if len(out) < n and len(rest):
        extra = rest.iloc[rng.choice(len(rest), size=min(n - len(out), len(rest)), replace=False)]
        out = pd.concat([out, extra])
    return out.sort_values("zip_path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    xlsx = ROOT / "OOC_datasheet.xlsx"
    if not xlsx.exists():
        xlsx.write_bytes(s.get(XLSX_URL, timeout=120).content)
    assert md5(xlsx) == XLSX_MD5, "datasheet checksum mismatch"

    z = zipfile.ZipFile(HTTPRangeFile(ZIP_URL, size=ZIP_SIZE, session=s))
    infos = {i.filename: i for i in z.infolist() if not i.is_dir()}
    df = pd.DataFrame([parse_member(k) for k in infos])
    sub = df if a.all else stratified_subset(df, a.n, a.seed)
    sub = sub.assign(local_path=[str(Path("images") / Path(p).relative_to("OOC_image_dataset")) for p in sub.zip_path])
    sub = sub.assign(crc32=[f"{infos[p].CRC:08x}" for p in sub.zip_path])

    todo = [(infos[p], ROOT / lp) for p, lp in zip(sub.zip_path, sub.local_path) if not (ROOT / lp).exists()]
    print(f"{len(df)} images in archive; subset {len(sub)}; to fetch {len(todo)}", flush=True)

    def job(t):
        fetch_member(t[0], t[1], requests.Session())
        return t[1]

    done = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for _ in ex.map(job, todo):
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
    sub.drop(columns=["zip_path"]).to_csv(ROOT / "manifest.csv", index=False)
    print("OK", len(sub), "images ->", ROOT / "manifest.csv")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
