"""Config, paths, secrets and provenance helpers shared by every Zelos module.

Inputs : configs/*.yaml, optional .env (COHERE, CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN)
Outputs: dicts / strings; never prints secrets.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from datetime import UTC
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = ROOT / "configs"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@lru_cache
def load_config(name: str = "default") -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())


def secret(name: str) -> str | None:
    """Return a secret from the environment (.env is loaded at import). None if unset."""
    v = os.environ.get(name)
    return v or None


def seed_everything(seed: int | None = None) -> int:
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def config_hash() -> str:
    blob = b"".join(p.read_bytes() for p in sorted(CONFIGS.glob("*.yaml")))
    return sha256_bytes(blob)[:16]


def provenance(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance block written into every results file."""
    import platform
    from datetime import datetime

    out = {
        "git_sha": git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_hash": config_hash(),
        "python": platform.python_version(),
        "seed": load_config()["seed"],
    }
    if extra:
        out.update(extra)
    return out


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=float) + "\n")
