"""LLM access for Zelos: Cloudflare Workers AI chat with a record/replay cache.

Inputs : chat messages (OpenAI style), sampling params.
Outputs: dict(text, usage{prompt_tokens, completion_tokens}, latency_ms, model, cached).

Mode (env ZELOS_LLM_MODE):
  auto   (default) cache hit -> replay; else live if keys are present; else raise LLMUnavailable
  live   always call the API (and record the response in the cache)
  replay cache only; a miss raises LLMUnavailable
Cache files live in eval/llm_cache/<sha256>.json (override dir with ZELOS_LLM_CACHE_DIR) and are
committed so published numbers reproduce without keys. Model output is treated as data, never as
instructions. Secrets are read from the environment and never logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from zelos.common.config import ROOT, load_config, secret

DEFAULT_CACHE_DIR = ROOT / "eval" / "llm_cache"


class LLMUnavailable(RuntimeError):
    """Raised when no LLM response can be produced (no keys, cache miss in replay, API failure)."""


def cache_dir() -> Path:
    return Path(os.environ.get("ZELOS_LLM_CACHE_DIR") or DEFAULT_CACHE_DIR)


def default_model() -> str:
    return load_config()["llm"]["model"]


def mode() -> str:
    m = (os.environ.get("ZELOS_LLM_MODE") or "auto").lower()
    return m if m in {"auto", "live", "replay"} else "auto"


def cache_key(model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool) -> str:
    blob = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature,
         "max_tokens": max_tokens, "json_mode": json_mode},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> dict | None:
    p = cache_dir() / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _cache_put(key: str, rec: dict) -> None:
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{key}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")


def have_keys() -> bool:
    return bool(secret("CLOUDFLARE_ACCOUNT_ID") and secret("CLOUDFLARE_API_TOKEN"))


def _live_call(model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool) -> dict:
    import requests

    acct, tok = secret("CLOUDFLARE_ACCOUNT_ID"), secret("CLOUDFLARE_API_TOKEN")
    if not (acct and tok):
        raise LLMUnavailable("Cloudflare Workers AI credentials are not configured")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                            "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}
    last: Exception | None = None
    for attempt in range(3):  # initial try + 2 retries
        t0 = time.perf_counter()
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")  # model may not support JSON mode; rely on the prompt
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                raise LLMUnavailable(f"HTTP {r.status_code}")
            r.raise_for_status()
            j = r.json()
            ch = j["choices"][0]["message"]["content"]
            if isinstance(ch, (dict, list)):
                ch = json.dumps(ch)
            u = j.get("usage") or {}
            return {
                "text": ch or "",
                "usage": {"prompt_tokens": int(u.get("prompt_tokens", 0)),
                          "completion_tokens": int(u.get("completion_tokens", 0))},
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "model": model,
            }
        except Exception as e:  # noqa: BLE001 - normalised to LLMUnavailable below
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise LLMUnavailable(f"Workers AI call failed: {type(last).__name__}")


def chat(messages: list[dict], *, model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 600, json_mode: bool = False) -> dict:
    """Return {text, usage, latency_ms, model, cached}. See module docstring for modes."""
    model = model or default_model()
    key = cache_key(model, messages, temperature, max_tokens, json_mode)
    m = mode()
    if m in ("auto", "replay"):
        hit = _cache_get(key)
        if hit is not None:
            return {"text": hit["text"], "usage": hit["usage"], "latency_ms": hit.get("latency_ms", 0.0),
                    "model": hit.get("model", model), "cached": True}
        if m == "replay":
            raise LLMUnavailable("cache miss in replay mode")
    if not have_keys():
        raise LLMUnavailable("no cached response and no credentials")
    res = _live_call(model, messages, temperature, max_tokens, json_mode)
    _cache_put(key, {**res, "request": {"temperature": temperature, "max_tokens": max_tokens,
                                         "json_mode": json_mode, "n_messages": len(messages)}})
    return {**res, "cached": False}
