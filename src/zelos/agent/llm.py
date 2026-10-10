"""LLM access for Zelos: Workers AI or Cohere chat with a record/replay cache.

Inputs : chat messages (OpenAI style), sampling params, optional provider ("workers_ai" | "cohere") and model.
Outputs: dict(text, usage{prompt_tokens, completion_tokens}, latency_ms, model, provider, cached).
Providers: workers_ai (default; model ids are used verbatim, e.g. "@cf/openai/gpt-oss-120b") and cohere
(POST https://api.cohere.com/v2/chat, secret COHERE). The default provider/model come from configs/default.yaml and
can be overridden by env ZELOS_LLM_PROVIDER / ZELOS_LLM_MODEL or, inside a block, by `use(provider, model)`.
Responses from reasoning models (text in `reasoning`, `content: null`) are normalised to the final answer text.

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
PROVIDERS = ("workers_ai", "cohere")
COHERE_URL = "https://api.cohere.com/v2/chat"
COHERE_DEFAULT_MODEL = "command-a-03-2025"
RETRY_STATUS = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 3
_BACKOFF_S = 1.5  # tests set this to 0


class LLMUnavailable(RuntimeError):
    """Raised when no LLM response can be produced (no keys, cache miss in replay, API failure)."""


class LLMQuotaExceeded(LLMUnavailable):
    """The provider reports an exhausted quota/allowance (not a transient rate limit)."""


class LLMBudgetExceeded(LLMUnavailable):
    """The per-process live-call budget set with set_call_budget() would be exceeded."""


# ---- live-call ledger and budget (process-local) ----------------------------------------------------------------
LEDGER: list[dict] = []
_BUDGET: dict[str, int] = {}
_override: dict[str, Any] = {"provider": None, "model": None, "max_tokens": None, "extra": None}


def set_call_budget(provider: str, max_http_calls: int | None) -> None:
    """Cap HTTP attempts (including retries) to a provider in this process; None removes the cap."""
    if max_http_calls is None:
        _BUDGET.pop(provider, None)
    else:
        _BUDGET[provider] = int(max_http_calls)


def calls_made(provider: str | None = None, model: str | None = None) -> int:
    return sum(1 for r in LEDGER if (provider is None or r["provider"] == provider)
               and (model is None or r["model"] == model))


class use:  # noqa: N801 - context manager used like a function
    """Temporarily set the default provider/model (and, for reasoning models, a max_tokens floor and extra
    request-body fields such as {"reasoning_effort": "low"}) for chat() calls. Used by the model sweep so the
    agent loop can run unmodified against another model."""

    def __init__(self, provider: str | None = None, model: str | None = None, max_tokens: int | None = None,
                 extra: dict | None = None):
        self.new = {"provider": provider, "model": model, "max_tokens": max_tokens, "extra": extra}

    def __enter__(self):
        self.old = dict(_override)
        _override.update(self.new)
        return self

    def __exit__(self, *exc):
        _override.update(self.old)


def cache_dir() -> Path:
    return Path(os.environ.get("ZELOS_LLM_CACHE_DIR") or DEFAULT_CACHE_DIR)


def default_provider() -> str:
    p = _override["provider"] or os.environ.get("ZELOS_LLM_PROVIDER") or load_config()["llm"].get("provider", "workers_ai")
    return p if p in PROVIDERS else "workers_ai"


def default_model(provider: str | None = None) -> str:
    provider = provider or default_provider()
    if _override["model"]:
        return _override["model"]
    if os.environ.get("ZELOS_LLM_MODEL"):
        return os.environ["ZELOS_LLM_MODEL"]
    if provider == "cohere":
        return COHERE_DEFAULT_MODEL
    return load_config()["llm"]["model"]


def mode() -> str:
    m = (os.environ.get("ZELOS_LLM_MODE") or "auto").lower()
    return m if m in {"auto", "live", "replay"} else "auto"


def cache_key(model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool,
              provider: str = "workers_ai", extra: dict | None = None) -> str:
    """sha256 over (provider, model, messages, params). For workers_ai the provider field (and an empty `extra`)
    is omitted so keys recorded before provider support existed still match."""
    d: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                         "max_tokens": max_tokens, "json_mode": json_mode}
    if provider != "workers_ai":
        d["provider"] = provider
    if extra:
        d["extra"] = extra
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


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


def have_keys(provider: str = "workers_ai") -> bool:
    if provider == "cohere":
        return bool(secret("COHERE"))
    return bool(secret("CLOUDFLARE_ACCOUNT_ID") and secret("CLOUDFLARE_API_TOKEN"))


# ---- response parsing (pure functions, unit-tested) --------------------------------------------------------------
def _text_of(x: Any) -> str:
    """Flatten a chat `content` value (str | list of parts | dict | None) to plain text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if isinstance(x.get("text"), str):
            return x["text"]
        if "content" in x:
            return _text_of(x["content"])
        return json.dumps(x)
    if isinstance(x, list):
        parts = []
        for p in x:
            if isinstance(p, dict) and str(p.get("type", "text")).lower() in ("thinking", "reasoning"):
                continue
            parts.append(_text_of(p))
        return "".join(parts)
    return str(x)


def _json_tail(text: str) -> str:
    """Return the last top-level {...} object in text that parses as JSON and mentions claims/abstain, else ''."""
    dec = json.JSONDecoder()
    best = ""
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = dec.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(obj, dict) and ("claims" in obj or "abstain" in obj):
            best = text[i:i + end]
    return best


def parse_workers_response(j: dict) -> str:
    """Final answer text from an OpenAI-compatible chat completion. Never raises; '' when nothing usable."""
    try:
        msg = (j.get("choices") or [{}])[0].get("message") or {}
    except Exception:  # noqa: BLE001
        msg = {}
    text = _text_of(msg.get("content")).strip()
    if text:
        return text
    # Native Workers AI shape: output = [{type: reasoning, ...}, {type: message, content: [{type: output_text, text}]}]
    out = j.get("output")
    if isinstance(out, list):
        msgs = [o for o in out if isinstance(o, dict) and o.get("type") == "message"]
        text = "".join(_text_of(o.get("content")) for o in msgs).strip()
        if text:
            return text
    # Reasoning-only response: the answer object may be at the end of the reasoning text.
    for k in ("reasoning_content", "reasoning"):
        r = _text_of(msg.get(k))
        if r.strip():
            tail = _json_tail(r)
            if tail:
                return tail
    return ""


def parse_cohere_response(j: dict) -> tuple[str, dict, dict]:
    """(text, usage, meta) from a Cohere v2 chat response; usage uses prompt_tokens/completion_tokens."""
    msg = j.get("message") or {}
    content = msg.get("content")
    text = _text_of(content).strip()
    if not text and isinstance(msg.get("tool_plan"), str):
        text = msg["tool_plan"].strip()
    toks = (j.get("usage") or {}).get("tokens") or (j.get("usage") or {}).get("billed_units") or {}
    return (text, {"prompt_tokens": int(toks.get("input_tokens") or 0),
                   "completion_tokens": int(toks.get("output_tokens") or 0)},
            {"finish_reason": j.get("finish_reason")})


def _is_quota(status: int, body: str) -> bool:
    """True for an exhausted allowance (daily neurons, monthly trial calls), False for a per-minute rate limit."""
    b = (body or "").lower()
    if "minute" in b or "per second" in b:
        return False
    return any(w in b for w in ("neurons", "free allocation", "daily", "per month", "/ month", "monthly", "quota"))


def _post(provider: str, model: str, url: str, headers: dict, body: dict) -> Any:
    """POST with a ledger entry per HTTP attempt and a budget check. Returns the requests.Response."""
    import requests

    cap = _BUDGET.get(provider)
    if cap is not None and calls_made(provider) >= cap:
        raise LLMBudgetExceeded(f"{provider} live-call budget of {cap} reached")
    t0 = time.perf_counter()
    rec = {"provider": provider, "model": model, "status": None, "latency_ms": 0.0}
    LEDGER.append(rec)
    try:
        r = requests.post(url, headers=headers, json=body, timeout=90)
        rec["status"] = r.status_code
        return r
    finally:
        rec["latency_ms"] = (time.perf_counter() - t0) * 1000.0


def _call_with_retry(provider: str, model: str, url: str, headers: dict, body: dict, parse) -> dict:
    last = "no attempt"
    for attempt in range(MAX_ATTEMPTS):
        t0 = time.perf_counter()
        try:
            r = _post(provider, model, url, headers, body)
        except LLMUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - network error: retry, then normalise
            last = type(e).__name__
            time.sleep(_BACKOFF_S * (attempt + 1))
            continue
        if r.status_code == 400 and "response_format" in body:
            body.pop("response_format")  # model may not support JSON mode; rely on the prompt
            last = "HTTP 400 (response_format dropped)"
            continue
        if r.status_code in RETRY_STATUS:
            if _is_quota(r.status_code, r.text[:600]):
                raise LLMQuotaExceeded(f"{provider} quota exhausted (HTTP {r.status_code})")
            last = f"HTTP {r.status_code}"
            try:
                wait = float(r.headers.get("retry-after", 0))
            except (TypeError, ValueError):
                wait = 0.0
            time.sleep(min(max(wait, _BACKOFF_S * (attempt + 1)) if _BACKOFF_S else 0.0, 30.0))
            continue
        if r.status_code >= 400:
            if _is_quota(r.status_code, r.text[:600]):
                raise LLMQuotaExceeded(f"{provider} quota exhausted (HTTP {r.status_code})")
            raise LLMUnavailable(f"{provider} call failed: HTTP {r.status_code}")
        try:
            text, usage, meta = parse(r.json())
        except Exception as e:  # noqa: BLE001
            raise LLMUnavailable(f"{provider} returned an unreadable response: {type(e).__name__}") from e
        return {"text": text, "usage": usage, "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "model": model, "provider": provider, "meta": meta}
    raise LLMUnavailable(f"{provider} call failed after {MAX_ATTEMPTS} attempts: {last}")


def _live_call(model: str, messages: list[dict], temperature: float, max_tokens: int, json_mode: bool,
               provider: str = "workers_ai", extra: dict | None = None) -> dict:
    if provider == "cohere":
        key = secret("COHERE")
        if not key:
            raise LLMUnavailable("Cohere credentials are not configured")
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(extra or {})
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        return _call_with_retry("cohere", model, COHERE_URL, headers, body, parse_cohere_response)
    acct, tok = secret("CLOUDFLARE_ACCOUNT_ID"), secret("CLOUDFLARE_API_TOKEN")
    if not (acct and tok):
        raise LLMUnavailable("Cloudflare Workers AI credentials are not configured")
    url = f"https://api.cloudflare.com/client/v4/accounts/{acct}/ai/v1/chat/completions"
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    body.update(extra or {})
    headers = {"Authorization": f"Bearer {tok}", "Content-Type": "application/json"}

    def parse(j: dict) -> tuple[str, dict, dict]:
        u = j.get("usage") or {}
        try:
            fin = (j.get("choices") or [{}])[0].get("finish_reason")
        except Exception:  # noqa: BLE001
            fin = None
        return (parse_workers_response(j), {"prompt_tokens": int(u.get("prompt_tokens") or 0),
                                            "completion_tokens": int(u.get("completion_tokens") or 0)},
                {"finish_reason": fin, **({"neurons": u["neurons"]} if isinstance(u.get("neurons"), (int, float)) else {})})

    return _call_with_retry("workers_ai", model, url, headers, body, parse)


def chat(messages: list[dict], *, model: str | None = None, temperature: float = 0.0,
         max_tokens: int = 600, json_mode: bool = False, provider: str | None = None,
         extra: dict | None = None) -> dict:
    """Return {text, usage, latency_ms, model, provider, cached}. See module docstring for modes."""
    provider = provider or default_provider()
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")
    model = model or default_model(provider)
    max_tokens = int(_override["max_tokens"] or max_tokens)
    extra = extra or _override["extra"] or None
    key = cache_key(model, messages, temperature, max_tokens, json_mode, provider, extra)
    m = mode()
    if m in ("auto", "replay"):
        hit = _cache_get(key)
        if hit is not None:
            return {"text": hit["text"], "usage": hit["usage"], "latency_ms": hit.get("latency_ms", 0.0),
                    "model": hit.get("model", model), "provider": hit.get("provider", provider),
                    "meta": hit.get("meta") or {}, "cached": True}
        if m == "replay":
            raise LLMUnavailable("cache miss in replay mode")
    if not have_keys(provider):
        raise LLMUnavailable("no cached response and no credentials")
    res = _live_call(model, messages, temperature, max_tokens, json_mode, provider, extra)
    _cache_put(key, {**res, "request": {"temperature": temperature, "max_tokens": max_tokens,
                                         "json_mode": json_mode, "n_messages": len(messages),
                                         **({"extra": extra} if extra else {})}})
    return {**res, "cached": False}
