"""LLM provider tests: Workers AI / Cohere response parsing, record/replay cache and bounded retry. No network."""
import json

import pytest

from zelos.agent import llm

MSGS = [{"role": "user", "content": "hello"}]
CLAIMS = '{"claims": [{"text": "Focus is 0.41 [qc-1].", "type": "numeric", "evidence": ["qc-1"]}]}'


class FakeResp:
    def __init__(self, status=200, payload=None, text=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)
        self.headers = headers or {}

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZELOS_LLM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ZELOS_LLM_MODE", "auto")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    monkeypatch.setenv("COHERE", "ck")
    monkeypatch.delenv("ZELOS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ZELOS_LLM_MODEL", raising=False)
    monkeypatch.setattr(llm, "_BACKOFF_S", 0.0)
    llm.LEDGER.clear()
    llm._BUDGET.clear()
    yield
    llm.LEDGER.clear()
    llm._BUDGET.clear()


def patch_post(monkeypatch, responses):
    """Make requests.post return the given responses in order; returns the list of captured calls."""
    import requests

    calls = []
    it = iter(responses)

    def post(url, headers=None, json=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "headers": headers, "json": json})
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(requests, "post", post)
    return calls


def workers_ok(content, **msg):
    return FakeResp(200, {"choices": [{"message": {"role": "assistant", "content": content, **msg},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 11, "completion_tokens": 7, "neurons": 3.5}})


# ---- parsing -----------------------------------------------------------------------------------------------------
def test_workers_parse_plain_and_null_content():
    assert llm.parse_workers_response({"choices": [{"message": {"content": CLAIMS}}]}) == CLAIMS
    # content null, answer only inside the reasoning text: recover the final JSON object, never crash
    j = {"choices": [{"message": {"content": None, "reasoning": f"thinking... final answer: {CLAIMS} done"}}]}
    assert json.loads(llm.parse_workers_response(j))["claims"][0]["evidence"] == ["qc-1"]
    # content null and reasoning without an answer object -> empty string
    assert llm.parse_workers_response({"choices": [{"message": {"content": None, "reasoning": "hmm"}}]}) == ""
    assert llm.parse_workers_response({}) == ""
    assert llm.parse_workers_response({"choices": []}) == ""
    # list-of-parts content and the native `output` channel list
    parts = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    assert llm.parse_workers_response(parts) == "ab"
    native = {"output": [{"type": "reasoning", "content": [{"text": "r"}]},
                         {"type": "message", "content": [{"type": "output_text", "text": CLAIMS}]}]}
    assert llm.parse_workers_response(native) == CLAIMS


def test_cohere_parse_text_blocks_and_usage():
    j = {"message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "x"},
                                                      {"type": "text", "text": CLAIMS}]},
         "usage": {"tokens": {"input_tokens": 120, "output_tokens": 33}, "billed_units": {"input_tokens": 99}},
         "finish_reason": "COMPLETE"}
    text, usage, meta = llm.parse_cohere_response(j)
    assert text == CLAIMS and usage == {"prompt_tokens": 120, "completion_tokens": 33}
    assert meta["finish_reason"] == "COMPLETE"
    assert llm.parse_cohere_response({"message": {"content": []}})[0] == ""


# ---- live calls ----------------------------------------------------------------------------------------------------
def test_workers_live_call_records_and_replays(monkeypatch):
    calls = patch_post(monkeypatch, [workers_ok(CLAIMS)])
    r = llm.chat(MSGS, model="@cf/openai/gpt-oss-120b", json_mode=True, max_tokens=900,
                 extra={"reasoning_effort": "low"})
    assert not r["cached"] and r["text"] == CLAIMS and r["provider"] == "workers_ai"
    assert r["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert r["meta"]["neurons"] == 3.5
    body = calls[0]["json"]
    assert body["model"] == "@cf/openai/gpt-oss-120b" and body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 900 and body["response_format"] == {"type": "json_object"}
    assert calls[0]["url"].endswith("/accounts/acct/ai/v1/chat/completions")
    # second identical call is served from the cache without any HTTP
    r2 = llm.chat(MSGS, model="@cf/openai/gpt-oss-120b", json_mode=True, max_tokens=900,
                  extra={"reasoning_effort": "low"})
    assert r2["cached"] and r2["text"] == CLAIMS and len(calls) == 1
    # a different `extra` is a different cache entry
    assert llm.cache_key("m", MSGS, 0.0, 9, True, "workers_ai", {"a": 1}) != llm.cache_key("m", MSGS, 0.0, 9, True)


def test_null_content_does_not_crash(monkeypatch):
    patch_post(monkeypatch, [workers_ok(None, reasoning="no answer here")])
    r = llm.chat(MSGS, model="@cf/google/gemma-4-26b-a4b-it")
    assert r["text"] == "" and r["usage"]["completion_tokens"] == 7


def test_cohere_live_call(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(200, {
        "message": {"content": [{"type": "text", "text": CLAIMS}]},
        "usage": {"tokens": {"input_tokens": 50, "output_tokens": 9}}})])
    r = llm.chat(MSGS, provider="cohere", json_mode=True)
    assert r["provider"] == "cohere" and r["model"] == "command-a-03-2025" and r["text"] == CLAIMS
    assert r["usage"] == {"prompt_tokens": 50, "completion_tokens": 9}
    c = calls[0]
    assert c["url"] == "https://api.cohere.com/v2/chat" and c["headers"]["Authorization"] == "Bearer ck"
    assert c["json"]["messages"] == MSGS and c["json"]["response_format"] == {"type": "json_object"}
    # cache key includes the provider: the same model string under workers_ai is a different entry
    assert llm.cache_key("command-a-03-2025", MSGS, 0.0, 600, True, "cohere") != \
        llm.cache_key("command-a-03-2025", MSGS, 0.0, 600, True, "workers_ai")
    assert llm.chat(MSGS, provider="cohere", json_mode=True)["cached"]


def test_legacy_cache_key_unchanged():
    """Keys recorded before provider support must still match the workers_ai default."""
    import hashlib

    blob = json.dumps({"model": "m", "messages": MSGS, "temperature": 0.0, "max_tokens": 600, "json_mode": True},
                      sort_keys=True, ensure_ascii=False)
    assert llm.cache_key("m", MSGS, 0.0, 600, True) == hashlib.sha256(blob.encode()).hexdigest()


# ---- retry / failure ----------------------------------------------------------------------------------------------
def test_429_then_success_is_retried(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(429, text="rate limit: 20 requests per minute"),
                                     FakeResp(503, text="busy"), workers_ok(CLAIMS)])
    r = llm.chat(MSGS)
    assert r["text"] == CLAIMS and len(calls) == 3
    assert [x["status"] for x in llm.LEDGER] == [429, 503, 200]


def test_retry_is_bounded(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(500, text="oops")] * 10)
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MSGS)
    assert len(calls) == llm.MAX_ATTEMPTS


def test_quota_exhausted_stops_without_retry(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(429, text="you have used up your daily free allocation of 10,000 neurons"),
                                     workers_ok(CLAIMS)])
    with pytest.raises(llm.LLMQuotaExceeded):
        llm.chat(MSGS)
    assert len(calls) == 1
    # Cohere trial monthly cap is a quota error too (and it is a subclass of LLMUnavailable)
    patch_post(monkeypatch, [FakeResp(429, text="You are using a Trial key, which is limited to 1000 API calls / month.")])
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MSGS, provider="cohere")


def test_non_retryable_client_error(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(401, text="bad token")])
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MSGS)
    assert len(calls) == 1


def test_json_mode_400_falls_back_without_response_format(monkeypatch):
    calls = patch_post(monkeypatch, [FakeResp(400, text="response_format unsupported"), workers_ok(CLAIMS)])
    assert llm.chat(MSGS, json_mode=True)["text"] == CLAIMS
    assert "response_format" not in calls[1]["json"]


def test_replay_mode_never_calls_http(monkeypatch):
    monkeypatch.setenv("ZELOS_LLM_MODE", "replay")
    calls = patch_post(monkeypatch, [])
    with pytest.raises(llm.LLMUnavailable):
        llm.chat(MSGS, provider="cohere")
    assert calls == []


def test_call_budget_caps_http_attempts(monkeypatch):
    patch_post(monkeypatch, [FakeResp(200, {"message": {"content": [{"type": "text", "text": "a"}]}, "usage": {}})] * 5)
    llm.set_call_budget("cohere", 2)
    llm.chat([{"role": "user", "content": "1"}], provider="cohere")
    llm.chat([{"role": "user", "content": "2"}], provider="cohere")
    with pytest.raises(llm.LLMBudgetExceeded):
        llm.chat([{"role": "user", "content": "3"}], provider="cohere")
    assert llm.calls_made("cohere") == 2


def test_use_overrides_provider_model_and_params(monkeypatch):
    calls = patch_post(monkeypatch, [workers_ok(CLAIMS)])
    with llm.use("workers_ai", "@cf/qwen/qwen3.8-27b", max_tokens=1500, extra={"reasoning_effort": "low"}):
        assert llm.default_model() == "@cf/qwen/qwen3.8-27b"
        r = llm.chat(MSGS, max_tokens=700)
    assert calls[0]["json"]["max_tokens"] == 1500 and calls[0]["json"]["reasoning_effort"] == "low"
    assert r["model"] == "@cf/qwen/qwen3.8-27b"
    assert llm.default_model() == "@cf/meta/llama-3.1-8b-instruct-fast"  # restored
