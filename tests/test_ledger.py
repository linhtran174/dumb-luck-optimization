"""LLM ledger: every response persisted before return; identical requests never paid twice; spend cap enforced."""
import json

import httpx
import pytest

from dlo.clients.openrouter_chat import BudgetExceeded, ChatRequest, OpenRouterClient
from dlo.store import LLMLedger


def _resp(content="{\"a\": 1}", cost=0.5):
    return {"id": "x", "model": "openai/gpt-5.6-luna", "provider": "OpenAI",
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost,
                      "completion_tokens_details": {"reasoning_tokens": 2}}}


def _client(tmp_path, handler):
    led = LLMLedger(tmp_path / "l.sqlite", tmp_path / "l.jsonl")
    c = OpenRouterClient(led, api_key="test")
    c.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c, led


def req(**kw):
    d = dict(model_key="luna", messages=[{"role": "user", "content": "hi"}], purpose="smoke", budget="trial")
    d.update(kw)
    return ChatRequest(**d)


async def test_cache_hit_no_second_http(tmp_path):
    n = {"calls": 0}

    def h(request):
        n["calls"] += 1
        return httpx.Response(200, json=_resp())

    c, led = _client(tmp_path, h)
    r1 = await c.chat(req())
    r2 = await c.chat(req())
    assert n["calls"] == 1 and not r1.cached and r2.cached
    assert r2.cost == r1.cost == 0.5
    assert len(led.query("SELECT * FROM llm_calls")) == 1
    assert len((tmp_path / "l.jsonl").read_text().splitlines()) == 1
    r3 = await c.chat(req(cache_salt="other"))
    assert n["calls"] == 2 and not r3.cached


async def test_failures_persisted_and_retried(tmp_path):
    seq = [httpx.Response(503, json={"error": "busy"}), httpx.Response(200, json=_resp())]

    def h(request):
        return seq.pop(0)

    c, led = _client(tmp_path, h)
    import dlo.clients.openrouter_chat as oc
    orig = oc.asyncio.sleep

    async def fast(_):
        return None
    oc.asyncio.sleep = fast
    try:
        r = await c.chat(req())
    finally:
        oc.asyncio.sleep = orig
    rows = led.query("SELECT ok, http_status FROM llm_calls ORDER BY created_at")
    assert [(x["ok"], x["http_status"]) for x in rows] == [(0, 503), (1, 200)]
    assert json.loads(r.content) == {"a": 1}


async def test_budget_cap(tmp_path, monkeypatch):
    from dlo import config
    caps = dict(config.costs())
    caps["spend_caps_usd"] = dict(caps["spend_caps_usd"], trial=0.505)
    monkeypatch.setattr(config, "costs", lambda: caps)

    def h(request):
        return httpx.Response(200, json=_resp(cost=0.5))

    c, led = _client(tmp_path, h)
    await c.chat(req())
    with pytest.raises(BudgetExceeded):
        await c.chat(req(cache_salt="2"))
