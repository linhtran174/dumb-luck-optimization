"""OpenRouter chat client with persistence-first caching, spend caps, retries and per-model concurrency.

Contract: every HTTP response (success or failure) is written to the LLM ledger before `chat()` returns.
A successful response is reused for any later request with the same request hash, so a crash, resume or
rerun never pays twice. `ChatResult.cost` always reports the original `usage.cost`, and `cached=True`
marks a ledger hit (the money was spent once, when first recorded).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from dlo import config, logs
from dlo.store import LLMLedger, canonical_json, sha256_text

log = logs.get("openrouter")
URL = "https://openrouter.ai/api/v1/chat/completions"


class BudgetExceeded(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class ChatRequest:
    model_key: str                     # key in config/models.yaml
    messages: list[dict[str, str]]
    purpose: str
    budget: str
    charged: bool = False
    trajectory_id: str | None = None
    sim_day: str | None = None
    attempt: int = 0
    json_schema: dict | None = None    # strict structured output schema (name, schema)
    cache_salt: str = ""               # distinguishes deliberately repeated identical prompts
    max_tokens: int | None = None


@dataclass
class ChatResult:
    content: str
    cost: float
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    call_id: str
    cached: bool
    provider: str | None
    response_model: str | None
    reasoning: str | None = None
    raw: dict = field(default_factory=dict)


class LLMClient(Protocol):
    async def chat(self, req: ChatRequest) -> ChatResult: ...


def build_body(req: ChatRequest) -> dict[str, Any]:
    m = config.models()[req.model_key]
    body: dict[str, Any] = {
        "model": m["slug"],
        "messages": req.messages,
        "max_tokens": req.max_tokens or m["max_tokens"],
        "usage": {"include": True},
        "provider": m["provider"],
    }
    if m.get("reasoning_effort"):
        body["reasoning"] = {"effort": m["reasoning_effort"]}
    elif m.get("reasoning_enabled") is False:
        body["reasoning"] = {"enabled": False}
    if m.get("temperature") is not None:
        body["temperature"] = m["temperature"]
    if req.json_schema is not None and m.get("structured_outputs"):
        body["response_format"] = {"type": "json_schema", "json_schema": req.json_schema}
    elif req.json_schema is not None:
        body["response_format"] = {"type": "json_object"}
    return body


def request_hash(req: ChatRequest, body: dict[str, Any]) -> str:
    return sha256_text(canonical_json({"body": body, "salt": req.cache_salt}))


class OpenRouterClient:
    def __init__(self, ledger: LLMLedger, api_key: str | None = None, timeout_s: float = 600.0,
                 max_http_attempts: int = 6):
        self.ledger = ledger
        self.api_key = api_key or config.secret("OPENROUTER_API_KEY")
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=30.0))
        self.max_http_attempts = max_http_attempts
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._reserved: dict[str, float] = {}
        self._hash_locks: dict[str, asyncio.Lock] = {}

    async def aclose(self) -> None:
        await self.http.aclose()

    def _sem(self, model_key: str) -> asyncio.Semaphore:
        if model_key not in self._sems:
            self._sems[model_key] = asyncio.Semaphore(config.models()[model_key].get("max_concurrency", 4))
        return self._sems[model_key]

    def _estimate(self, req: ChatRequest, body: dict) -> float:
        p = config.models()[req.model_key]["list_price_per_m"]
        in_tok = len(canonical_json(body["messages"])) / 3.5
        out_tok = min(body["max_tokens"], 6000)
        return (in_tok * p["input"] + out_tok * p["output"]) / 1e6

    def _check_budget(self, req: ChatRequest, est: float) -> None:
        cap = config.costs()["spend_caps_usd"][req.budget]
        spent = self.ledger.spent(req.budget)
        reserved = self._reserved.get(req.budget, 0.0)
        if spent + reserved + est > cap:
            raise BudgetExceeded(
                f"budget '{req.budget}' cap ${cap:.2f}: spent ${spent:.4f} + in-flight ${reserved:.4f} + est ${est:.4f}"
            )

    async def chat(self, req: ChatRequest) -> ChatResult:
        body = build_body(req)
        h = request_hash(req, body)
        lock = self._hash_locks.setdefault(h, asyncio.Lock())
        async with lock:  # identical concurrent requests wait for one paid call
            hit = self.ledger.cached(h)
            if hit is not None:
                return _result_from_row(hit, cached=True)
            est = self._estimate(req, body)
            self._check_budget(req, est)
            self._reserved[req.budget] = self._reserved.get(req.budget, 0.0) + est
            try:
                async with self._sem(req.model_key):
                    return await self._call_with_retries(req, body, h)
            finally:
                self._reserved[req.budget] -= est

    async def _call_with_retries(self, req: ChatRequest, body: dict, h: str) -> ChatResult:
        delay = 2.0
        last_err = ""
        for http_attempt in range(1, self.max_http_attempts + 1):
            t0 = time.monotonic()
            status, data, err = None, None, None
            try:
                r = await self.http.post(
                    URL, json=body,
                    headers={"Authorization": f"Bearer {self.api_key}", "X-Title": "dlo-bench"},
                )
                status = r.status_code
                try:
                    data = r.json()
                except json.JSONDecodeError:
                    data = {"_raw_text": r.text[:20000]}
            except httpx.HTTPError as e:
                err = f"{type(e).__name__}: {e}"
            latency = time.monotonic() - t0

            ok, content, problem = _inspect(status, data, err)
            usage = (data or {}).get("usage") or {}
            row = {
                "request_hash": h, "purpose": req.purpose, "budget": req.budget, "charged": int(req.charged),
                "trajectory_id": req.trajectory_id, "sim_day": req.sim_day, "attempt": req.attempt,
                "model_key": req.model_key, "model_slug": body["model"],
                "response_model": (data or {}).get("model"), "provider": (data or {}).get("provider"),
                "request_json": canonical_json(body), "response_json": canonical_json(data) if data else None,
                "http_status": status, "ok": int(ok), "error": None if ok else (problem or err),
                "latency_s": round(latency, 3),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "cost": float(usage.get("cost") or 0.0),
            }
            call_id = self.ledger.record(row)  # persisted before anything else happens
            if ok:
                logs.event(log, "llm_call", purpose=req.purpose, model=req.model_key, traj=req.trajectory_id,
                           day=req.sim_day, cost=row["cost"], latency=row["latency_s"], provider=row["provider"])
                return _result_from_row({**row, "call_id": call_id}, cached=False)
            last_err = row["error"] or "unknown"
            retryable = status is None or status in (408, 409, 425, 429) or (status and status >= 500) or \
                ((problem or "").startswith("empty") and "finish_reason=length" not in (problem or "")) or \
                (problem or "").startswith("upstream")
            logs.event(log, "llm_call_failed", purpose=req.purpose, model=req.model_key, status=status,
                       error=last_err[:300], attempt=http_attempt, retryable=bool(retryable))
            if not retryable or http_attempt == self.max_http_attempts:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
        raise LLMError(f"{req.purpose}/{req.model_key}: {last_err}")


def _inspect(status: int | None, data: dict | None, err: str | None) -> tuple[bool, str | None, str | None]:
    if err:
        return False, None, err
    if status != 200 or not isinstance(data, dict):
        msg = (data or {}).get("error") if isinstance(data, dict) else None
        return False, None, f"http {status}: {canonical_json(msg)[:500] if msg else ''}"
    if data.get("error"):
        return False, None, f"upstream error: {canonical_json(data['error'])[:500]}"
    try:
        content = data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return False, None, "empty: no choices"
    if not content:
        fr = (data.get("choices") or [{}])[0].get("finish_reason")
        return False, None, f"empty content (finish_reason={fr})"
    return True, content, None


def _result_from_row(row: Any, cached: bool) -> ChatResult:
    row = dict(row)
    data = json.loads(row["response_json"]) if row.get("response_json") else {}
    msg = data["choices"][0]["message"]
    return ChatResult(
        content=msg.get("content") or "",
        cost=float(row.get("cost") or 0.0),
        prompt_tokens=int(row.get("prompt_tokens") or 0),
        completion_tokens=int(row.get("completion_tokens") or 0),
        reasoning_tokens=int(row.get("reasoning_tokens") or 0),
        call_id=row["call_id"],
        cached=cached,
        provider=row.get("provider"),
        response_model=row.get("response_model"),
        reasoning=msg.get("reasoning"),
        raw=data,
    )


class MockLLMClient:
    """Test double: `responder(req) -> str` produces content; usage and cost are synthetic."""

    def __init__(self, responder, cost: float = 0.001):
        self.responder = responder
        self.cost = cost
        self.calls: list[ChatRequest] = []

    async def chat(self, req: ChatRequest) -> ChatResult:
        self.calls.append(req)
        content = self.responder(req)
        return ChatResult(content=content, cost=self.cost, prompt_tokens=100, completion_tokens=50,
                          reasoning_tokens=0, call_id=f"mock-{len(self.calls)}", cached=False,
                          provider="mock", response_model=req.model_key)


async def chat_json(client: LLMClient, req: ChatRequest, tries: int = 3) -> tuple[dict, ChatResult]:
    """Infrastructure helper: call and parse JSON; on invalid or truncated JSON, re-ask with a corrective note.

    Each retry changes the request body, so it is a new ledger row rather than a cache hit of the bad answer.
    """
    last: Exception | None = None
    base = list(req.messages)
    for k in range(tries):
        if k:
            req.messages = base + [{"role": "user", "content":
                                    "Your previous output was invalid or truncated JSON. Reply again with compact, "
                                    "valid JSON only, shorter than before."}]
            req.attempt = k
        res = await client.chat(req)
        try:
            return json.loads(res.content), res
        except json.JSONDecodeError as e:
            last = e
    raise LLMError(f"{req.purpose}: no valid JSON after {tries} tries: {last}")
