"""Synthetic news text: DeepSeek V4 Flash writes items in the same scrubbed wire style as real news.

Specs (count, event flag, event type, sentiment z) are pre-generated; text is produced once per asset-day, stored,
and reused by every trajectory. Sentiment is conveyed in words only, so the planted news-lead signal reaches the
agent the same way real news would.
"""
from __future__ import annotations

import hashlib
import json

from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError
from dlo.store import RunStore, now_utc

SECTOR_VOICE = {
    "information_technology": "a large technology company (chips, devices, software, cloud, AI infrastructure)",
    "financials": "a large financial company (banking, lending, insurance, payments, capital markets)",
    "health_care": "a large health care company (drug pipelines, trials, approvals, devices, pricing policy)",
    "energy": "a large integrated energy company (oil and gas output, refining margins, upstream projects)",
    "consumer_staples": "a large consumer staples retailer or brand (store traffic, membership, pricing, grocery)",
}

FILLER_TOPICS = ["operations update", "industry trend", "investor conference remarks", "minor product update",
                 "hiring or facilities", "sector commentary", "routine filing", "dividend or buyback routine notice"]

GEN_SCHEMA = {"name": "synth_news", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {"type": "string"}}}}}

GEN_SYSTEM = """You write short anonymized financial news items for a blinded research study, in neutral wire style.
Each item is 1 to 3 sentences. The company is always written as the token [[SELF]]. Follow these rules exactly:
- Never name real companies, products, people, places, tickers, exchanges or indexes. Other firms are "a competitor", "a customer", "a supplier".
- No dates of any kind (no years, months, weekdays, quarters, holidays). Use "the latest quarter", "recently".
- No absolute money amounts, prices or counts; use percentages and relative terms.
- Match the style of a terse news wire summary: factual, no hype, no questions, no advice."""

SENT_WORDS = [(-1.5, "strongly negative"), (-0.75, "negative"), (-0.25, "mildly negative"), (0.25, "neutral"),
              (0.75, "mildly positive"), (1.5, "positive"), (float("inf"), "strongly positive")]


def sentiment_word(z: float) -> str:
    for hi, w in SENT_WORDS:
        if z < hi:
            return w
    return "neutral"


def _pick(seq: list[str], key: str) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return seq[h % len(seq)]


def build_prompt(asset: str, sector: str, day_key: str, n: int, event: bool, event_type: str,
                 z: float | None) -> str:
    lines = [f"Company profile: {SECTOR_VOICE[sector]}.", f"Write exactly {n} item(s)."]
    k = 0
    if event:
        lines.append(f"Item 1: a {event_type.replace('_', ' ')} event for [[SELF]] whose news is "
                     f"{sentiment_word(z or 0.0)} for the company; state concrete relative facts.")
        k = 1
    for i in range(k, n):
        topic = _pick(FILLER_TOPICS, f"{asset}:{day_key}:{i}")
        lines.append(f"Item {i + 1}: neutral filler about [[SELF]] ({topic}), in this sector's vocabulary; "
                     f"no clear positive or negative implication.")
    return "\n".join(lines)


async def generate_day(client: LLMClient, store: RunStore, asset: str, sector: str, day: str, n: int,
                       event: bool, event_type: str, z: float | None) -> list[str]:
    """Return tokenized texts for one synthetic asset-day (stored; idempotent)."""
    if n <= 0:
        return []
    rows = store.query("SELECT text_tokenized, status FROM synth_news WHERE asset=? AND sim_day=? ORDER BY item_id",
                       (asset, day))
    if rows:
        return [r["text_tokenized"] for r in rows if r["status"] == "ok"]
    prompt = build_prompt(asset, sector, day, n, event, event_type, z)
    texts: list[str] = []
    status = "ok"
    try:
        res = await client.chat(ChatRequest(
            model_key="deepseek", purpose="synth_news", budget="infra", sim_day=day, json_schema=GEN_SCHEMA,
            max_tokens=900, messages=[{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": prompt}]))
        texts = [t.strip() for t in json.loads(res.content)["items"] if t.strip()][:n]
    except (LLMError, json.JSONDecodeError, KeyError):
        status = "error"
    from dlo.anon import aliases
    out = []
    for i, t in enumerate(texts or [""]):
        t = t.replace("[[SELF]]", f"[[{asset}]]")
        st = status if t else "error"
        if st == "ok" and aliases.leak_hits(t):
            st = "dropped_leak"
        store.put("synth_news", {
            "item_id": f"{asset}:{day}:{i}", "asset": asset, "sim_day": day,
            "kind": "event" if (event and i == 0) else "filler", "event_type": event_type if (event and i == 0) else "",
            "sentiment": z if (event and i == 0) else None, "text_tokenized": t, "status": st, "created_at": now_utc()})
        if st == "ok":
            out.append(t)
    return out
