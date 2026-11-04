"""Synthetic news text, written by DeepSeek V4 Flash and then passed through the SAME scrubber and leak checks
as real news (v2), so real and synthetic items share one style and one industry-neutral vocabulary.

Specs (count, event flag, event type, sentiment z) are pre-generated; text is produced once per asset-day, stored,
and reused by every trajectory. Sentiment is conveyed in words only, so the planted news-lead signal reaches the
agent the same way real news would. An item the scrubber drops is regenerated (up to 3 rounds).
"""
from __future__ import annotations

import hashlib

from dlo.anon.scrub import SCRUB_VERSION, scrub_item
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError, chat_json
from dlo.store import RunStore, now_utc

# Filler items mirror randomly drawn real items (scrubbed, industry-neutral) from the dev window, which ends before
# the test window starts: same mix of event kinds and directions as real news, without ever showing test-window
# news early. A template's direction is independent of the synthetic asset's future returns, so it plants nothing.
TEMPLATE_WINDOW = "dev"


GEN_SCHEMA = {"name": "synth_news", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["items"],
    "properties": {"items": {"type": "array", "items": {"type": "string"}}}}}

GEN_SYSTEM = """You write short news items about a large US listed company for a blinded research study, in the style of a terse financial news wire. Each item is 1 to 3 sentences. The company is always written as the token [[SELF]]. Some items mirror a real template: write a NEW item of the same kind of event and the same direction (positive, negative or neutral), with different specifics and wording. Follow these rules exactly:
- Industry-neutral: never reveal what the company sells or which industry it is in. Use generic business terms only ("a product line", "a business segment", "a project", "a facility", "sales", "margins", "a regulatory approval").
- Never name real companies, products, people, places, tickers, exchanges or indexes. Other firms are "another company", "a competitor", "a customer", "a supplier", "a partner".
- No dates of any kind (no years, months, weekdays, quarters, holidays). Use "the latest quarter", "recently".
- No absolute money amounts, prices or counts; use whole-number percentages and relative terms.
- Never describe the company's share-price moves, records or levels.
- Factual, no hype, no questions, no advice."""

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


def build_prompt(n_event: int, event_type: str, z: float | None, templates: list[str], round_: int) -> str:
    lines = [f"Write exactly {n_event + len(templates)} item(s)."]
    if n_event:
        lines.append(f"Item 1: a {event_type.replace('_', ' ')} event for [[SELF]] whose news is "
                     f"{sentiment_word(z or 0.0)} for the company; state concrete relative facts.")
    for i, t in enumerate(templates, start=n_event + 1):
        lines.append(f"Item {i}: mirror this template (same kind of event, same direction, new specifics): {t}")
    if round_:
        lines.append(f"(Revision {round_}: an earlier draft was rejected for revealing the industry. Keep it generic.)")
    return "\n".join(lines)


_POOL: list[str] | None = None


def template_pool(store: RunStore) -> list[str]:
    """Accepted real company items (current scrub version) from the template window, with every token generic."""
    global _POOL
    if _POOL is None:
        from dlo import config
        from dlo.anon import aliases
        a, b = config.window(TEMPLATE_WINDOW)
        rows = store.query("SELECT ticker, text_tokenized FROM news_scrubbed WHERE news_id LIKE ? AND status='ok' "
                           "AND sim_day BETWEEN ? AND ? ORDER BY news_id",
                           (f"{SCRUB_VERSION}:asset:%", a.isoformat(), b.isoformat()))
        _POOL = [aliases.TOKEN_RE.sub(lambda m, t=r["ticker"]: "[[SELF]]" if m.group(1) == t else "another company",
                                      r["text_tokenized"]) for r in rows]
        if not _POOL:
            raise RuntimeError("no scrubbed real items in the template window; run `dlo anon scrub dev` first")
    return _POOL


async def _draft(client: LLMClient, prompt: str, k: int, day: str, round_: int) -> list[str]:
    msgs = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": prompt}]
    for attempt in range(3):
        try:
            out, _ = await chat_json(client, ChatRequest(
                model_key="deepseek", purpose="synth_news", budget="infra", sim_day=day, json_schema=GEN_SCHEMA,
                max_tokens=900, attempt=round_ * 3 + attempt, messages=list(msgs)))
            texts = [t.strip() for t in out["items"] if t.strip()][:k]
        except (LLMError, KeyError):
            texts = []
        if len(texts) == k:
            return texts
        # Some hosts occasionally return an empty array; re-ask (a new request, so not a cache hit).
        msgs = msgs + [{"role": "user", "content": f"Your answer had {len(texts)} usable items. "
                                                   f"Return exactly {k} non-empty items."}]
    return texts


def _item_id(asset: str, day: str, i: int) -> str:
    return f"{SCRUB_VERSION}:{asset}:{day}:{i}"


def stored_texts(store: RunStore, day: str, asset: str | None = None) -> dict[str, list[str]]:
    """Accepted synthetic texts of the current scrub version for a day, per asset."""
    q = "SELECT asset, text_tokenized FROM synth_news WHERE item_id LIKE ? AND sim_day=? AND status='ok'"
    args: tuple = (f"{SCRUB_VERSION}:%", day)
    if asset:
        q, args = q + " AND asset=?", args + (asset,)
    out: dict[str, list[str]] = {}
    for r in store.query(q + " ORDER BY item_id", args):
        out.setdefault(r["asset"], []).append(r["text_tokenized"])
    return out


async def generate_day(client: LLMClient, store: RunStore, asset: str, sector: str, day: str, n: int,
                       event: bool, event_type: str, z: float | None) -> list[str]:
    """Return tokenized, scrubbed texts for one synthetic asset-day (stored; idempotent)."""
    if n <= 0:
        return []
    if store.query("SELECT 1 FROM synth_news WHERE item_id LIKE ? AND asset=? AND sim_day=? LIMIT 1",
                   (f"{SCRUB_VERSION}:%", asset, day)):
        return stored_texts(store, day, asset).get(asset, [])
    event_text: str | None = None
    fillers: list[str] = []
    pool = template_pool(store)
    topics_all = [_pick(pool, f"{asset}:{day}:{i}") for i in range(n)]
    for round_ in range(3):
        n_event = int(event and event_text is None)
        topics = topics_all[len(fillers) + int(event):n]
        if not n_event and not topics:
            break
        if round_:
            topics = [_pick(pool, f"{asset}:{day}:r{round_}:{i}") for i in range(len(topics))]
        drafts = await _draft(client, build_prompt(n_event, event_type, z, topics, round_),
                              n_event + len(topics), day, round_)
        for j, t in enumerate(drafts):
            row = await scrub_item(client, store, {"id": f"{asset}:{day}:r{round_}:{j}",
                                                   "headline": t.replace("[[SELF]]", f"[[{asset}]]")},
                                   asset, "synth", day)
            if row["status"] != "ok":
                continue
            if n_event and j == 0:
                event_text = row["text_tokenized"]
            else:
                fillers.append(row["text_tokenized"])
    texts = ([event_text] if event_text else []) + fillers[:n - int(event)]
    for i, t in enumerate(texts):
        is_event = event and i == 0 and event_text is not None
        store.put("synth_news", {
            "item_id": _item_id(asset, day, i), "asset": asset, "sim_day": day,
            "kind": "event" if is_event else "filler", "event_type": event_type if is_event else "",
            "sentiment": z if is_event else None, "text_tokenized": t, "status": "ok", "created_at": now_utc()})
    if len(texts) < n or (event and event_text is None):
        store.put("synth_news", {
            "item_id": _item_id(asset, day, 99), "asset": asset, "sim_day": day, "kind": "shortfall",
            "event_type": event_type if event and event_text is None else "", "sentiment": None,
            "text_tokenized": f"{len(texts)} of {n} items accepted", "status": "short", "created_at": now_utc()})
    return texts
