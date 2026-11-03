"""One-way news scrubbing with DeepSeek V4 Flash (temperature 0), regex leak check, and entailment check.

Scrubbed text stores companies as [[KEY]] tokens (KEY = real ticker or synthetic S01..S10). Tokens are mapped to
per-trajectory blind labels only at render time, so one scrub serves every trajectory and real names never
reach a prompt.
"""
from __future__ import annotations

import json

from dlo import config
from dlo.anon import aliases
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError
from dlo.store import RunStore, now_utc

SCRUB_SCHEMA = {"name": "scrub", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["text", "drop"],
    "properties": {"text": {"type": "string"}, "drop": {"type": "boolean"}}}}

ENTAIL_SCHEMA = {"name": "entail", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["supported", "unsupported_sentences"],
    "properties": {"supported": {"type": "boolean"},
                   "unsupported_sentences": {"type": "array", "items": {"type": "string"}}}}}

SCRUB_SYSTEM = """You anonymize financial news for a blinded research study. Rewrite the item into 1 to 3 neutral, wire-style sentences.

Hard rules:
- Use ONLY facts stated in the item. Never add context, background, outcomes or anything you know from elsewhere.
- Write the subject company exactly as its token (given below). Write any other listed company as its token. Every other company becomes a generic phrase such as "a competitor", "a customer", "a supplier" or "another company".
- Products, brands, drugs, executives, founders, investors, subsidiaries, partners, cities, countries tied to a company, and named events become generic phrases ("a flagship product", "the chief executive", "a major customer", "a key overseas market").
- No absolute money amounts, share prices, share counts, market values, index levels or unit counts. Turn them into relative terms ("revenue up 18% year over year, above consensus", "a large buyback", "shares fell about 3%"). Percentages are fine.
- No dates of any kind: no years, months, weekdays, quarters (Q1, fourth quarter), fiscal years, holidays or seasons tied to a date. Use "the latest quarter", "recently", "next year".
- No ticker symbols, exchange names or index names; call the overall market "the broad market".
- Neutral tone: no hype, no opinions, no questions, no clickbait.
- Set drop=true (text may be empty) when the item has no substantive information about the subject, e.g. a list of many tickers, an advertisement, or a pure price-move notice with no cause."""

STRICT_ADDENDUM = """
A previous rewrite leaked identifying terms: {hits}. Be stricter: remove every proper noun except the tokens, every number that is not a percentage, and every date reference."""

ENTAIL_SYSTEM = """You check an anonymized rewrite against its source news item. Tokens like [[XYZ]] and generic phrases ("a competitor", "the chief executive") stand for entities in the source, and converting amounts into relative terms is allowed. A sentence is unsupported if it states a fact, outcome, cause or number that the source does not state or directly imply. Report supported=false if any sentence is unsupported, and list those sentences."""


def _universe_tokens(subject: str | None) -> str:
    rows = []
    for sec in config.universe()["sectors"].values():
        for c in sec["real"]:
            tag = " (SUBJECT)" if c["ticker"] == subject else ""
            rows.append(f"- {c['name']} -> [[{c['ticker']}]]{tag}")
    return "\n".join(rows)


def _user_msg(item: dict, subject: str | None) -> str:
    subj = f"Subject company token: [[{subject}]]" if subject else "Market-wide item: there is no subject company."
    return (f"{subj}\nListed companies and their tokens:\n{_universe_tokens(subject)}\n\n"
            f"HEADLINE: {item['headline']}\nSUMMARY: {item.get('summary') or '(none)'}")


async def scrub_item(client: LLMClient, store: RunStore, item: dict, subject: str | None, kind: str,
                     sim_day: str) -> dict:
    """Scrub one raw item (idempotent: reuses a stored result). Returns the news_scrubbed row."""
    news_id = f"{kind}:{subject or 'MKT'}:{item['id']}"
    prev = store.query("SELECT * FROM news_scrubbed WHERE news_id=?", (news_id,))
    if prev:
        return dict(prev[0])
    attempts, hits_log, text, status = 0, [], "", "dropped"
    entail_ok = None
    for strict in (False, True):
        attempts += 1
        sys = SCRUB_SYSTEM + (STRICT_ADDENDUM.format(hits=", ".join(hits_log[-1])) if strict else "")
        try:
            res = await client.chat(ChatRequest(
                model_key="deepseek", purpose="scrub_strict" if strict else "scrub", budget="infra",
                sim_day=sim_day, json_schema=SCRUB_SCHEMA, max_tokens=600,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": _user_msg(item, subject)}]))
            out = json.loads(res.content)
        except (LLMError, json.JSONDecodeError) as e:
            hits_log.append([f"error:{type(e).__name__}"])
            continue
        if out.get("drop") or not out.get("text", "").strip():
            status, text = "dropped_uninformative", ""
            break
        text = out["text"].strip()
        hits = aliases.leak_hits(text)
        if not hits:
            status = "ok"
            break
        hits_log.append(hits)
        status, text = "dropped_leak", ""
    if status == "ok":
        entail_ok = await entailment_ok(client, item, text, sim_day)
        if not entail_ok:
            status, text = "dropped_entailment", ""
    row = {"news_id": news_id, "kind": kind, "ticker": subject, "published_at": str(item["created_at"]),
           "sim_day": sim_day, "status": status, "text_tokenized": text, "attempts": attempts,
           "regex_hits": json.dumps(hits_log), "entail_ok": None if entail_ok is None else int(entail_ok),
           "created_at": now_utc()}
    store.put("news_scrubbed", row)
    return row


async def entailment_ok(client: LLMClient, item: dict, text: str, sim_day: str) -> bool:
    try:
        res = await client.chat(ChatRequest(
            model_key="deepseek", purpose="entail", budget="infra", sim_day=sim_day, json_schema=ENTAIL_SCHEMA,
            max_tokens=500,
            messages=[{"role": "system", "content": ENTAIL_SYSTEM},
                      {"role": "user", "content": f"SOURCE HEADLINE: {item['headline']}\nSOURCE SUMMARY: "
                                                  f"{item.get('summary') or '(none)'}\n\nREWRITE: {text}"}]))
        return bool(json.loads(res.content)["supported"])
    except (LLMError, json.JSONDecodeError, KeyError):
        return False


def render(text: str, labels: dict[str, str]) -> str:
    """Map [[KEY]] tokens to a trajectory's blind labels; unknown keys become a generic phrase."""
    return aliases.TOKEN_RE.sub(lambda m: labels.get(m.group(1), "another company"), text)
