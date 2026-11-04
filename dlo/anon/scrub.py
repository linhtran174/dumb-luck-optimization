"""One-way news scrubbing with DeepSeek V4 Flash (temperature 0), regex leak check, and entailment check.

Scrubbed text stores companies as [[KEY]] tokens (KEY = real ticker or synthetic S01..S10). Tokens are mapped to
per-trajectory blind labels only at render time, so one scrub serves every trajectory and real names never
reach a prompt.
"""
from __future__ import annotations

import json
import re

from dlo import config
from dlo.anon import aliases
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError, chat_json
from dlo.store import RunStore, now_utc

SCRUB_SCHEMA = {"name": "scrub", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["text", "drop"],
    "properties": {"text": {"type": "string"}, "drop": {"type": "boolean"}}}}

ENTAIL_SCHEMA = {"name": "entail", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["supported", "unsupported_sentences", "named_entities"],
    "properties": {"supported": {"type": "boolean"},
                   "unsupported_sentences": {"type": "array", "items": {"type": "string"}},
                   "named_entities": {"type": "array", "items": {"type": "string"}}}}}

SCRUB_VERSION = "v2"   # v1 (trial-v1) kept sector and business-model detail and failed G3; rows stay for audit

SCRUB_SYSTEM = """You anonymize financial news for a blinded research study. A reader of your rewrite must not be able to tell which company the item is about, or even which industry it is in. Rewrite the item into 1 to 3 neutral, wire-style sentences that keep only what an investor can act on: what kind of event it is, whether it is good or bad for the subject company, how large it is, and whether it beat or missed expectations.

Hard rules:
- Use ONLY facts stated in the item. Never add context, background, outcomes or anything you know from elsewhere.
- Write the subject company exactly as its token (given below). Write any other listed company as its token. Every other company becomes a generic phrase: "another company", "a competitor", "a customer", "a supplier", "a partner".
- INDUSTRY-NEUTRAL WORDING. Remove every word that reveals the industry, the business model or what the company sells. Banned, for example: drugs, medicines, diseases, patients, trials, doctors, chips, semiconductors, AI, GPUs, data centers, devices, phones, apps, software, cloud, oil, gas, crude, barrels, wells, fields, offshore, refining, LNG, stores, shoppers, groceries, membership, e-commerce, loans, deposits, credit cards, insurance, underwriting, trading desks, payments. Also banned: industry metrics (same-store or comparable sales, net interest income, production volumes, refining margins, renewal rates), industry-specific regulators and approvals, and industry-specific events (a drug approval, a store opening, an oilfield stake).
- Use generic business vocabulary instead: "a product", "a product line", "a new offering", "a business segment", "a project", "an asset", "a facility", "sales", "volumes", "margins", "a key operating metric", "a regulator", "a regulatory approval", "a legal dispute", "a government inquiry", "a partnership", "an acquisition", "a stake in another company's assets".
- People: "the chief executive", "the finance chief" or "a senior executive"; never a function that hints at the industry ("head of AI", "design chief", "chief investment officer").
- Places: no countries, regions, states or cities tied to the company. Use "a key market", "an overseas market" or "the domestic market".
- Numbers: keep percentages only for changes, surprises, margins and price moves, rounded to whole numbers. No stake sizes, absolute money amounts, share prices, share counts, market values, unit counts or index levels; use relative terms ("a large buyback", "revenue up 18% year over year, above consensus").
- No dates of any kind: no years, months, weekdays, quarters, fiscal years, holidays or seasons. Use "the latest quarter", "recently", "next year".
- Do not describe the subject's own share-price moves, records or levels ("shares rose 3%", "hit an all-time high"): the reader already has the price data. Keep the event behind the move, if any.
- No ticker symbols, exchange names or index names; call the overall market "the broad market".
- Neutral tone: no hype, no opinions, no questions, no clickbait.
- Set drop=true (text may be empty) when the item has no substantive company-specific information: a list of many tickers, an advertisement, options-activity or valuation commentary, a "what a past investment would be worth" piece, or a pure price-move notice with no cause."""

MARKET_SYSTEM = """You anonymize market-wide financial news for a blinded research study. Rewrite the item into 1 to 3 neutral, wire-style sentences about the broad economy and markets: central banks, interest rates, inflation, jobs, growth, trade policy, currencies, commodity prices and broad sector moves.

Hard rules:
- Use ONLY facts stated in the item. Never add context or anything you know from elsewhere.
- Never name or token any company: every specific company becomes "a large company". Products, people (other than central bank and government officials by role, e.g. "the central bank chair") and brands become generic phrases.
- No absolute index levels, prices or money amounts; percentages are fine. No dates of any kind (years, months, weekdays, quarters, holidays). No ticker symbols, exchange names or index names; call the overall market "the broad market".
- Neutral tone: no hype, no opinions, no questions.
- Set drop=true (text may be empty) when the item has no substantive macro or market information."""

STRICT_ADDENDUM = """
A previous rewrite still gave the company or its industry away: {hits}. Be stricter: remove every proper noun except the tokens, every industry-specific word or event (describe it only as a generic business event), every number that is not a percentage, and every date reference."""

ENTAIL_SYSTEM = """You check an anonymized rewrite against its source news item. Tokens like [[XYZ]] and generic phrases ("a competitor", "a product", "a regulatory approval") stand for entities and specifics in the source; replacing specifics with generic terms and converting amounts into relative terms is allowed. A sentence is unsupported if it states a fact, outcome, cause or number that the source does not state or directly imply. Report supported=false if any sentence is unsupported, and list those sentences.
Also list in named_entities every proper name left in the REWRITE (outside [[...]] tokens): people, companies, banks, brokers, analyst firms, products, brands, drugs, platforms, cities, sites. Government bodies and regulators (the Federal Reserve, a court, the FDA) and countries are allowed and must not be listed."""

INDUSTRIES = {"information_technology": "technology", "financials": "financials", "health_care": "health care",
              "energy": "energy", "consumer_staples": "consumer staples and retail"}

_IND_KEYS = {"information_technology": "technology", "financials": "financials", "health_care": "health_care",
             "energy": "energy", "consumer_staples": "consumer_staples_retail"}
ATTACK_SCHEMA = {"name": "attack", "strict": True, "schema": {
    "type": "object", "additionalProperties": False,
    "required": ["industry_probability", "company_guess", "company_probability", "cue"],
    "properties": {
        "industry_probability": {"type": "object", "additionalProperties": False,
                                 "required": list(_IND_KEYS.values()),
                                 "properties": {k: {"type": "number"} for k in _IND_KEYS.values()}},
        "company_guess": {"type": "string"}, "company_probability": {"type": "number"},
        "cue": {"type": "string"}}}}
ATTACK_INDUSTRY_MAX = 0.40   # flag when the true industry gets at least twice the 0.20 chance level
ATTACK_COMPANY_MAX = 0.25

ATTACK_SYSTEM = """You play the adversary in a blinded study. The subject company is written as [[SUBJECT]]; it is a large US listed company. From the text alone, estimate which industry it is in and which company it is.
- industry_probability: your probability for each of the five industries (technology, financials, health_care, energy, consumer_staples_retail); they must sum to 1. If nothing in the text points anywhere, give 0.2 each.
- company_guess: your single best guess of the real company, and company_probability: your probability that the guess is right.
- cue: quote the words that drove your estimate, or an empty string."""


def news_id(kind: str, subject: str | None, item_id) -> str:
    return f"{SCRUB_VERSION}:{kind}:{subject or 'MKT'}:{item_id}"


def sector_of(key: str) -> str | None:
    """Sector of a real ticker or a pre-generated synthetic key (S01..S10)."""
    for sec, d in config.universe()["sectors"].items():
        if key in {c["ticker"] for c in d["real"]}:
            return sec
    from dlo.synth import pregenerate
    try:
        return {s["key"]: s["sector"] for s in pregenerate.load()[1]}.get(key)
    except FileNotFoundError:
        return None


def _universe_tokens(subject: str | None) -> str:
    rows = []
    for sec in config.universe()["sectors"].values():
        for c in sec["real"]:
            tag = " (SUBJECT)" if c["ticker"] == subject else ""
            rows.append(f"- {c['name']} -> [[{c['ticker']}]]{tag}")
    if subject and subject not in config.real_tickers():
        rows.append(f"- the subject company -> [[{subject}]] (SUBJECT)")
    return "\n".join(rows)


def _user_msg(item: dict, subject: str | None) -> str:
    src = f"HEADLINE: {item['headline']}\nSUMMARY: {item.get('summary') or '(none)'}"
    if not subject:
        return src
    return (f"Subject company token: [[{subject}]]\nListed companies and their tokens:\n"
            f"{_universe_tokens(subject)}\n\n{src}")


async def scrub_item(client: LLMClient, store: RunStore, item: dict, subject: str | None, kind: str,
                     sim_day: str) -> dict:
    """Scrub one raw item (idempotent: reuses a stored result). Returns the news_scrubbed row.

    Checks, in order: alias/date/amount regex, an industry/identity attack on the rewrite alone (company items),
    then entailment against the source plus leftover proper names. Any failure triggers one strict re-scrub;
    a second failure drops the item.
    """
    nid = news_id(kind, subject, item["id"])
    prev = store.query("SELECT * FROM news_scrubbed WHERE news_id=?", (nid,))
    if prev:
        return dict(prev[0])
    truth = sector_of(subject) if subject else None
    attempts, hits_log, text, status = 0, [], "", "dropped"
    entail_ok = None
    for strict in (False, True):
        attempts += 1
        base = SCRUB_SYSTEM if subject else MARKET_SYSTEM
        sys = base + (STRICT_ADDENDUM.format(hits=", ".join(hits_log[-1])) if strict and hits_log else "")
        try:
            out, _ = await chat_json(client, ChatRequest(
                model_key="deepseek", purpose="scrub_strict" if strict else "scrub", budget="infra",
                sim_day=sim_day, json_schema=SCRUB_SCHEMA, max_tokens=600,
                messages=[{"role": "system", "content": sys}, {"role": "user", "content": _user_msg(item, subject)}]))
        except LLMError as e:
            hits_log.append([f"error:{type(e).__name__}"])
            status = "dropped_error"
            continue
        if out.get("drop") or not out.get("text", "").strip():
            status, text = "dropped_uninformative", ""
            break
        text = out["text"].strip()
        if not subject and aliases.TOKEN_RE.search(text):
            text = aliases.TOKEN_RE.sub("a large company", text)
        if subject:
            text = re.sub(r"\b[Tt]he subject company\b", f"[[{subject}]]", text)
        hits = aliases.leak_hits(text)
        if subject and f"[[{subject}]]" not in text:
            hits.append(f"the subject must appear as its token [[{subject}]]")
        if not hits and subject:
            hits = await attack(client, text, subject, truth, sim_day)
        if not hits:
            check = await entailment_check(client, item, text, sim_day)
            entail_ok = check["supported"]
            hits = [f"name:{n}" for n in check["named_entities"]]
            if not hits:
                status = "ok" if entail_ok else "dropped_entailment"
                if not entail_ok:
                    text = ""
                break
        hits_log.append(hits)
        status, text = "dropped_leak", ""
    row = {"news_id": nid, "kind": kind, "ticker": subject, "published_at": str(item.get("created_at", "")),
           "sim_day": sim_day, "status": status, "text_tokenized": text, "attempts": attempts,
           "regex_hits": json.dumps(hits_log), "entail_ok": None if entail_ok is None else int(entail_ok),
           "created_at": now_utc()}
    store.put("news_scrubbed", row)
    return row


async def attack(client: LLMClient, text: str, subject: str, truth_sector: str | None, sim_day: str,
                 model_key: str = "deepseek_r") -> list[str]:
    """An adversary (DeepSeek with reasoning) estimates the subject's industry and identity from the rewrite alone.

    A leak is probability >= 0.40 on the true industry (chance 0.20), or >= 0.25 on a named company (for a real
    subject: the right one; for a synthetic subject, any confident name is a false trail that still marks the
    text as identifying). Errors count as a leak, so they never let text through unchecked.
    """
    rendered = aliases.TOKEN_RE.sub(lambda m: "[[SUBJECT]]" if m.group(1) == subject else "another company", text)
    try:
        out, _ = await chat_json(client, ChatRequest(
            model_key=model_key, purpose="attack", budget="infra", sim_day=sim_day, json_schema=ATTACK_SCHEMA,
            max_tokens=6000, messages=[{"role": "system", "content": ATTACK_SYSTEM},
                                       {"role": "user", "content": rendered}]))
        probs = out["industry_probability"]
        tot = sum(max(0.0, float(v)) for v in probs.values()) or 1.0
        p_true = float(probs.get(_IND_KEYS.get(truth_sector, ""), 0.0)) / tot if truth_sector else 0.0
        guess, p_guess = str(out.get("company_guess") or ""), float(out.get("company_probability") or 0.0)
    except (LLMError, KeyError, TypeError, ValueError):
        return ["attack:error"]
    cue = str(out.get("cue", ""))[:80]
    hits = []
    if p_true >= ATTACK_INDUSTRY_MAX:
        hits.append(f"industry {truth_sector} p={p_true:.2f} ({cue})")
    if p_guess >= ATTACK_COMPANY_MAX and _names_company(guess, subject):
        hits.append(f"company {guess[:40]} p={p_guess:.2f} ({cue})")
    return hits


def _names_company(guess: str, subject: str) -> bool:
    g = guess.lower()
    if not g or any(w in g for w in ("cannot", "unknown", "none", "unclear", "n/a", "no idea")):
        return False
    if subject not in config.real_tickers():
        return True
    from dlo.anon.aliases import load_aliases
    name = next(c["name"] for s in config.universe()["sectors"].values() for c in s["real"] if c["ticker"] == subject)
    first = name.split()[0].lower().strip(",")
    return first in g or subject.lower() == g.strip() or any(
        t.lower() == g.strip() for t in load_aliases().get(subject, [])[:20])


async def entailment_check(client: LLMClient, item: dict, text: str, sim_day: str) -> dict:
    """Entailment plus leftover-name check. Any failure counts as unsupported (the item is then dropped)."""
    try:
        out, _ = await chat_json(client, ChatRequest(
            model_key="deepseek", purpose="entail", budget="infra", sim_day=sim_day, json_schema=ENTAIL_SCHEMA,
            max_tokens=500,
            messages=[{"role": "system", "content": ENTAIL_SYSTEM},
                      {"role": "user", "content": f"SOURCE HEADLINE: {item['headline']}\nSOURCE SUMMARY: "
                                                  f"{item.get('summary') or '(none)'}\n\nREWRITE: {text}"}]))
        return {"supported": bool(out["supported"]),
                "named_entities": [n for n in out.get("named_entities", []) if n.strip()]}
    except (LLMError, KeyError):
        return {"supported": False, "named_entities": []}


def render(text: str, labels: dict[str, str]) -> str:
    """Map [[KEY]] tokens to a trajectory's blind labels; unknown keys become a generic phrase."""
    return aliases.TOKEN_RE.sub(lambda m: labels.get(m.group(1), "another company"), text)
