"""Build alias dictionaries with DeepSeek V4 Flash from the raw news of both windows, merged with hand seeds."""
from __future__ import annotations

import asyncio
import json
import re

import numpy as np
import polars as pl
import yaml

from dlo import config
from dlo.anon import aliases
from dlo.anon.alias_seed import SEED
from dlo.clients.alpaca import NEWS_ALIASES
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, chat_json
from dlo.data.snapshot import Snapshot

CATS = ["legal_and_common_names", "products_and_brands", "executives", "subsidiaries", "major_partners",
        "facilities_and_places"]
SCHEMA = {"name": "aliases", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": CATS,
    "properties": {c: {"type": "array", "items": {"type": "string"}} for c in CATS}}}

SYSTEM = """You build an alias dictionary used to detect when anonymized text still identifies a company.
From the company name and the sample news lines, list every term that would let a reader identify this specific company: legal and common names, abbreviations, products, brands, drugs, platforms, executives (full names and distinctive surnames), subsidiaries, major partners or acquisition targets, distinctive facilities and places tied to it. Only include terms that appear in the samples or that you are confident identify this company. Exclude generic words (e.g. "cloud", "bank", "oil", "stores", "chips"). Return at most 80 terms in total, each listed once, most identifying first."""

COMMON = set("""the and for with from that this market stock stocks shares bank oil gas energy drug drugs data cloud
ai chips chip store stores retail ceo cfo inc corp company group holdings services fund trust health care
growth revenue earnings sales first one new global united american national international""".split())


def _samples(snap: Snapshot, ticker: str, n: int = 220, seed: int = 0) -> list[str]:
    al = NEWS_ALIASES.get(ticker, {ticker})
    df = snap.news.filter(pl.col("symbols").list.eval(pl.element().is_in(list(al))).list.any())
    rows = df.select("headline", "summary").to_dicts()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))[:n]
    return [f"- {rows[i]['headline']} | {(rows[i]['summary'] or '')[:200]}" for i in sorted(idx)]


def _clean(terms: list[str]) -> list[str]:
    out = set()
    for t in terms:
        t = t.strip().strip(".,;:\"'")
        if len(t) < 3 and not t.isupper():
            continue
        if t.lower() in COMMON or re.fullmatch(r"[\d\W]+", t):
            continue
        if len(t.split()) > 5:
            continue
        out.add(t)
    return sorted(out)


async def build_one(client: LLMClient, snap: Snapshot, company: dict) -> dict:
    t = company["ticker"]
    msg = f"Company: {company['name']} (ticker {t})\nSample news lines:\n" + "\n".join(_samples(snap, t))
    built, _ = await chat_json(client, ChatRequest(
        model_key="deepseek", purpose="alias", budget="infra", json_schema=SCHEMA, max_tokens=5000,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]))
    terms = {c: _clean(built.get(c, [])) for c in CATS}
    terms["hand_seed"] = sorted(set(SEED.get(t, [])))
    return {"ticker": t, "name": company["name"], "status": "built_unreviewed", "terms": terms}


async def build_all(client: LLMClient, snap: Snapshot) -> list[dict]:
    companies = [c for s in config.universe()["sectors"].values() for c in s["real"]]
    docs = await asyncio.gather(*(build_one(client, snap, c) for c in companies))
    aliases.ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    for d in docs:
        aliases.alias_file(d["ticker"]).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    aliases.reset_cache()
    return docs
