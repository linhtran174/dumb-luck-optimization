"""Select the raw news items each simulated day shows, before scrubbing.

Per real ticker: items published in (prev cutoff, cutoff], whose symbol list includes the ticker and has at most
3 distinct companies, near-duplicate headlines removed, at most 3 per asset per day (most recent kept).
Market: up to 5 items tagged with index ETFs that name no universe company, macro keywords first.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import polars as pl

from dlo import config
from dlo.clients.alpaca import NEWS_ALIASES, NEWS_SYMBOL
from dlo.data import calendar
from dlo.data.snapshot import Snapshot

MAX_PER_ASSET = 3
MAX_MARKET = 5
MAX_SYMBOLS = 3
MACRO = re.compile(r"\b(fed|fomc|powell|inflation|cpi|ppi|jobs|payrolls|unemployment|gdp|treasury|yields?|"
                   r"rates?|tariffs?|recession|s&p 500|nasdaq|dow|stocks|futures|oil|crude|dollar)\b", re.I)


def _norm_symbols(symbols: list[str]) -> set[str]:
    out = set()
    for s in symbols:
        canon = next((t for t, al in NEWS_ALIASES.items() if s in al), s)
        out.add(canon)
    return out


def _tokens(h: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", h.lower()))


def _dedup(rows: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for r in rows:
        t = _tokens(r["headline"])
        if any(len(t & _tokens(k["headline"])) / max(1, len(t | _tokens(k["headline"]))) >= 0.7 for k in kept):
            continue
        kept.append(r)
    return kept


def day_window(day: date) -> tuple[datetime, datetime]:
    cut = config.windows()["decision_cutoff_et"]
    return calendar.cutoff_utc(calendar.prev_session(day), cut), calendar.cutoff_utc(day, cut)


def select_for_day(snap: Snapshot, day: date) -> dict:
    a, b = day_window(day)
    raw = snap.news_between(a, b).sort("created_at", descending=True).to_dicts()
    real = config.real_tickers()
    market_syms = set(config.universe()["market_news_symbols"])
    per_asset: dict[str, list[dict]] = {}
    for t in real:
        cands = [r for r in raw if t in _norm_symbols(r["symbols"]) and len(_norm_symbols(r["symbols"])) <= MAX_SYMBOLS]
        per_asset[t] = _dedup(cands)[:MAX_PER_ASSET]
    mkt = []
    for r in raw:
        syms = _norm_symbols(r["symbols"])
        if syms & market_syms and not (syms & set(real)) and len(syms) <= MAX_SYMBOLS:
            mkt.append(r)
    mkt = _dedup(mkt)
    mkt.sort(key=lambda r: (0 if MACRO.search(r["headline"]) else 1, -r["created_at"].timestamp()))
    return {"assets": per_asset, "market": mkt[:MAX_MARKET], "window": [a.isoformat(), b.isoformat()]}


def real_daily_counts(snap: Snapshot, days: list[date]) -> pl.DataFrame:
    rows = []
    for d in days:
        sel = select_for_day(snap, d)
        for t, items in sel["assets"].items():
            rows.append({"session": d, "ticker": t, "n": len(items)})
    return pl.DataFrame(rows)


__all__ = ["select_for_day", "real_daily_counts", "day_window", "NEWS_SYMBOL"]
