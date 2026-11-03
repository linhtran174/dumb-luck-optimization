"""Alias dictionaries and the regex leak check.

One YAML per real company in config/aliases/. Built once by DeepSeek from raw news in both windows, merged with
a hand-written seed list, reviewed, then frozen. The leak regex also catches tickers, real dates and absolute
dollar figures, which identify assets or the calendar.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from dlo import config

ALIAS_DIR = config.CONFIG_DIR / "aliases"
TOKEN_RE = re.compile(r"\[\[([A-Z0-9.]+)\]\]")

# Calendar / magnitude leaks that apply to every item.
GENERIC_PATTERNS = [
    r"\b(19|20)\d{2}\b",                                   # years
    r"\b(January|February|March|April|June|July|August|September|October|November|December)\b",
    r"\bMay\s+\d", r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
    r"\b(Q[1-4]|[1-4]Q|FY\d{2,4}|H[12]\s?\d{2})\b",
    r"\b(Thanksgiving|Christmas|Black Friday|Cyber Monday|New Year)\b",
    r"\$\s?\d", r"\b\d+(\.\d+)?\s?(billion|trillion|million|bn|mn)\b",
    r"\bS&P\s?500\b|\bNasdaq\b|\bDow Jones\b|\bNYSE\b",
]


def alias_file(ticker: str) -> Path:
    return ALIAS_DIR / f"{ticker.replace('.', '_')}.yaml"


@lru_cache
def load_aliases() -> dict[str, list[str]]:
    out = {}
    for t in config.real_tickers():
        p = alias_file(t)
        if p.exists():
            d = yaml.safe_load(p.read_text())
            terms = set()
            for v in d.get("terms", {}).values():
                terms.update(v or [])
            terms.add(t)
            out[t] = sorted(terms, key=len, reverse=True)
        else:
            out[t] = [t]
    return out


def _term_regex(term: str) -> str:
    esc = re.escape(term)
    return rf"(?<![A-Za-z0-9]){esc}(?![A-Za-z0-9])"


@lru_cache
def compiled() -> list[tuple[str, re.Pattern]]:
    pats: list[tuple[str, re.Pattern]] = []
    for t, terms in load_aliases().items():
        for term in terms:
            # Proper nouns and tickers match case-sensitively ("COST" vs "cost", "Chase" vs "chase");
            # all-lowercase terms (generic drug names such as "tirzepatide") match case-insensitively.
            flags = re.I if term.islower() else 0
            pats.append((f"{t}:{term}", re.compile(_term_regex(term), flags)))
    for p in GENERIC_PATTERNS:
        pats.append((f"generic:{p}", re.compile(p)))
    return pats


def strip_tokens(text: str) -> str:
    return TOKEN_RE.sub(" ", text)


def leak_hits(text: str) -> list[str]:
    body = strip_tokens(text)
    return [name for name, rx in compiled() if rx.search(body)]


def reset_cache() -> None:
    load_aliases.cache_clear()
    compiled.cache_clear()
