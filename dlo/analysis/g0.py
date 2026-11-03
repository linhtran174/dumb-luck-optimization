"""G0 post-cutoff check: do the evaluated models know test-window events?

20 real events derived mechanically from the frozen snapshot (price moves of the universe and the S&P 500 between
test-window month ends) are mixed with 20 fabricated but plausible ones (the same templates with the direction
flipped on other company-periods). Each statement is asked once per model: did this happen (yes / no / don't
know)? Pass: the model's "yes" rate on real events is within 10 points of its "yes" rate on fabricated events.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date

import numpy as np
import polars as pl

from dlo import config
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError
from dlo.data import calendar

SCHEMA = {"name": "g0", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "don't know"]}}}}

SYSTEM = ("Answer from your own knowledge whether the following statement about a past event is true. "
          "Reply yes, no, or don't know.")


def _month_ends(a: date, b: date) -> list[date]:
    s = calendar.sessions(a, b)
    out = [s[0]]
    for x, y in zip(s, s[1:]):
        if y.month != x.month:
            out.append(x)
    out.append(s[-1])
    return sorted(set(out))


def build_statements(bars: pl.DataFrame, seed: int = 2026) -> list[dict]:
    a, b = config.window("test")
    ends = _month_ends(a, b)
    names = {c["ticker"]: c["name"] for s in config.universe()["sectors"].values() for c in s["real"]}
    names[config.universe()["market"]] = "The S&P 500 index (via the SPY ETF)"
    close = {(r["symbol"], r["session"]): r["close"] for r in bars.select("symbol", "session", "close").iter_rows(named=True)}
    cands = []
    for sym, name in names.items():
        for d0, d1 in zip(ends, ends[1:]):
            if (sym, d0) in close and (sym, d1) in close:
                ret = close[(sym, d1)] / close[(sym, d0)] - 1
                if abs(ret) >= 0.05:
                    cands.append((sym, name, d0, d1, ret))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cands))
    real, fake = idx[:20], idx[20:40]
    out = []
    for i in real:
        sym, name, d0, d1, ret = cands[i]
        out.append({"kind": "real", "statement": _text(name, d0, d1, ret > 0, abs(ret)), "symbol": sym})
    for i in fake:
        sym, name, d0, d1, ret = cands[i]
        out.append({"kind": "fabricated", "statement": _text(name, d0, d1, ret < 0, abs(ret)), "symbol": sym})
    rng.shuffle(out)
    return out


def _text(name: str, d0: date, d1: date, up: bool, mag: float) -> str:
    size = "more than 10%" if mag >= 0.10 else "more than 5%"
    verb = "rose" if up else "fell"
    sub = name if name.startswith("The S&P") else f"{name} stock"
    return f"{sub} {verb} {size} between {d0:%B} {d0.day}, {d0.year} and {d1:%B} {d1.day}, {d1.year}."


FORCED_SCHEMA = {"name": "g0_forced", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["answer"],
    "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}}}}
FORCED_SYSTEM = ("Is the following statement about a past event true? If you are unsure, give your best guess. "
                 "Reply yes or no.")


async def run(client: LLMClient, bars: pl.DataFrame, models: list[str], forced: bool = False) -> dict:
    """forced=False allows "don't know" (the plan's protocol); forced=True requires a yes/no guess, which
    exposes partial knowledge that abstention would hide."""
    stmts = build_statements(bars)
    schema, system = (FORCED_SCHEMA, FORCED_SYSTEM) if forced else (SCHEMA, SYSTEM)

    async def ask(model: str, s: dict):
        try:
            out = await client.chat(ChatRequest(
                model_key=model, purpose="g0_forced" if forced else "g0", budget="infra", json_schema=schema,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": s["statement"]}]))
            return json.loads(out.content)["answer"]
        except (LLMError, json.JSONDecodeError, KeyError) as e:
            return f"error:{type(e).__name__}"

    result = {"gate": "G0", "forced": forced, "statements": stmts, "models": {}}
    for m in models:
        answers = await asyncio.gather(*(ask(m, s) for s in stmts))
        rows = [dict(s, answer=a) for s, a in zip(stmts, answers)]
        yes = {k: np.mean([r["answer"] == "yes" for r in rows if r["kind"] == k]) for k in ("real", "fabricated")}
        acc = {"real": np.mean([r["answer"] == "yes" for r in rows if r["kind"] == "real"]),
               "fabricated": np.mean([r["answer"] == "no" for r in rows if r["kind"] == "fabricated"])}
        dk = np.mean([r["answer"] == "don't know" for r in rows])
        gap = float(yes["real"] - yes["fabricated"])
        result["models"][m] = {"yes_rate_real": float(yes["real"]), "yes_rate_fabricated": float(yes["fabricated"]),
                               "accuracy_real": float(acc["real"]), "accuracy_fabricated": float(acc["fabricated"]),
                               "dont_know_rate": float(dk), "knowledge_gap": gap, "pass": abs(gap) <= 0.10,
                               "answers": rows}
    result["pass"] = all(v["pass"] for v in result["models"].values())
    return result
