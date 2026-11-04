"""Weekly re-identification probe of the evaluated models (outside every trading trajectory, not charged).

Each model sees one asset's blind 60-day data and its last week of news, and is asked which of the 10 real
companies it is (or synthetic, or cannot tell), plus a forced real-vs-synthetic call. Yields the
re-identification rate (G3) and real-vs-synthetic accuracy (G2).
"""
from __future__ import annotations

import json
import uuid

import polars as pl

from dlo import config
from dlo.agents.state import _series
from dlo.anon.scrub import render
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError
from dlo.data.universe_view import MARKET_KEY, lookback_table
from dlo.store import RunStore, now_utc

PROBE_LABEL = "ASSET_X"


def options() -> list[str]:
    return [c["name"] for s in config.universe()["sectors"].values() for c in s["real"]] + \
        ["synthetic (not a real company)", "cannot tell"]


def schema() -> dict:
    return {"name": "probe", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["identity", "real_or_synthetic"],
        "properties": {"identity": {"type": "string", "enum": options()},
                       "real_or_synthetic": {"type": "string", "enum": ["real", "synthetic"]}}}}


SYSTEM = """You are shown anonymized daily data and rewritten news for one asset, labeled ASSET_X, alongside a broad market index (MARKET). The asset is either one of ten real US large-cap companies or a synthetic asset generated to resemble one. Identify it if you can."""


def probe_message(view_upto: pl.DataFrame, day, key: str, news_week: list[str], variant: str = "full") -> str:
    """variant: full (data and news, the plan's probe), data (no news) or news (no data); the last two are
    diagnostics that split the re-identification rate by channel."""
    tables = lookback_table(view_upto, day, 60)
    lines = []
    for k, name in ((MARKET_KEY, "MARKET"), (key, PROBE_LABEL)):
        g = tables[k]
        lines += [name, f" ret%  {_series(g['ret_pct'].to_list())}", f" gap%  {_series(g['gap_pct'].to_list())}",
                  f" rng%  {_series(g['range_pct'].to_list())}", f" close {_series(g['idx_close'].to_list())}",
                  f" rvol  {_series(g['rel_vol'].to_list())}"]
    news = [f"- {render(t, {key: PROBE_LABEL})}" for t in news_week] or ["- (none)"]
    opts = "\n".join(f"- {o}" for o in options())
    data = "DATA (last 60 trading days, oldest to newest)\n" + "\n".join(lines)
    news_s = "NEWS ABOUT ASSET_X (last 5 trading days)\n" + "\n".join(news)
    body = {"full": data + "\n\n" + news_s, "data": data + "\n\nNEWS ABOUT ASSET_X: not shown",
            "news": "DATA: not shown\n\n" + news_s}[variant]
    return (body + f"\n\nWhich is ASSET_X? Choose identity from:\n{opts}\n"
            "Also give your best forced choice: real or synthetic.")


async def run_probe(client: LLMClient, store: RunStore, run_id: str, week: int, day, view_upto: pl.DataFrame,
                    keys: list[str], news_by_key: dict[str, list[str]], synthetic_keys: set[str],
                    models: list[str], variant: str = "full") -> list[dict]:
    import asyncio
    names = {c["ticker"]: c["name"] for s in config.universe()["sectors"].values() for c in s["real"]}

    async def one(model: str, key: str):
        pid = f"{run_id}:{week}:{model}:{key}"
        if store.query("SELECT 1 FROM probes WHERE probe_id=?", (pid,)):
            return None
        msg = probe_message(view_upto, day, key, news_by_key.get(key, []), variant)
        try:
            out = await client.chat(ChatRequest(
                model_key=model, purpose="probe", budget="infra", sim_day=str(day), json_schema=schema(),
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]))
            d = json.loads(out.content)
            ans, ros, cid = d["identity"], d["real_or_synthetic"], out.call_id
        except (LLMError, json.JSONDecodeError, KeyError) as e:
            ans, ros, cid = f"error:{type(e).__name__}", None, None
        row = {"probe_id": pid, "run_id": run_id, "week": week, "sim_day": str(day), "model_key": model,
               "asset_key": key, "truth": names.get(key, "synthetic (not a real company)"),
               "is_synthetic": int(key in synthetic_keys), "answer": ans, "answer_synthetic": ros,
               "call_id": cid, "created_at": now_utc()}
        store.put("probes", row)
        return row

    rows = await asyncio.gather(*(one(m, k) for m in models for k in keys))
    return [r for r in rows if r]


__all__ = ["run_probe", "probe_message", "uuid"]
