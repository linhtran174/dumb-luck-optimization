"""Standalone G3/G2 probe over many weeks of a window, outside any trajectory (infra budget, not charged).

Uses the same probe as the replay's weekly probe, on every 5th session of the window, with the current scrub
version's news. Variants split the leak by channel: full (the plan's probe), data (no news), news (no data).
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

import numpy as np
import polars as pl

from dlo import config
from dlo.anon.pipeline import scrub_days, scrubbed_for_day
from dlo.anon.probe import run_probe
from dlo.anon.scrub import SCRUB_VERSION
from dlo.clients.openrouter_chat import LLMClient
from dlo.data import calendar
from dlo.data.snapshot import Snapshot
from dlo.data.universe_view import UniverseView, asset_keys
from dlo.store import RunStore
from dlo.synth import pregenerate
from dlo.synth.news_gen import generate_day, stored_texts


def run_id(window: str, variant: str) -> str:
    return f"g3-{window}-{variant}-{SCRUB_VERSION}"


async def run(client: LLMClient, window: str, weeks: int, variant: str, models: list[str],
              real_only: bool = False, skip_weeks: int = 0) -> dict:
    store, snap = RunStore(), Snapshot()
    synth, specs = pregenerate.load()
    a, b = config.window(window)
    days = calendar.sessions(a, b)
    view = UniverseView(snap, synth, specs, window_start=days[0])
    keys = asset_keys(specs)
    if real_only:
        keys = [k for k in keys if k in config.real_tickers()]
    probe_idx = list(range(4, len(days), 5))[skip_weeks:skip_weeks + weeks]
    rid = run_id(window, variant)
    for i in probe_idx:
        week_days = days[i - 4:i + 1]
        await scrub_days(client, store, snap, week_days)
        news: dict[str, list[str]] = defaultdict(list)
        rows = synth.filter(pl.col("session").is_in(week_days) & (pl.col("news_count") > 0)).to_dicts()

        async def syn(r):
            z = None if r["sentiment"] is None or r["sentiment"] != r["sentiment"] else float(r["sentiment"])
            await generate_day(client, store, r["asset"], r["sector"], r["session"].isoformat(),
                               int(r["news_count"]), bool(r["event"]), r["event_type"], z)
        if variant != "data":
            await asyncio.gather(*(syn(r) for r in rows))
        for d in week_days:
            for k, v in scrubbed_for_day(store, snap, d)["assets"].items():
                news[k].extend(v)
            for k, v in stored_texts(store, d.isoformat()).items():
                news[k].extend(v)
        await run_probe(client, store, rid, i // 5 + 1, days[i], view.upto(days[i]), keys, dict(news),
                        {s["key"] for s in specs}, models, variant)
    return summarize(store, rid)


def summarize(store: RunStore, rid: str) -> dict:
    sector = {c["name"]: sec for sec, d in config.universe()["sectors"].items() for c in d["real"]}
    spec_sector = {s["key"]: s["sector"] for s in pregenerate.load()[1]}
    rows = [dict(r) for r in store.query("SELECT * FROM probes WHERE run_id=?", (rid,))]
    out: dict = {"run_id": rid, "models": {}}
    for m in sorted({r["model_key"] for r in rows}):
        ps = [r for r in rows if r["model_key"] == m and not str(r["answer"]).startswith("error")]
        real = [r for r in ps if not r["is_synthetic"]]
        syn = [r for r in ps if r["is_synthetic"]]
        true_sec = lambda r: sector.get(r["truth"]) or spec_sector.get(r["asset_key"])  # noqa: E731
        by_asset = defaultdict(list)
        for r in real:
            by_asset[r["asset_key"]].append(r["answer"] == r["truth"])
        acc = [(r["answer_synthetic"] == "synthetic") == bool(r["is_synthetic"]) for r in ps if r["answer_synthetic"]]
        reid = float(np.mean([r["answer"] == r["truth"] for r in real])) if real else None
        out["models"][m] = {
            "n_real": len(real), "n_synth": len(syn), "weeks": len({r["week"] for r in ps}),
            "reidentification_rate": reid, "G3_pass": reid is not None and reid <= 0.20,
            "sector_hit_rate_real": float(np.mean([sector.get(r["answer"]) == true_sec(r) for r in real])) if real else None,
            "sector_hit_rate_synth": float(np.mean([sector.get(r["answer"]) == true_sec(r) for r in syn])) if syn else None,
            "cannot_tell_rate": float(np.mean([r["answer"] == "cannot tell" for r in ps])) if ps else None,
            "says_synthetic_rate": {"real": float(np.mean([r["answer"] == "synthetic (not a real company)" for r in real])) if real else None,
                                    "synth": float(np.mean([r["answer"] == "synthetic (not a real company)" for r in syn])) if syn else None},
            "real_vs_synth_accuracy": float(np.mean(acc)) if (acc and syn) else None,
            "G2_pass": bool(acc and syn and abs(np.mean(acc) - 0.5) <= 0.10),
            "reid_by_asset": {k: float(np.mean(v)) for k, v in sorted(by_asset.items())}}
    return out
