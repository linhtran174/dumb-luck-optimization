"""Pre-generate all synthetic shocks, events and news specs for both windows plus lookbacks, with hashes."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import polars as pl

from dlo import config, logs
from dlo.anon.pipeline import scrubbed_for_day
from dlo.data import calendar
from dlo.data.snapshot import Snapshot
from dlo.store import RunStore, now_utc, sha256_file
from dlo.synth import donor_pool, generator

log = logs.get("pregen")
PREGEN_DIR = config.DATA_DIR / "pregen"


def timeline() -> list[date]:
    dev_start, _ = config.window("dev")
    lb = config.windows()["lookback_days"]
    start = calendar.sessions_before(dev_start, lb + 6)[0]
    end = config._d(config.windows()["snapshot"]["bars_end"])
    return calendar.sessions(start, end)


def window_sessions() -> list[date]:
    out = []
    for w in ("dev", "test"):
        a, b = config.window(w)
        out += calendar.sessions(a, b)
    return out


def real_scrubbed_counts(store: RunStore, snap: Snapshot, days: list[date]) -> pl.DataFrame:
    """Per real asset-day count of news items that survive scrubbing, i.e. what agents actually see."""
    rows, missing = [], 0
    for d in days:
        sc = scrubbed_for_day(store, snap, d)
        missing += sc["missing"]
        rows += [{"session": d, "ticker": t, "n": len(v)} for t, v in sc["assets"].items()]
    if missing:
        raise RuntimeError(f"{missing} selected news items are not scrubbed yet; run `dlo anon scrub all` first")
    return pl.DataFrame(rows)


def run(snap: Snapshot | None = None, store: RunStore | None = None) -> dict:
    snap = snap or Snapshot()
    store = store or RunStore()
    PREGEN_DIR.mkdir(parents=True, exist_ok=True)
    pools = donor_pool.build(snap.bars)
    seed = config.universe()["synthetic"]["seed"]
    specs = generator.make_specs(pools, seed)
    counts = real_scrubbed_counts(store, snap, window_sessions())
    counts.write_parquet(PREGEN_DIR / "real_news_counts.parquet")
    real_counts = counts["n"].to_numpy()
    mret = dict(zip(*donor_pool.market_returns(snap.bars).select("session", "mret").to_dict(as_series=False).values()))
    sessions = timeline()
    frames = [generator.generate(s, pools[s.sector], sessions, mret, real_counts) for s in specs]
    synth = pl.concat(frames)
    synth.write_parquet(PREGEN_DIR / "synthetic.parquet")
    pool_meta = {sec: {"donors": p.donors, "pair_beta": p.pair_beta, "pair_sigma": p.pair_sigma,
                       "n_days": [len(x) for x in p.resid]} for sec, p in pools.items()}
    (PREGEN_DIR / "specs.json").write_text(json.dumps(
        {"seed": seed, "specs": generator.spec_dicts(specs), "pools": pool_meta,
         "timeline": [str(sessions[0]), str(sessions[-1])], "real_count_dist":
             {str(k): int(v) for k, v in zip(*np.unique(real_counts, return_counts=True))}}, indent=2))
    manifest = {"created_at": now_utc(), "files": {}}
    for f in ("synthetic.parquet", "specs.json", "real_news_counts.parquet"):
        manifest["files"][f] = sha256_file(PREGEN_DIR / f)
    (PREGEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logs.event(log, "pregen_done", assets=len(specs), sessions=len(sessions))
    return {"specs": generator.spec_dicts(specs), "manifest": manifest}


def load() -> tuple[pl.DataFrame, list[dict]]:
    synth = pl.read_parquet(PREGEN_DIR / "synthetic.parquet")
    specs = json.loads((PREGEN_DIR / "specs.json").read_text())["specs"]
    return synth, specs
