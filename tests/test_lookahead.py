"""M2 poison-row test: rows timestamped after the simulated day's cutoff must never reach a prompt."""
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

from dlo import config
from dlo.agents.state import TrajState, build_user_message, permutation_labels
from dlo.anon.select import select_for_day
from dlo.data import calendar
from dlo.data.snapshot import Snapshot
from dlo.data.universe_view import UniverseView, asset_keys

DAY = calendar.sessions(config.window("trial")[0], config.window("trial")[1])[2]
SPECS = [{"key": f"S{i:02d}"} for i in range(1, 11)]


def _bars(sessions, poison_after=None):
    rng = np.random.default_rng(0)
    rows = []
    for sym in config.real_tickers() + [config.universe()["market"]]:
        p = 100.0
        for d in sessions:
            p *= float(np.exp(rng.normal(0, 0.01)))
            o, c = p * 0.999, p
            if poison_after and d > poison_after:
                o = c = 123456.0
            rows.append({"symbol": sym, "session": d, "open": o, "high": max(o, c) * 1.01, "low": min(o, c) * 0.99,
                         "close": c, "volume": 1e6 * (1 + rng.random()), "vwap": c, "trades": 1})
    return pl.DataFrame(rows)


def _synth(sessions, poison_after=None):
    rows = []
    for s in SPECS:
        for j, d in enumerate(sessions):
            c = 100 + j * 0.1
            if poison_after and d > poison_after:
                c = 654321.0
            rows.append({"asset": s["key"], "session": d, "open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": 1e7})
    return pl.DataFrame(rows)


def _news(poison=False):
    cut = calendar.cutoff_utc(DAY)
    rows = [{"id": 1, "created_at": cut - timedelta(hours=2), "updated_at": None, "headline": "NVDA clean item",
             "summary": "s", "symbols": ["NVDA"], "source": "b", "query_tags": ["NVDA"]}]
    if poison:
        rows.append({"id": 2, "created_at": cut + timedelta(minutes=1), "updated_at": None,
                     "headline": "POISON future NVDA item", "summary": "", "symbols": ["NVDA"], "source": "b",
                     "query_tags": ["NVDA"]})
    return pl.DataFrame(rows).with_columns(pl.col("created_at").dt.replace_time_zone("UTC"))


def _message(poison: bool) -> str:
    sessions = calendar.sessions_before(DAY, 80) + calendar.sessions(DAY, calendar.next_session(DAY, 5))
    snap = Snapshot(bars=_bars(sessions, DAY if poison else None), news=_news(poison))
    view = UniverseView(snap, _synth(sessions, DAY if poison else None), SPECS, window_start=sessions[80 - 2])
    sel = select_for_day(snap, DAY)
    news = {t: [it["headline"] for it in items] for t, items in sel["assets"].items()}
    keys = asset_keys(SPECS)
    st = TrajState(traj_id="t", arm="BLIND-FULL", model_key="luna", seed=1,
                   labels=permutation_labels(keys, "BLIND-FULL", 1), cash=1000.0)
    upto = view.upto(DAY)
    assert upto["session"].max() == DAY
    return build_user_message(st, upto, DAY, 2, news, [])


def test_poison_rows_never_reach_prompt():
    clean, poisoned = _message(False), _message(True)
    assert "POISON" not in poisoned
    assert "123456" not in poisoned and "654321" not in poisoned
    assert clean == poisoned


def test_message_has_no_real_identifiers():
    msg = _message(False)
    for t in config.real_tickers():
        # the raw headline "NVDA clean item" is rendered as given in this test, so skip the news line itself
        body = "\n".join(l for l in msg.splitlines() if "clean item" not in l)
        assert t not in body
    assert str(DAY.year) not in msg
