"""Non-LLM comparators, run free on the same 20-asset universe with the same execution rules (no compute cost).

equal-weight daily rebalanced, equal-weight buy-and-hold, 20-day momentum top 3, 5-day reversal top 3, and a ridge
model on 5d/20d returns plus news count, refit monthly on past data only. Each decides at day t's close using
data <= t and fills at the open of t+1.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from dlo import config
from dlo.agents.state import TrajState
from dlo.sim.execution import equity, execute

TOP = 3


def _closes(view_upto: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (view_upto.filter(pl.col("key").is_in(keys)).pivot(on="key", index="session", values="close")
            .sort("session"))


def _trailing(view_upto: pl.DataFrame, keys: list[str], n: int) -> dict[str, float]:
    w = _closes(view_upto, keys)
    if w.height <= n:
        return {k: 0.0 for k in keys}
    return {k: float(w[k][-1] / w[k][-1 - n] - 1) for k in keys}


def _top(scores: dict[str, float], reverse: bool) -> dict[str, float]:
    cap = config.costs()["max_weight"]
    order = sorted(scores, key=lambda k: scores[k], reverse=reverse)[:TOP]
    return {k: cap for k in order}


class RidgeModel:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.model: Ridge | None = None
        self.fit_month: tuple[int, int] | None = None

    def _frame(self, panel: pl.DataFrame, news_counts: dict[tuple[str, date], int]) -> pl.DataFrame:
        df = panel.filter(pl.col("key").is_in(self.keys)).sort(["key", "session"]).with_columns(
            (pl.col("close") / pl.col("close").shift(5).over("key") - 1).alias("r5"),
            (pl.col("close") / pl.col("close").shift(20).over("key") - 1).alias("r20"),
            (pl.col("open").shift(-2).over("key") / pl.col("open").shift(-1).over("key") - 1).alias("y"),
            pl.col("session").shift(-2).over("key").alias("y_known_on"),
        )
        nc = [news_counts.get((k, d), 0) for k, d in zip(df["key"].to_list(), df["session"].to_list())]
        return df.with_columns(pl.Series("news", nc, dtype=pl.Float64))

    def weights(self, full_panel: pl.DataFrame, day: date, news_counts) -> dict[str, float]:
        df = self._frame(full_panel.filter(pl.col("session") <= day), news_counts)
        # Strict look-ahead guard: the target of a training row must be observable by `day`.
        if self.fit_month != (day.year, day.month) or self.model is None:
            tr = df.filter(pl.col("y_known_on").is_not_null() & (pl.col("y_known_on") <= day)).drop_nulls(
                ["r5", "r20", "y"])
            if tr.height >= 50:
                self.model = Ridge(alpha=1.0).fit(tr.select("r5", "r20", "news").to_numpy(), tr["y"].to_numpy())
                self.fit_month = (day.year, day.month)
        today = df.filter(pl.col("session") == day).drop_nulls(["r5", "r20"])
        if self.model is None or today.height == 0:
            return {k: 1 / len(self.keys) for k in self.keys}
        pred = self.model.predict(today.select("r5", "r20", "news").to_numpy())
        return _top(dict(zip(today["key"].to_list(), pred.tolist())), reverse=True)


def run_baselines(view, days: list[date], keys: list[str], news_counts: dict[tuple[str, date], int]) -> dict[str, list[dict]]:
    """Return per-baseline daily rows: sim_day, net_equity, weights."""
    capital = config.costs()["capital_usd"]
    bps = config.costs()["trading_cost_bps"]
    panel = view.full_for_simulator()
    names = ["ew_daily", "ew_buy_hold", "mom20_top3", "rev5_top3", "ridge_top3"]
    states = {n: TrajState(traj_id=n, arm="baseline", model_key="none", seed=0, labels={}, cash=capital)
              for n in names}
    ridge = RidgeModel(keys)
    out: dict[str, list[dict]] = {n: [] for n in names}
    pending: dict[str, dict[str, float] | None] = {n: None for n in names}
    ew = {k: 1 / len(keys) for k in keys}
    for i, day in enumerate(days):
        opens, closes = view.prices(day, "open"), view.prices(day, "close")
        for n, st in states.items():
            execute(st, pending[n], opens, bps)
            pending[n] = None
        upto = view.upto(day)
        pending["ew_daily"] = ew
        if i == 0:
            pending["ew_buy_hold"] = ew
        pending["mom20_top3"] = _top(_trailing(upto, keys, 20), reverse=True)
        pending["rev5_top3"] = _top(_trailing(upto, keys, 5), reverse=False)
        pending["ridge_top3"] = ridge.weights(panel, day, news_counts)
        for n, st in states.items():
            net = equity(st, closes)
            out[n].append({"sim_day": day.isoformat(), "net_equity": net,
                           "weights": {k: sh * closes[k] / net for k, sh in st.shares.items()}})
    return out


__all__ = ["run_baselines", "np"]
