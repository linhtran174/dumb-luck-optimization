"""Donor pool: standardized SPY-residuals and candle shapes of same-sector stocks outside the universe.

Estimated only on pre-window history (bars_start .. day before the dev window), so the generator never
uses information from either replay window. Real-pair beta/sigma come from the same period.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from dlo import config


@dataclass
class SectorPool:
    sector: str
    donors: list[str]
    dates: list[list[date]]          # per donor, aligned with arrays below
    resid: list[np.ndarray]          # standardized residuals, per donor
    ret: list[np.ndarray]            # donor close-to-close log return
    gap: list[np.ndarray]            # log(open / prev close)
    hi: list[np.ndarray]             # log(high / prev close)
    lo: list[np.ndarray]             # log(low / prev close)
    relvol: list[np.ndarray]         # volume / median of previous 60 sessions
    pair_beta: float
    pair_sigma: float
    real_beta: list[float]
    real_sigma: list[float]


def estimation_end() -> date:
    return config.window("dev")[0]


def _features(bars: pl.DataFrame, sym: str, spy: pl.DataFrame) -> pl.DataFrame:
    g = bars.filter(pl.col("symbol") == sym).sort("session")
    g = g.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("ret"),
        (pl.col("open") / pl.col("close").shift(1)).log().alias("gap"),
        (pl.col("high") / pl.col("close").shift(1)).log().alias("hi"),
        (pl.col("low") / pl.col("close").shift(1)).log().alias("lo"),
        (pl.col("volume") / pl.col("volume").shift(1).rolling_median(60)).alias("relvol"),
    )
    return g.join(spy, on="session", how="inner").drop_nulls(["ret", "relvol", "mret"])


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return float(coef[0]), float(coef[1]), resid


AR_LAGS = 5


def prewhiten(e: np.ndarray, lags: int = AR_LAGS) -> np.ndarray:
    """AR(lags) innovations of a residual series, standardized. Removes linear serial dependence (which would
    otherwise plant an unintended reversal/momentum signal in 'pure noise' assets) while keeping volatility
    clustering, which lives in |e| rather than e."""
    X = np.column_stack([e[lags - k - 1:len(e) - k - 1] for k in range(lags)])
    yv = e[lags:]
    phi, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yv)), X]), yv, rcond=None)
    u = yv - np.column_stack([np.ones(len(yv)), X]) @ phi
    return u / u.std(ddof=1)


def market_returns(bars: pl.DataFrame) -> pl.DataFrame:
    m = config.universe()["market"]
    return (bars.filter(pl.col("symbol") == m).sort("session")
            .with_columns((pl.col("close") / pl.col("close").shift(1)).log().alias("mret"))
            .select("session", "mret").drop_nulls())


def build(bars: pl.DataFrame, end: date | None = None) -> dict[str, SectorPool]:
    end = end or estimation_end()
    hist = bars.filter(pl.col("session") < end)
    spy = market_returns(hist)
    pools: dict[str, SectorPool] = {}
    for sector, sec in config.universe()["sectors"].items():
        donors, dates, resid, ret, gap, hi, lo, rv = [], [], [], [], [], [], [], []
        for d in sec["donors"]:
            f = _features(hist, d, spy)
            if f.height < 250:
                continue
            _, _, e = _ols(f["ret"].to_numpy(), f["mret"].to_numpy())
            L = AR_LAGS
            donors.append(d)
            dates.append(f["session"].to_list()[L:])
            resid.append(prewhiten(e / e.std(ddof=1)))
            ret.append(f["ret"].to_numpy()[L:])
            gap.append(f["gap"].to_numpy()[L:])
            hi.append(f["hi"].to_numpy()[L:])
            lo.append(f["lo"].to_numpy()[L:])
            rv.append(f["relvol"].to_numpy()[L:])
        betas, sigmas = [], []
        for c in sec["real"]:
            f = _features(hist, c["ticker"], spy)
            _, b, e = _ols(f["ret"].to_numpy(), f["mret"].to_numpy())
            betas.append(b)
            sigmas.append(float(e.std(ddof=1)))
        pools[sector] = SectorPool(sector, donors, dates, resid, ret, gap, hi, lo, rv,
                                   float(np.mean(betas)), float(np.mean(sigmas)), betas, sigmas)
    return pools
