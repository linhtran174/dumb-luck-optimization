"""Synthetic asset generator: r_t = beta * r_M,t + sigma * eps_t, with signal planted only in eps.

eta  : stationary block bootstrap (mean block 10) over a sector's donor residuals; each step also records the
       donor day it came from, whose candle shape and relative volume the synthetic day borrows.
eps  : noise      eps = eta
       momentum   eps_t = a * s_{t-1} + sqrt(1-a^2) * eta_t,  s_{t-1} = sqrt(5) * mean(eps_{t-5..t-1})
       news_lead  eps_{t+1} = c * z_t + sqrt(1-c^2) * eta_{t+1} when day t carried an event with sentiment z_t
Events happen on `event_day_prob` of days for every synthetic asset; only news_lead events predict returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import polars as pl

from dlo import config
from dlo.synth.donor_pool import SectorPool

EVENT_TYPES = ["earnings", "guidance", "analyst_action", "product", "contract", "regulatory", "management"]
MOM_WINDOW = 5


@dataclass
class SynthSpec:
    key: str            # S01..S10
    group: str          # noise | momentum | news_lead
    strength: float     # a or c (0 for noise)
    sector: str
    beta: float
    sigma: float
    seed: int


def make_specs(pools: dict[str, SectorPool], seed: int) -> list[SynthSpec]:
    cfg = config.universe()["synthetic"]["groups"]
    groups = [("noise", 0.0)] * cfg["noise"]["count"]
    groups += [("momentum", a) for a in cfg["momentum"]["strengths"]]
    groups += [("news_lead", c) for c in cfg["news_lead"]["strengths"]]
    rng = np.random.default_rng(seed)
    sectors = [s for s in config.universe()["sectors"] for _ in range(2)]
    rng.shuffle(sectors)
    order = rng.permutation(len(groups))
    specs = []
    for i, gi in enumerate(order):
        g, strength = groups[gi]
        sec = sectors[i]
        p = pools[sec]
        specs.append(SynthSpec(
            key=f"S{i + 1:02d}", group=g, strength=float(strength), sector=sec,
            beta=float(p.pair_beta + rng.normal(0, 0.1)),
            sigma=float(p.pair_sigma * rng.uniform(0.85, 1.15)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ))
    return specs


def bootstrap_eta(pool: SectorPool, n: int, rng: np.random.Generator, mean_block: float = 10.0):
    """Stationary block bootstrap. Returns eta[n], donor_idx[n], day_idx[n]."""
    eta = np.empty(n)
    di = np.empty(n, dtype=np.int64)
    ti = np.empty(n, dtype=np.int64)
    p_new = 1.0 / mean_block
    d = t = -1
    for k in range(n):
        if d < 0 or rng.random() < p_new or t + 1 >= len(pool.resid[d]):
            d = int(rng.integers(len(pool.donors)))
            t = int(rng.integers(len(pool.resid[d])))
        else:
            t += 1
        eta[k], di[k], ti[k] = pool.resid[d][t], d, t
    return eta, di, ti


def plant(group: str, strength: float, eta: np.ndarray, events: np.ndarray, z: np.ndarray):
    """Return eps and the planted signal series aligned to eps (s_{t-1} for momentum, z_{t-1} for news)."""
    n = len(eta)
    eps = np.empty(n)
    sig = np.full(n, np.nan)
    k = np.sqrt(1.0 - strength**2)
    for t in range(n):
        if group == "momentum":
            if t >= MOM_WINDOW:
                s = np.sqrt(MOM_WINDOW) * eps[t - MOM_WINDOW:t].mean()
                sig[t] = s
                eps[t] = strength * s + k * eta[t]
            else:
                eps[t] = eta[t]
        elif group == "news_lead":
            if t >= 1 and events[t - 1]:
                sig[t] = z[t - 1]
                eps[t] = strength * z[t - 1] + k * eta[t]
            else:
                eps[t] = eta[t]
        else:
            eps[t] = eta[t]
    return eps, sig


def _count_sampler(real_counts: np.ndarray, p_event: float):
    """Draw (event, count) so the marginal count distribution equals the real one and event days have >= 1 item."""
    vals, freq = np.unique(real_counts, return_counts=True)
    prob = freq / freq.sum()
    p0 = float(prob[vals == 0].sum()) if (vals == 0).any() else 0.0
    if 1 - p0 < p_event:
        raise ValueError("too few real asset-days with news to place events")
    pos = vals > 0
    cond_pos = np.where(pos, prob, 0) / (1 - p0)                     # P(k | k>=1)
    non_event = (prob - p_event * cond_pos) / (1 - p_event)          # makes the marginal exact
    non_event = np.clip(non_event, 0, None)
    non_event /= non_event.sum()

    def draw(rng: np.random.Generator) -> tuple[bool, int]:
        if rng.random() < p_event:
            return True, int(rng.choice(vals, p=cond_pos))
        return False, int(rng.choice(vals, p=non_event))

    return draw


def generate(spec: SynthSpec, pool: SectorPool, sessions: list[date], market_ret: dict[date, float],
             real_counts: np.ndarray, p_event: float | None = None, mean_block: float | None = None,
             seed: int | None = None) -> pl.DataFrame:
    """Generate one synthetic asset over `sessions` (first session only seeds the price)."""
    ucfg = config.universe()["synthetic"]
    p_event = ucfg["event_day_prob"] if p_event is None else p_event
    mean_block = ucfg["bootstrap_mean_block"] if mean_block is None else mean_block
    rng = np.random.default_rng(spec.seed if seed is None else seed)
    n = len(sessions)
    eta, di, ti = bootstrap_eta(pool, n, rng, mean_block)
    draw = _count_sampler(real_counts, p_event)
    ev_cnt = [draw(rng) for _ in range(n)]
    events = np.array([e for e, _ in ev_cnt])
    counts = np.array([c for _, c in ev_cnt])
    z = np.where(events, rng.standard_normal(n), np.nan)
    etype = np.array([EVENT_TYPES[int(rng.integers(len(EVENT_TYPES)))] if e else "" for e in events])
    eps, sig = plant(spec.group, spec.strength, eta, events, np.nan_to_num(z))

    rm = np.array([market_ret.get(d, 0.0) for d in sessions])
    r = spec.beta * rm + spec.sigma * eps

    # Candles: borrow donor-day shape, shifted toward the synthetic return.
    gap_d = np.array([pool.gap[a][b] for a, b in zip(di, ti)])
    hi_d = np.array([pool.hi[a][b] for a, b in zip(di, ti)])
    lo_d = np.array([pool.lo[a][b] for a, b in zip(di, ti)])
    ret_d = np.array([pool.ret[a][b] for a, b in zip(di, ti)])
    rv_d = np.array([pool.relvol[a][b] for a, b in zip(di, ti)])
    delta = r - ret_d
    lo_open = gap_d + delta / 2
    lo_hi = hi_d + delta
    lo_lo = lo_d + delta
    lo_close = r
    lo_hi = np.maximum.reduce([lo_hi, lo_open, lo_close])
    lo_lo = np.minimum.reduce([lo_lo, lo_open, lo_close])

    r[0] = 0.0
    close = 100.0 * np.exp(np.cumsum(r))
    prev_close = np.concatenate([[100.0], close[:-1]])
    base_vol = 1.0e7
    return pl.DataFrame({
        "asset": spec.key, "session": sessions, "group": spec.group, "strength": spec.strength,
        "sector": spec.sector,
        "open": np.where(np.arange(n) == 0, 100.0, prev_close * np.exp(lo_open)),
        "high": np.where(np.arange(n) == 0, 100.0, prev_close * np.exp(lo_hi)),
        "low": np.where(np.arange(n) == 0, 100.0, prev_close * np.exp(lo_lo)),
        "close": close,
        "volume": base_vol * rv_d,
        "ret": np.where(np.arange(n) == 0, 0.0, r),
        "mret": rm, "eta": eta, "eps": eps, "signal": sig,
        "event": events, "sentiment": z, "event_type": etype, "news_count": counts,
        "donor": [pool.donors[a] for a in di], "donor_session": [pool.dates[a][b] for a, b in zip(di, ti)],
    })


def spec_dicts(specs: list[SynthSpec]) -> list[dict]:
    return [asdict(s) for s in specs]
