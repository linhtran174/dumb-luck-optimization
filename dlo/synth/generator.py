"""Synthetic asset generator: r_t = beta * r_M,t + sigma * eps_t, with signal planted only in eps.

eta  : stationary block bootstrap (mean block 10) over a sector's donor residuals; each step also records the
       donor day it came from, whose candle shape and relative volume the synthetic day borrows.
eps  : noise      eps = eta
       momentum   eps_t = a * s_{t-1} + sqrt(1-a^2) * eta_t,  s_{t-1} = sqrt(5) * mean(eps_{t-5..t-1}) / c(a),
           c(a) = stationary sd of the raw trailing mean, so s has unit variance and corr(eps, s) = a
       news_lead  eps_{t+1} = c * z_t + sqrt(1-c^2) * eta_{t+1} when day t carried an event with sentiment z_t
Events happen on `event_day_prob` of days for every synthetic asset; only news_lead events predict returns.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from datetime import date

import numpy as np
import polars as pl

from dlo import config
from dlo.synth.donor_pool import SectorPool

EVENT_TYPES = ["earnings", "guidance", "analyst_action", "product", "contract", "regulatory", "management"]
MOM_WINDOW = 5
# Generator calibration, tuned in M3 against gate G1 on pre-test data only (2024-01 .. dev window end).
RANGE_ELASTICITY = 0.5     # 0 recovers the plan's candle rule exactly
VOLUME_ELASTICITY = 0.3
VOLUME_SHRINK = 0.92       # 1 = donor relative volume as is
BETA_JITTER = (0.82, 1.08) # multiplier on one real sector company's beta
SIGMA_JITTER = (0.95, 1.20)
BETA_DRIFT = 0.0


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
    reals = {sec: config.universe()["sectors"][sec]["real"] for sec in config.universe()["sectors"]}
    for i, gi in enumerate(order):
        g, strength = groups[gi]
        sec = sectors[i]
        p = pools[sec]
        # beta/sigma near one real company of the sector (not the pair average, which inflated market correlation)
        j = int(rng.integers(len(reals[sec])))
        specs.append(SynthSpec(
            key=f"S{i + 1:02d}", group=g, strength=float(strength), sector=sec,
            beta=float(p.real_beta[j] * rng.uniform(*BETA_JITTER)),
            sigma=float(p.real_sigma[j] * rng.uniform(*SIGMA_JITTER)),
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


@lru_cache
def momentum_scale(a: float, n: int = 400_000, iters: int = 6) -> float:
    """Stationary sd of sqrt(5)*mean(eps_{t-5..t-1}) under the planted process, so s has unit variance.
    eps feeds back into s, so the naive sqrt(5) scaling overstates neither the signal nor eps' variance only when
    the scale is solved jointly; fixed point on Gaussian innovations (deterministic seed)."""
    c = 1.0
    eta = np.random.default_rng(12345).standard_normal(n)
    k = np.sqrt(1 - a * a)
    for _ in range(iters):
        eps = np.zeros(n)
        raw = np.zeros(n)
        for t in range(MOM_WINDOW, n):
            raw[t] = np.sqrt(MOM_WINDOW) * eps[t - MOM_WINDOW:t].mean()
            eps[t] = a * raw[t] / c + k * eta[t]
        c = float(raw[1000:].std())
    return c


def plant(group: str, strength: float, eta: np.ndarray, events: np.ndarray, z: np.ndarray):
    """Return eps and the planted signal series aligned to eps (s_{t-1} for momentum, z_{t-1} for news)."""
    n = len(eta)
    eps = np.empty(n)
    sig = np.full(n, np.nan)
    k = np.sqrt(1.0 - strength**2)
    c = momentum_scale(round(strength, 6)) if group == "momentum" else 1.0
    for t in range(n):
        if group == "momentum":
            if t >= MOM_WINDOW:
                s = np.sqrt(MOM_WINDOW) * eps[t - MOM_WINDOW:t].mean() / c
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
    # Slowly drifting beta (AR(1), rho=0.98, stationary sd BETA_DRIFT in log terms); real betas wander too.
    x = np.zeros(n)
    innov = rng.standard_normal(n) * np.sqrt(1 - 0.98**2)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = 0.98 * x[t - 1] + innov[t]
    beta_t = spec.beta * np.exp(BETA_DRIFT * x - 0.5 * BETA_DRIFT**2)
    r = beta_t * rm + spec.sigma * eps

    # Candles: borrow the donor day's shape. The plan's rule (shift open by half and high/low/close by the full
    # log-return difference) is generalized with a magnitude factor so the intraday range and volume scale with
    # the synthetic day's move, as they do for real stocks (G1 failed at AUC 0.70 without it). scale=1 recovers
    # the plan's rule exactly.
    gap_d = np.array([pool.gap[a][b] for a, b in zip(di, ti)])
    hi_d = np.array([pool.hi[a][b] for a, b in zip(di, ti)])
    lo_d = np.array([pool.lo[a][b] for a, b in zip(di, ti)])
    ret_d = np.array([pool.ret[a][b] for a, b in zip(di, ti)])
    rv_d = np.array([pool.relvol[a][b] for a, b in zip(di, ti)])
    # Rescale each borrowed candle from the donor's volatility to this asset's volatility first.
    donor_vol = np.array([np.std(x) for x in pool.ret])
    f = float(np.std(r[1:])) / donor_vol[di]
    gap_d, hi_d, lo_d, ret_d = gap_d * f, hi_d * f, lo_d * f, ret_d * f
    m = float(np.median(np.abs(r[1:])))
    scale = np.clip((np.abs(r) + m) / (np.abs(ret_d) + m), 0.4, 3.0)
    sr = scale ** RANGE_ELASTICITY
    shift = r - sr * ret_d
    lo_open = sr * gap_d + shift / 2
    lo_hi = sr * hi_d + shift
    lo_lo = sr * lo_d + shift
    lo_close = r
    lo_hi = np.maximum.reduce([lo_hi, lo_open, lo_close])
    lo_lo = np.minimum.reduce([lo_lo, lo_open, lo_close])
    rv_d = np.exp(VOLUME_SHRINK * np.log(np.clip(rv_d, 1e-6, None))) * scale ** VOLUME_ELASTICITY

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
