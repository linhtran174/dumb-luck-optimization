"""Paper execution: fills at the next open, 5 bps per side, long-only, compute costs debited from cash.

Rules (exact, so the hand-computed toy test can match them):
* Equity at the open E = cash + sum(shares * open).
* Target value v_i = w_i * E. Trade notional d_i = v_i - shares_i * open_i. Cost = bps * sum|d_i|.
* Cash after = cash - sum(d_i) - cost.
* Cash reserve: if cash after the fill would fall below `reserve` (the next expected inference cost), all risky
  targets are scaled by a common factor k < 1 until cash after = reserve (fixed-point on k). Logged as forced.
* If the trajectory holds (no new target) and cash < reserve, the same pro-rata scaling is applied to holdings.
* Ruin: net equity <= 0 after any debit ends the trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass

from dlo.agents.state import TrajState


@dataclass
class Fill:
    traded_value: float
    cost: float
    forced: bool
    scale: float
    trades: dict[str, float]          # key -> traded notional (signed)


def equity(st: TrajState, prices: dict[str, float]) -> float:
    return st.cash + sum(sh * prices[k] for k, sh in st.shares.items())


def _plan(st: TrajState, targets: dict[str, float], prices: dict[str, float], bps: float, k: float):
    E = equity(st, prices)
    trades, gross = {}, 0.0
    for key in set(targets) | set(st.shares):
        cur = st.shares.get(key, 0.0) * prices[key]
        tgt = k * targets.get(key, 0.0) * E
        d = tgt - cur
        if abs(d) > 1e-12:
            trades[key] = d
            gross += abs(d)
    cost = bps / 1e4 * gross
    cash_after = st.cash - sum(trades.values()) - cost
    return trades, gross, cost, cash_after


def execute(st: TrajState, targets: dict[str, float] | None, prices: dict[str, float], bps: float,
            reserve: float = 0.0) -> Fill | None:
    """Apply target weights (by asset key) at open prices. targets=None means hold (reserve check only)."""
    if targets is None:
        E = equity(st, prices)
        if st.cash >= reserve or E <= 0:
            return None
        targets = {k: sh * prices[k] / E for k, sh in st.shares.items()}
        hold_forced = True
    else:
        hold_forced = False
    k, forced = 1.0, hold_forced
    trades, gross, cost, cash_after = _plan(st, targets, prices, bps, k)
    if cash_after < reserve - 1e-12:
        forced = True
        E = equity(st, prices)
        risky = sum(targets.values()) * E
        for _ in range(50):
            if risky <= 0:
                break
            # cash_after(k) = E - k*risky - cost(k)  ->  solve E - k*risky - cost = reserve
            k_new = max(0.0, min(1.0, (E - cost - reserve) / risky))
            trades, gross, cost, cash_after = _plan(st, targets, prices, bps, k_new)
            if abs(k_new - k) < 1e-13:
                k = k_new
                break
            k = k_new
    if not trades:
        return None
    for key, d in trades.items():
        st.shares[key] = st.shares.get(key, 0.0) + d / prices[key]
        if abs(st.shares[key]) < 1e-12:
            del st.shares[key]
    st.cash = cash_after
    st.trading_cost_cum += cost
    return Fill(traded_value=gross, cost=cost, forced=forced, scale=k, trades=trades)


def charge_compute(st: TrajState, cost: float) -> None:
    st.cash -= cost
    st.compute_cost_cum += cost
    st.recent_call_costs = (st.recent_call_costs + [cost])[-5:]


def check_ruin(st: TrajState, prices: dict[str, float], day: str) -> bool:
    if not st.ruined and equity(st, prices) <= 0:
        st.ruined, st.ruin_day = True, day
    return st.ruined


def expected_call_cost(st: TrajState, default: float) -> float:
    return max(st.recent_call_costs) if st.recent_call_costs else default
