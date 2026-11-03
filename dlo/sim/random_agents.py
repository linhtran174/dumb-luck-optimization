"""1,000 random agents: Dirichlet weights on review days, sleep drawn from the LLMs' pooled distribution.

Vectorized version of the same execution rules (fill at next open, 5 bps per side, 30% cap, no compute cost).
"""
from __future__ import annotations

import numpy as np

from dlo import config


def simulate(opens: np.ndarray, closes: np.ndarray, sleep_values: np.ndarray, sleep_probs: np.ndarray,
             n_agents: int = 1000, seed: int = 0) -> np.ndarray:
    """opens/closes: [T, A] price matrices for the window. Returns net equity [n_agents, T] at each close."""
    rng = np.random.default_rng(seed)
    cap = config.costs()["max_weight"]
    bps = config.costs()["trading_cost_bps"] / 1e4
    T, A = opens.shape
    cash = np.full(n_agents, config.costs()["capital_usd"])
    shares = np.zeros((n_agents, A))
    pending = np.full((n_agents, A), np.nan)
    next_review = np.zeros(n_agents, dtype=int)
    eq = np.zeros((n_agents, T))
    for t in range(T):
        has = ~np.isnan(pending[:, 0])
        if has.any():
            E = cash[has] + shares[has] @ opens[t]
            tgt_val = pending[has] * E[:, None]
            cur_val = shares[has] * opens[t]
            d = tgt_val - cur_val
            cost = bps * np.abs(d).sum(1)
            cash[has] = cash[has] - d.sum(1) - cost
            shares[has] = tgt_val / opens[t]
            pending[has] = np.nan
        eq[:, t] = cash + shares @ closes[t]
        due = next_review <= t
        if due.any():
            w = rng.dirichlet(np.ones(A + 1), size=int(due.sum()))
            risky = w[:, :A]
            excess = np.clip(risky - cap, 0, None).sum(1)
            risky = np.minimum(risky, cap)
            pending[due] = risky
            _ = excess  # the clipped excess stays in cash
            next_review[due] = t + rng.choice(sleep_values, size=int(due.sum()), p=sleep_probs)
    return eq
