"""Trajectory state and the daily user message. Only blind labels and day indexes ever appear in the text."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np
import polars as pl

from dlo.agents.schema import LABELS
from dlo.anon.scrub import render
from dlo.data.universe_view import MARKET_KEY, fmt, lookback_table

MAX_STATE_BYTES = 30 * 1024


@dataclass
class TrajState:
    traj_id: str
    arm: str
    model_key: str
    seed: int
    labels: dict[str, str]                         # asset key -> ASSET_xx (stored outside the prompt path)
    cash: float
    shares: dict[str, float] = field(default_factory=dict)
    compute_cost_cum: float = 0.0
    trading_cost_cum: float = 0.0
    pending: dict[str, float] | None = None        # target weights by asset key, applied at next open
    next_review_index: int = 0
    history: list[dict] = field(default_factory=list)   # last decisions (label space)
    last_fill: dict | None = None
    memo: str | None = None
    memo_round: int = 0
    ruined: bool = False
    ruin_day: str | None = None
    recent_call_costs: list[float] = field(default_factory=list)
    n_calls: int = 0

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "TrajState":
        return TrajState(**d)

    def label_to_key(self) -> dict[str, str]:
        return {v: k for k, v in self.labels.items()}


def permutation_labels(keys: list[str], arm: str, seed: int) -> dict[str, str]:
    """Blind labels per (arm, seed); shared by both models so model comparisons are paired."""
    h = int(hashlib.sha256(f"labels:{arm}:{seed}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(h)
    perm = rng.permutation(len(keys))
    return {keys[i]: LABELS[j] for j, i in enumerate(perm)}


def _series(vals: list, nd: int = 1) -> str:
    return " ".join(fmt(v, nd) for v in vals)


SHORT_LOOKBACK = 20   # gap%, rng%, rvol shown for the last 20 days; ret% and close for all 60 (30 KB budget)


def market_block(tables: dict[str, pl.DataFrame], labels: dict[str, str]) -> str:
    lines = []
    n = SHORT_LOOKBACK
    order = [MARKET_KEY] + sorted(labels, key=lambda k: labels[k])
    for key in order:
        g = tables.get(key)
        if g is None:
            continue
        name = "MARKET" if key == MARKET_KEY else labels[key]
        lines.append(f"{name}")
        lines.append(f" ret%  {_series(g['ret_pct'].to_list())}")
        lines.append(f" close {_series(g['idx_close'].to_list(), 0)}")
        lines.append(f" gap%  {_series(g['gap_pct'].to_list()[-n:])}")
        lines.append(f" rng%  {_series(g['range_pct'].to_list()[-n:])}")
        lines.append(f" rvol  {_series(g['rel_vol'].to_list()[-n:])}")
    return "\n".join(lines)


def news_block(news: dict[str, list[str]], market: list[str], labels: dict[str, str]) -> str:
    render_map = dict(labels)
    out = ["NEWS (since the previous decision time)"]
    any_item = False
    for key in sorted(labels, key=lambda k: labels[k]):
        items = news.get(key) or []
        for t in items:
            out.append(f"- {labels[key]}: {render(t, render_map)}")
            any_item = True
    if not any_item:
        out.append("- (no asset news)")
    out.append("MARKET HEADLINES")
    out += [f"- {render(t, render_map)}" for t in market] or ["- (none)"]
    return "\n".join(out)


def portfolio_block(st: TrajState, close: dict[str, float], day_index: int) -> str:
    pos_val = {k: st.shares.get(k, 0.0) * close[k] for k in st.labels if st.shares.get(k, 0.0)}
    net = st.cash + sum(pos_val.values())
    gross = net + st.compute_cost_cum
    lines = [f"DAY INDEX {day_index} (D0 = today)",
             f"PORTFOLIO cash={st.cash:.2f} net_equity={net:.2f} gross_equity_before_compute={gross:.2f} "
             f"cumulative_compute_cost={st.compute_cost_cum:.4f} cumulative_trading_cost={st.trading_cost_cum:.4f}"]
    if pos_val:
        parts = [f"{st.labels[k]} value={v:.2f} weight={v / net:.3f}" for k, v in
                 sorted(pos_val.items(), key=lambda kv: st.labels[kv[0]]) if net > 0]
        lines.append("POSITIONS " + "; ".join(parts))
    else:
        lines.append("POSITIONS none (all cash)")
    if st.last_fill:
        f = st.last_fill
        lines.append(f"LAST FILLS at D{f['day_offset']} open: traded_value={f['traded_value']:.2f} "
                     f"trading_cost={f['cost']:.4f}" + (" (forced sale to cover inference costs)" if f.get("forced") else ""))
    return "\n".join(lines)


def history_block(st: TrajState, day_index: int) -> str:
    if not st.history:
        return "YOUR RECENT DECISIONS none yet"
    lines = ["YOUR RECENT DECISIONS (most recent last)"]
    for h in st.history[-5:]:
        w = ", ".join(f"{k} {v:.2f}" for k, v in sorted(h["weights"].items()) if v > 0)
        lines.append(f"- D{h['day_index'] - day_index}: weights [{w or 'none'}] cash {h['cash']:.2f}; "
                     f"view: {h['market_view']}")
    return "\n".join(lines)


def build_user_message(st: TrajState, view_upto: pl.DataFrame, day: date, day_index: int,
                       news: dict[str, list[str]], market_news: list[str], lookback: int = 60) -> str:
    tables = lookback_table(view_upto, day, lookback)
    close = {k: float(g["close"][-1]) for k, g in tables.items()}
    parts = [portfolio_block(st, close, day_index), history_block(st, day_index)]
    if st.memo is not None:
        parts.append(f"YOUR STRATEGY MEMO (revision {st.memo_round})\n{st.memo}")
    parts.append(f"MARKET DATA (last {lookback} trading days, oldest to newest; the last value is D0)")
    parts.append(market_block(tables, st.labels))
    parts.append(news_block(news, market_news, st.labels))
    parts.append("Return your decision as JSON.")
    return "\n\n".join(parts)


def state_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
