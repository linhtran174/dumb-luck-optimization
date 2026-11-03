"""M6: hand-computed P&L on a 5-day toy case must match exactly."""
import pytest

from dlo.agents.state import TrajState
from dlo.sim.execution import charge_compute, equity, execute


def st():
    return TrajState(traj_id="t", arm="a", model_key="m", seed=1, labels={"A": "ASSET_01", "B": "ASSET_02"},
                     cash=1000.0)


def test_toy_five_days():
    s = st()
    # Day 1 close: decide A 0.5, B 0.3, cash 0.2; decision call costs 0.10
    charge_compute(s, 0.10)
    assert s.cash == pytest.approx(999.90, abs=1e-12)
    # Day 2 open: A=10, B=20
    f = execute(s, {"A": 0.5, "B": 0.3}, {"A": 10.0, "B": 20.0}, bps=5.0)
    assert f.traded_value == pytest.approx(799.92, abs=1e-9)
    assert f.cost == pytest.approx(0.39996, abs=1e-12)
    assert s.cash == pytest.approx(199.58004, abs=1e-9)
    assert s.shares["A"] == pytest.approx(49.995, abs=1e-12)
    assert s.shares["B"] == pytest.approx(14.9985, abs=1e-12)
    assert equity(s, {"A": 11.0, "B": 19.0}) == pytest.approx(1034.49654, abs=1e-9)
    # Day 2: agent sleeps -> day 3 open, no trade
    assert execute(s, None, {"A": 11.5, "B": 18.0}, bps=5.0) is None
    assert equity(s, {"A": 12.0, "B": 18.5}) == pytest.approx(1076.99229, abs=1e-9)
    # Day 3 close: decide A 0, B 0.3, cash 0.7; cost 0.10
    charge_compute(s, 0.10)
    # Day 4 open: A=12.5, B=19
    f = execute(s, {"B": 0.3}, {"A": 12.5, "B": 19.0}, bps=5.0)
    assert f.traded_value == pytest.approx(672.782712, abs=1e-9)
    assert f.cost == pytest.approx(0.336391356, abs=1e-12)
    assert s.cash == pytest.approx(776.235936644, abs=1e-9)
    assert "A" not in s.shares
    # Day 5 close: B=20
    assert equity(s, {"B": 20.0}) == pytest.approx(1126.569317696632, abs=1e-9)
    assert s.compute_cost_cum == pytest.approx(0.20, abs=1e-12)
    assert s.trading_cost_cum == pytest.approx(0.39996 + 0.336391356, abs=1e-12)


def test_forced_sale_keeps_reserve():
    s = st()
    s.cash, s.shares = 0.05, {"A": 100.0}
    f = execute(s, None, {"A": 10.0}, bps=5.0, reserve=0.12)
    assert f is not None and f.forced
    assert s.cash == pytest.approx(0.12, abs=1e-9)


def test_target_cash_below_reserve_scales_risky():
    s = st()
    f = execute(s, {"A": 0.3, "B": 0.3}, {"A": 10.0, "B": 10.0}, bps=5.0, reserve=500.0)
    assert f.forced and s.cash == pytest.approx(500.0, abs=1e-6)
