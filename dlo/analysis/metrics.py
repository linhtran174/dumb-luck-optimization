"""Run-level metrics for RQ1-RQ5 plus gate checks G2, G3, G5 and the cost check, from persisted run data."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import polars as pl
from scipy import stats

from dlo import config
from dlo.agents.state import TrajState
from dlo.data import calendar
from dlo.data.universe_view import UniverseView
from dlo.sim import baselines, random_agents
from dlo.store import RunStore


def _bootstrap_mean_ci(x: np.ndarray, block: int = 5, reps: int = 2000, seed: int = 0) -> list[float] | None:
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return None
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block, x, seed=seed)
    means = np.array([d[0][0].mean() for d in bs.bootstrap(reps)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


class RunData:
    def __init__(self, store: RunStore, run_id: str, view: UniverseView, synth: pl.DataFrame, specs: list[dict]):
        self.store, self.run_id, self.view, self.synth = store, run_id, view, synth
        self.specs = {s["key"]: s for s in specs}
        cfg = json.loads(store.query("SELECT config_json FROM runs WHERE run_id=?", (run_id,))[0]["config_json"])
        a, b = cfg["days"]
        self.days = calendar.sessions(pl.Series([a]).str.to_date()[0], pl.Series([b]).str.to_date()[0])
        self.day_str = [d.isoformat() for d in self.days]
        ck = store.query("SELECT trajectory_id, state_json FROM checkpoints WHERE run_id=? ORDER BY sim_day DESC",
                         (run_id,))
        self.states: dict[str, TrajState] = {}
        for r in ck:
            if r["trajectory_id"] not in self.states:
                self.states[r["trajectory_id"]] = TrajState.from_json(json.loads(r["state_json"]))
        self.decisions = [dict(r) for r in store.query("SELECT * FROM decisions WHERE run_id=? ORDER BY sim_day",
                                                       (run_id,))]
        self.equity = [dict(r) for r in store.query("SELECT * FROM equity WHERE run_id=? ORDER BY sim_day", (run_id,))]
        panel = view.full_for_simulator()
        self.open = {(k, d): o for k, d, o in panel.select("key", "session", "open").iter_rows()}
        self.close = {(k, d): c for k, d, c in panel.select("key", "session", "close").iter_rows()}
        self.signal = {(a_, d): s for a_, d, s in synth.select("asset", "session", "signal").iter_rows()}

    def fwd(self, key: str, day, h: int) -> float:
        """Open-to-open return from the next session's open over h sessions."""
        try:
            d1, d2 = calendar.next_session(day, 1), calendar.next_session(day, 1 + h)
            return self.open[(key, d2)] / self.open[(key, d1)] - 1
        except (KeyError, IndexError):
            return np.nan


def ic_metrics(rd: RunData) -> dict:
    real = config.real_tickers()
    out: dict = defaultdict(lambda: defaultdict(list))
    for dec in rd.decisions:
        if not dec["decision_json"]:
            continue
        st = rd.states[dec["trajectory_id"]]
        l2k = st.label_to_key()
        d = json.loads(dec["decision_json"])
        day = pl.Series([dec["sim_day"]]).str.to_date()[0]
        sc = {l2k[lab]: v for lab, v in d["scores"].items()}
        for h, name in ((1, "ic1"), (5, "ic5")):
            xs = [sc[k] for k in real]
            ys = [rd.fwd(k, day, h) for k in real]
            m = np.isfinite(ys)
            if m.sum() >= 5 and np.std(np.array(xs)[m]) > 0:
                out[dec["trajectory_id"]][name].append(stats.spearmanr(np.array(xs)[m], np.array(ys)[m])[0])
        ys1 = {k: rd.fwd(k, day, 1) for k in real}
        if all(np.isfinite(v) for v in ys1.values()):
            order = sorted(real, key=lambda k: sc[k])
            out[dec["trajectory_id"]]["spread"].append(np.mean([ys1[k] for k in order[-3:]]) -
                                                       np.mean([ys1[k] for k in order[:3]]))
        # RQ2 detection: score at t vs the planted signal that drives eps at t+1
        nxt = calendar.next_session(day, 1)
        for key, spec in rd.specs.items():
            sig = rd.signal.get((key, nxt))
            if spec["group"] != "noise" and sig is not None and np.isfinite(sig):
                out[dec["trajectory_id"]][f"det:{spec['group']}:{spec['strength']}"].append((sc[key], sig))
    res = {}
    for t, m in out.items():
        r = {}
        for name in ("ic1", "ic5", "spread"):
            x = np.array(m.get(name, []), float)
            r[name] = {"mean": float(np.nanmean(x)) if len(x) else None, "n": int(len(x)),
                       "ci95": _bootstrap_mean_ci(x)}
        for name, pairs in m.items():
            if name.startswith("det:"):
                a = np.array(pairs, float)
                ok = len(a) >= 3 and np.std(a[:, 0]) > 0 and np.std(a[:, 1]) > 0
                r[name] = {"corr": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if ok else None, "n": len(a)}
        res[t] = r
    return res


def portfolio_metrics(rd: RunData, ew_final: float | None) -> dict:
    cap = config.costs()["capital_usd"]
    by_t = defaultdict(list)
    for e in rd.equity:
        by_t[e["trajectory_id"]].append(e)
    res = {}
    for t, rows in by_t.items():
        net = np.array([r["net_equity"] for r in rows])
        gross = np.array([r["gross_equity"] for r in rows])
        rets = np.diff(np.concatenate([[cap], net])) / np.concatenate([[cap], net[:-1]])
        decs = [d for d in rd.decisions if d["trajectory_id"] == t]
        cost = rows[-1]["compute_cost_cum"]
        r_gross = gross[-1] / cap - 1
        r_ew = (ew_final / cap - 1) if ew_final else None
        kstar = (cost / (r_gross - r_ew)) if (r_ew is not None and r_gross - r_ew > 0) else None
        st = rd.states[t]
        toks = [d for d in decs]
        res[t] = {"final_net_equity": float(net[-1]), "final_gross_equity": float(gross[-1]),
                  "net_return": float(net[-1] / cap - 1), "gross_return": float(r_gross),
                  "compute_cost": float(cost), "compute_drag": float(cost / cap),
                  "trading_cost": float(rows[-1]["trading_cost_cum"]),
                  "net_sharpe_ann": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else None,
                  "break_even_capital": kstar if kstar is not None else "never breaks even",
                  "sleep_share": 1 - len(decs) / max(1, len(rows)), "decisions": len(toks),
                  "ruined": bool(st.ruined), "ruin_day": st.ruin_day,
                  "capital_sweep": {str(k): float(k * (1 + r_gross) - cost) for k in (250, 500, 1000, 5000)}}
    return res


def behavior(rd: RunData) -> dict:
    """RQ3 action-coded taxonomy (thresholds as in the plan)."""
    keys = list(rd.view.keys)
    ew = np.full(len(keys), 1 / len(keys))
    by_t = defaultdict(list)
    for e in rd.equity:
        by_t[e["trajectory_id"]].append(e)
    decs = defaultdict(dict)
    for d in rd.decisions:
        if d["decision_json"]:
            decs[d["trajectory_id"]][d["sim_day"]] = json.loads(d["decision_json"])
    res = {}
    for t, rows in by_t.items():
        st = rd.states[t]
        l2k = st.label_to_key()
        W = []
        for r in rows:
            w = json.loads(r["weights_json"])
            W.append(np.array([w.get(k, 0.0) for k in keys]))
        W = np.array(W)
        net = np.array([r["net_equity"] for r in rows])
        peak = np.maximum.accumulate(net)
        dd = (peak - net) / peak >= 0.03
        cash = 1 - W.sum(1)
        tv = 0.5 * np.abs(W - ew).sum(1)
        turnover = np.concatenate([[0], 0.5 * np.abs(np.diff(W, axis=0)).sum(1)])
        sleeps = {r["sim_day"]: decs[t].get(r["sim_day"], {}).get("next_review_in_days") for r in rows}
        retreat = []
        for i in np.where(dd)[0]:
            win = range(i, min(len(rows), i + 6))
            retreat.append(any(cash[j] >= 0.7 or (sleeps.get(rows[j]["sim_day"]) or 0) >= 3 for j in win))
        mom, gam, dbl = [], [], []
        for i in range(1, len(rows)):
            day = pl.Series([rows[i]["sim_day"]]).str.to_date()[0]
            r5, down3 = [], []
            for k in keys:
                try:
                    prev5 = calendar.sessions_before(day, 5)[0]
                    r5.append(rd.close[(k, day)] / rd.close[(k, prev5)] - 1)
                    seq = calendar.sessions_before(day, 3) + [day]
                    down3.append(all(rd.close[(k, b)] < rd.close[(k, a)] for a, b in zip(seq, seq[1:])))
                except KeyError:
                    r5.append(np.nan)
                    down3.append(False)
            dw = W[i] - W[i - 1]
            r5 = np.array(r5)
            m = np.isfinite(r5)
            if np.std(dw[m]) > 0 and np.std(r5[m]) > 0:
                mom.append(float(np.corrcoef(dw[m], r5[m])[0, 1]))
            if any(down3):
                gam.append(bool(np.any(dw[np.array(down3)] > 1e-6)))
            if dd[i - 1]:
                dbl.append(bool(W[i].sum() > W[i - 1].sum() + 1e-6))
        res[t] = {"retreat_rate_after_drawdown": float(np.mean(retreat)) if retreat else None,
                  "index_reversion_share": float(np.mean(tv <= 0.1)),
                  "momentum_chasing_corr": float(np.mean(mom)) if mom else None,
                  "gamblers_fallacy_rate": float(np.mean(gam)) if gam else None,
                  "doubling_down_rate": float(np.mean(dbl)) if dbl else None,
                  "mean_turnover": float(turnover.mean()), "mean_cash": float(cash.mean()),
                  "drawdown_days": int(dd.sum()), "noise_asset_mean_weight":
                      float(np.mean([W[:, keys.index(k)].mean() for k, s in rd.specs.items() if s["group"] == "noise"]))}
        _ = l2k
    return res


def gate_checks(rd: RunData) -> dict:
    dec = rd.decisions
    n = len(dec)
    first = sum(d["status"] == "valid_first" for d in dec)
    by_model = defaultdict(lambda: [0, 0])
    for d in dec:
        m = rd.states[d["trajectory_id"]].model_key
        by_model[m][0] += d["status"] == "valid_first"
        by_model[m][1] += 1
    probes = [dict(r) for r in rd.store.query("SELECT * FROM probes WHERE run_id=?", (rd.run_id,))]
    g2, g3 = {}, {}
    for m in sorted({p["model_key"] for p in probes}):
        ps = [p for p in probes if p["model_key"] == m and not str(p["answer"]).startswith("error")]
        acc = np.mean([(p["answer_synthetic"] == "synthetic") == bool(p["is_synthetic"]) for p in ps]) if ps else None
        real_ps = [p for p in ps if not p["is_synthetic"]]
        reid = np.mean([p["answer"] == p["truth"] for p in real_ps]) if real_ps else None
        g2[m] = {"real_vs_synth_accuracy": acc, "n": len(ps), "pass": acc is not None and abs(acc - 0.5) <= 0.10}
        g3[m] = {"reidentification_rate": reid, "n": len(real_ps), "pass": reid is not None and reid <= 0.20}
    from dlo.anon import aliases
    hits = 0
    from dlo.anon.scrub import SCRUB_VERSION
    for r in rd.store.query("SELECT text_tokenized FROM news_scrubbed WHERE status='ok' AND news_id LIKE ?",
                            (f"{SCRUB_VERSION}:%",)):
        hits += bool(aliases.leak_hits(r["text_tokenized"]))
    sizes = [d["state_bytes"] for d in dec]
    return {"G5": {"valid_first_share": first / n if n else None, "n": n, "pass": n > 0 and first / n >= 0.98,
                   "by_model": {m: v[0] / v[1] for m, v in by_model.items()}},
            "state_bytes": {"max": max(sizes) if sizes else None, "mean": float(np.mean(sizes)) if sizes else None,
                            "all_under_30kb": all(s <= 30 * 1024 for s in sizes)},
            "G2": g2, "G3": {"regex_hits_in_accepted_text": hits, "probe": g3,
                             "pass": hits == 0 and all(v["pass"] for v in g3.values()) if g3 else hits == 0}}


def comparators(rd: RunData, news_counts: dict) -> dict:
    keys = list(rd.view.keys)
    base = baselines.run_baselines(rd.view, rd.days, keys, news_counts)
    opens = np.array([[rd.open[(k, d)] for k in keys] for d in rd.days])
    closes = np.array([[rd.close[(k, d)] for k in keys] for d in rd.days])
    sl = [json.loads(d["decision_json"])["next_review_in_days"] for d in rd.decisions if d["decision_json"]]
    vals, cnt = np.unique(sl, return_counts=True) if sl else (np.array([1]), np.array([1]))
    eq = random_agents.simulate(opens, closes, vals, cnt / cnt.sum(), n_agents=1000, seed=1)
    return {"baselines": {k: v[-1]["net_equity"] for k, v in base.items()},
            "random_final": eq[:, -1], "sleep_dist": dict(zip(map(int, vals), map(int, cnt)))}
