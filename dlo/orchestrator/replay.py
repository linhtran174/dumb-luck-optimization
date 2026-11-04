"""Day-by-day replay with checkpoint/resume.

Per simulated day i:
  1. Open: apply each trajectory's pending targets at the open (or the cash-reserve check); mark to close.
  2. News: scrubbed real items (pre-scrubbed; scrubbed on the fly if missing) and synthetic items for the day.
  3. Decide: every due, non-ruined trajectory gets one decision call (bounded concurrency); costs debited.
  4. Revise (REVISE arms, every N days) and weekly probe (every 5 days).
  5. Checkpoint every trajectory and mark the day done. A rerun resumes after the last done day, and every LLM
     request it repeats is answered from the persisted ledger, so nothing is paid twice.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl

from dlo import config, logs
from dlo.agents import revise
from dlo.agents.llm_agent import decide
from dlo.agents.state import MAX_STATE_BYTES, TrajState, build_user_message, permutation_labels, state_hash
from dlo.anon.pipeline import scrubbed_for_day
from dlo.anon.probe import run_probe
from dlo.anon.scrub import SCRUB_VERSION, scrub_item
from dlo.anon.select import select_for_day
from dlo.clients.openrouter_chat import LLMClient
from dlo.data import calendar
from dlo.data.snapshot import Snapshot
from dlo.data.universe_view import UniverseView, asset_keys
from dlo.sim.execution import charge_compute, check_ruin, equity, execute, expected_call_cost
from dlo.store import RunStore, canonical_json, now_utc
from dlo.synth.news_gen import generate_day, stored_texts

log = logs.get("replay")
DEFAULT_CALL_COST = {"terra": 0.12, "luna": 0.012}


@dataclass
class TrajSpec:
    arm: str
    model_key: str
    seed: int

    @property
    def traj_id(self) -> str:
        return f"{self.arm}/{self.model_key}/s{self.seed}"


@dataclass
class RunConfig:
    run_id: str
    window: str
    budget: str
    trajectories: list[TrajSpec]
    revise_every: int
    probe_every: int = 5
    probes: bool = True
    probe_models: list[str] = field(default_factory=lambda: ["terra", "luna"])
    max_days: int | None = None


def default_trajectories(models: list[str] | None = None) -> list[TrajSpec]:
    out = []
    for arm, a in config.arms()["arms"].items():
        for m in models or a["models"]:
            for s in range(1, a["seeds"] + 1):
                out.append(TrajSpec(arm, m, s))
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True).strip()
    except Exception:
        return "unknown"


class Replay:
    def __init__(self, cfg: RunConfig, client: LLMClient, store: RunStore, snap: Snapshot, synth: pl.DataFrame,
                 specs: list[dict]):
        self.cfg, self.client, self.store, self.snap = cfg, client, store, snap
        a, b = config.window(cfg.window)
        self.days = calendar.sessions(a, b)
        if cfg.max_days:
            self.days = self.days[:cfg.max_days]
        self.view = UniverseView(snap, synth, specs, window_start=self.days[0])
        self.keys = asset_keys(specs)
        self.synth = synth
        self.specs = {s["key"]: s for s in specs}
        self.bps = config.costs()["trading_cost_bps"]
        self.capital = config.costs()["capital_usd"]

    # ---------- state ----------
    def _init_states(self) -> dict[str, TrajState]:
        out = {}
        for t in self.cfg.trajectories:
            memo = "" if config.arms()["arms"][t.arm]["memo"] != "none" else None
            out[t.traj_id] = TrajState(traj_id=t.traj_id, arm=t.arm, model_key=t.model_key, seed=t.seed,
                                       labels=permutation_labels(self.keys, t.arm, t.seed), cash=self.capital,
                                       memo=memo)
        return out

    def _resume(self) -> tuple[int, dict[str, TrajState]]:
        done = {r["sim_day"] for r in self.store.query("SELECT sim_day FROM day_done WHERE run_id=?", (self.cfg.run_id,))}
        last = -1
        for i, d in enumerate(self.days):
            if d.isoformat() in done:
                last = i
            else:
                break
        if last < 0:
            return 0, self._init_states()
        rows = self.store.query("SELECT trajectory_id, state_json FROM checkpoints WHERE run_id=? AND sim_day=?",
                                (self.cfg.run_id, self.days[last].isoformat()))
        states = {r["trajectory_id"]: TrajState.from_json(json.loads(r["state_json"])) for r in rows}
        logs.event(log, "resume", run=self.cfg.run_id, after_day=str(self.days[last]), trajectories=len(states))
        return last + 1, states

    # ---------- news ----------
    async def _news(self, day: date) -> tuple[dict[str, list[str]], list[str]]:
        sc = scrubbed_for_day(self.store, self.snap, day)
        if sc["missing"]:
            sel = select_for_day(self.snap, day)
            jobs = [scrub_item(self.client, self.store, it, t, "asset", day.isoformat())
                    for t, items in sel["assets"].items() for it in items]
            jobs += [scrub_item(self.client, self.store, it, None, "market", day.isoformat()) for it in sel["market"]]
            await asyncio.gather(*jobs)
            sc = scrubbed_for_day(self.store, self.snap, day)
        news = dict(sc["assets"])
        rows = self.synth.filter(pl.col("session") == day).to_dicts()

        async def syn(r):
            z = None if r["sentiment"] is None or (isinstance(r["sentiment"], float) and np.isnan(r["sentiment"])) \
                else float(r["sentiment"])
            return r["asset"], await generate_day(self.client, self.store, r["asset"], r["sector"], day.isoformat(),
                                                  int(r["news_count"]), bool(r["event"]), r["event_type"], z)

        for key, texts in await asyncio.gather(*(syn(r) for r in rows)):
            news[key] = texts
        return news, sc["market"]

    # ---------- main loop ----------
    async def run(self) -> dict:
        cfg = self.cfg
        start, states = self._resume()
        if start == 0:
            self.store.put("runs", {"run_id": cfg.run_id, "window": cfg.window, "started_at": now_utc(),
                                    "finished_at": None, "status": "running", "git_commit": git_commit(),
                                    "config_json": canonical_json({"scrub_version": SCRUB_VERSION, "days": [str(self.days[0]), str(self.days[-1])],
                                                                   "trajectories": [t.traj_id for t in cfg.trajectories],
                                                                   "revise_every": cfg.revise_every,
                                                                   "budget": cfg.budget}),
                                    "notes": None})
        elif start >= len(self.days):
            logs.event(log, "run_already_complete", run=cfg.run_id)
            return {"run_id": cfg.run_id, "days": len(self.days), "already_complete": True}
        else:
            self.store.event(cfg.run_id, None, None, "resume", {"from_day": str(self.days[start])})
        panel = self.view.full_for_simulator()
        for i in range(start, len(self.days)):
            day = self.days[i]
            ds = day.isoformat()
            opens, closes = self.view.prices(day, "open"), self.view.prices(day, "close")

            # 1. fills at the open, then mark to close
            for st in states.values():
                if st.ruined:
                    continue
                reserve = expected_call_cost(st, DEFAULT_CALL_COST.get(st.model_key, 0.05))
                fill = execute(st, st.pending, opens, self.bps, reserve=reserve)
                st.pending = None
                if fill:
                    st.last_fill = {"day_offset": 0, "traded_value": fill.traded_value, "cost": fill.cost,
                                    "forced": fill.forced, "abs_day": i}
                    self.store.put("fills", {"run_id": cfg.run_id, "trajectory_id": st.traj_id, "sim_day": ds,
                                             "fill_json": canonical_json(fill.__dict__), "forced_sale": int(fill.forced),
                                             "created_at": now_utc()})
                    if fill.forced:
                        self.store.event(cfg.run_id, ds, st.traj_id, "forced_sale", fill.__dict__)
                elif st.last_fill:
                    st.last_fill["day_offset"] = st.last_fill.get("abs_day", i) - i
                check_ruin(st, closes, ds)

            # 2. news for the day
            news, market_news = await self._news(day)
            view_upto = self.view.upto(day)

            # 3. decisions
            due = [st for st in states.values() if not st.ruined and i >= st.next_review_index]
            await asyncio.gather(*(self._decide(st, view_upto, day, i, news, market_news, closes) for st in due))

            # 4a. revise loop
            if cfg.revise_every and (i + 1) % cfg.revise_every == 0:
                await asyncio.gather(*(self._revise(st, i, panel) for st in states.values()
                                       if st.memo is not None and not st.ruined))

            # 5. mark equity and checkpoint
            for st in states.values():
                pv = sum(sh * closes[k] for k, sh in st.shares.items())
                net = st.cash + pv
                w = {k: sh * closes[k] / net for k, sh in st.shares.items()} if net > 0 else {}
                check_ruin(st, closes, ds)
                self.store.put("equity", {"run_id": cfg.run_id, "trajectory_id": st.traj_id, "sim_day": ds,
                                          "cash": st.cash, "positions_value": pv, "net_equity": net,
                                          "gross_equity": net + st.compute_cost_cum,
                                          "compute_cost_cum": st.compute_cost_cum,
                                          "trading_cost_cum": st.trading_cost_cum,
                                          "weights_json": canonical_json(w), "ruined": int(st.ruined)})

            # 4b. weekly probe (infrastructure, never charged to trajectories)
            if cfg.probes and (i + 1) % cfg.probe_every == 0:
                await self._probe(i, day, view_upto)

            for st in states.values():
                self.store.put("checkpoints", {"run_id": cfg.run_id, "trajectory_id": st.traj_id, "sim_day": ds,
                                               "state_json": canonical_json(st.to_json()), "created_at": now_utc()})
            self.store.put("day_done", {"run_id": cfg.run_id, "sim_day": ds, "created_at": now_utc()})
            logs.event(log, "day_done", run=cfg.run_id, day=ds, index=i, due=len(due),
                       equity={st.traj_id: round(equity(st, closes), 2) for st in states.values()})
        self.store.execute("UPDATE runs SET finished_at=?, status='done' WHERE run_id=?", (now_utc(), cfg.run_id))
        return {"run_id": cfg.run_id, "days": len(self.days)}

    async def _decide(self, st: TrajState, view_upto, day: date, i: int, news, market_news, closes) -> None:
        ds = day.isoformat()
        msg = build_user_message(st, view_upto, day, i, news, market_news)
        nbytes = len(msg.encode())
        if nbytes > MAX_STATE_BYTES:
            self.store.event(self.cfg.run_id, ds, st.traj_id, "state_over_30kb", {"bytes": nbytes})
        res = await decide(self.client, st.model_key, msg, st.traj_id, ds, self.cfg.budget)
        charge_compute(st, res.cost)
        st.n_calls += res.attempts
        l2k = st.label_to_key()
        if res.decision is not None:
            d = res.decision
            st.pending = {l2k[lab]: w for lab, w in d.weights.items() if w > 0}
            st.next_review_index = i + d.next_review_in_days
            st.history = (st.history + [{"day_index": i, "weights": d.weights, "cash": d.cash,
                                         "market_view": d.market_view}])[-10:]
        else:
            st.pending = None          # hold previous holdings
            st.next_review_index = i + 1
        check_ruin(st, closes, ds)
        self.store.put("decisions", {
            "run_id": self.cfg.run_id, "trajectory_id": st.traj_id, "sim_day": ds, "day_index": i,
            "status": res.status, "n_attempts": res.attempts,
            "decision_json": canonical_json(res.decision.to_json()) if res.decision else None,
            "state_hash": state_hash(msg), "state_bytes": nbytes, "cost": res.cost,
            "call_ids": canonical_json(res.call_ids), "error": canonical_json(res.errors) if res.errors else None,
            "created_at": now_utc()})

    async def _revise(self, st: TrajState, i: int, panel: pl.DataFrame) -> None:
        n = self.cfg.revise_every
        day = self.days[i]
        rows = self.store.query("SELECT * FROM equity WHERE run_id=? AND trajectory_id=? ORDER BY sim_day",
                                (self.cfg.run_id, st.traj_id))
        past = [dict(r) for r in rows if r["sim_day"] < day.isoformat()]
        # Period = the last n trading days ending today; baseline = the equity row just before it.
        # Today's equity row is written after this call, so it is built from the live state.
        closes = self.view.prices(day, "close")
        pv = sum(sh * closes[k] for k, sh in st.shares.items())
        net = st.cash + pv
        today = {"sim_day": day.isoformat(), "net_equity": net, "compute_cost_cum": st.compute_cost_cum,
                 "weights_json": canonical_json({k: sh * closes[k] / net for k, sh in st.shares.items()})}
        if len(past) >= n:
            base, body = past[-n], past[-(n - 1):] if n > 1 else []
        else:
            base = {"sim_day": None, "net_equity": self.capital, "compute_cost_cum": 0.0, "weights_json": "{}"}
            body = past
        period = [base] + body + [today]
        close_map = {(k, d.isoformat() if hasattr(d, "isoformat") else d): c for k, d, c in
                     panel.select("key", "session", "close").iter_rows()}

        def prev_close(key: str, ds: str) -> float:
            prev = calendar.prev_session(date.fromisoformat(ds)).isoformat()
            return close_map[(key, prev)]

        rep = revise.build_report(period, close_map, st, prev_close)
        shuffled = config.arms()["arms"][st.arm]["memo"] == "shuffled"
        if shuffled:
            seed = int(hashlib.sha256(f"shuffle:{st.traj_id}:{st.memo_round}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed)
            rep = revise.shuffle_report(rep, rng)
        text = revise.render_report(rep)
        out, cost, call_ids, err = await revise.rewrite_memo(self.client, st, text, st.history, day.isoformat(),
                                                             self.cfg.budget)
        charge_compute(st, cost)
        st.n_calls += 1
        if out:
            st.memo_round += 1
            st.memo = out["memo"]
        self.store.put("memos", {"run_id": self.cfg.run_id, "trajectory_id": st.traj_id,
                                 "round": st.memo_round if out else -(i + 1), "sim_day": day.isoformat(),
                                 "memo": out["memo"] if out else None, "changes": out["changes"] if out else err,
                                 "report_json": canonical_json(rep), "shuffled": int(shuffled), "cost": cost,
                                 "call_ids": canonical_json(call_ids), "created_at": now_utc()})

    async def _probe(self, i: int, day: date, view_upto) -> None:
        week = (i + 1) // self.cfg.probe_every
        lo = max(0, i - self.cfg.probe_every + 1)
        news_week: dict[str, list[str]] = {}
        for d in self.days[lo:i + 1]:
            sc = scrubbed_for_day(self.store, self.snap, d)
            for k, v in sc["assets"].items():
                news_week.setdefault(k, []).extend(v)
            for k, v in stored_texts(self.store, d.isoformat()).items():
                news_week.setdefault(k, []).extend(v)
        await run_probe(self.client, self.store, self.cfg.run_id, week, day, view_upto, self.keys, news_week,
                        set(self.specs), self.cfg.probe_models)
