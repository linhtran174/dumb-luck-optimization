"""CLI subcommands, grouped by milestone."""
from __future__ import annotations

import asyncio
import json

import typer

from dlo.cli import app

snap_app = typer.Typer(help="M2 frozen market snapshot")
app.add_typer(snap_app, name="snapshot")


@snap_app.command("fetch")
def snapshot_fetch() -> None:
    """Fetch bars and news once from Alpaca into data/raw/snapshot (refuses to overwrite)."""
    from dlo.clients.alpaca import AlpacaClient
    from dlo.data import snapshot

    if (snapshot.SNAP_DIR / "manifest.json").exists():
        raise typer.BadParameter("snapshot already frozen; delete data/raw/snapshot to refetch deliberately")

    async def run():
        c = AlpacaClient()
        try:
            return await snapshot.fetch(c)
        finally:
            await c.aclose()

    typer.echo(json.dumps(asyncio.run(run()), indent=2))


@snap_app.command("validate")
def snapshot_validate() -> None:
    from dlo.data.snapshot import Snapshot, verify_hashes
    from dlo.data.validate import validate

    typer.echo(json.dumps({"hashes": verify_hashes()}, indent=2))
    rep = validate(Snapshot())
    bad = {k: v for k, v in rep["symbols"].items() if v["missing_sessions"] or v["bad_ohlc"]}
    typer.echo(json.dumps({"ok": rep["ok"], "news_items": rep["news_items"], "news_range": rep["news_range"],
                           "n_symbols": len(rep["symbols"]), "problems": bad}, indent=2))


anon_app = typer.Typer(help="M4 anonymizer")
app.add_typer(anon_app, name="anon")


def _client():
    from dlo.clients.openrouter_chat import OpenRouterClient
    from dlo.store import LLMLedger
    return OpenRouterClient(LLMLedger())


@anon_app.command("build-aliases")
def build_aliases() -> None:
    from dlo.anon.alias_build import build_all
    from dlo.data.snapshot import Snapshot

    async def run():
        c = _client()
        try:
            return await build_all(c, Snapshot())
        finally:
            await c.aclose()

    for d in asyncio.run(run()):
        typer.echo(f"{d['ticker']}: " + ", ".join(f"{k}={len(v)}" for k, v in d["terms"].items()))


@anon_app.command("scrub")
def scrub(window: str = typer.Argument(..., help="trial | dev | test | all"),
          concurrency: int = 16) -> None:
    """Scrub all selected news for a window into the run store (idempotent)."""
    from dlo import config
    from dlo.anon.pipeline import scrub_days
    from dlo.data import calendar
    from dlo.data.snapshot import Snapshot
    from dlo.store import RunStore

    names = ["dev", "test"] if window == "all" else [window]
    days = []
    for n in names:
        a, b = config.window(n)
        days += calendar.sessions(a, b)

    async def run():
        c = _client()
        try:
            return await scrub_days(c, RunStore(), Snapshot(), days, concurrency)
        finally:
            await c.aclose()

    typer.echo(json.dumps(dict(asyncio.run(run()))))


synth_app = typer.Typer(help="M3 synthetic assets")
app.add_typer(synth_app, name="synth")


@synth_app.command("pregen")
def synth_pregen() -> None:
    """Pre-generate all synthetic shocks/events/candles for both windows plus lookbacks, with hashes."""
    from dlo.synth import pregenerate
    out = pregenerate.run()
    typer.echo(json.dumps(out, indent=1, default=str))


@synth_app.command("news")
def synth_news(window: str = typer.Argument("all", help="trial | dev | test | all"), concurrency: int = 16) -> None:
    """Generate (once) the synthetic news text for every synthetic asset-day of a window."""
    import polars as pl
    from dlo import config
    from dlo.data import calendar
    from dlo.store import RunStore
    from dlo.synth import pregenerate
    from dlo.synth.news_gen import generate_day

    synth, _ = pregenerate.load()
    names = ["dev", "test"] if window == "all" else [window]
    days = set()
    for n in names:
        a, b = config.window(n)
        days |= set(calendar.sessions(a, b))
    rows = synth.filter(pl.col("session").is_in(sorted(days)) & (pl.col("news_count") > 0)).to_dicts()
    store = RunStore()

    async def run():
        c = _client()
        sem = asyncio.Semaphore(concurrency)

        async def one(r):
            async with sem:
                z = r["sentiment"] if r["sentiment"] == r["sentiment"] else None
                return await generate_day(c, store, r["asset"], r["sector"], r["session"].isoformat(),
                                          int(r["news_count"]), bool(r["event"]), r["event_type"], z)
        try:
            out = await asyncio.gather(*(one(r) for r in rows))
        finally:
            await c.aclose()
        return sum(len(x) for x in out)

    typer.echo(json.dumps({"asset_days": len(rows), "items_ok": asyncio.run(run())}))


gate_app = typer.Typer(help="Validation gates")
app.add_typer(gate_app, name="gate")


def _write_gate(name: str, res: dict) -> None:
    from dlo import config
    d = config.DATA_DIR / "derived" / "gates"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))


@gate_app.command("g1")
def gate_g1() -> None:
    import polars as pl
    from dlo.analysis.gates import g1
    from dlo.data.snapshot import Snapshot
    from dlo.synth.pregenerate import PREGEN_DIR
    res = g1(Snapshot().bars, pl.read_parquet(PREGEN_DIR / "real_news_counts.parquet"))
    _write_gate("G1", res)
    typer.echo(json.dumps({k: res[k] for k in ("gate", "auc", "pass", "n_real", "n_synth")}) + "\n" +
               json.dumps(dict(list(res["univariate_auc"].items())[:8])))


@gate_app.command("g4")
def gate_g4() -> None:
    from dlo.analysis.gates import g4
    from dlo.data.snapshot import Snapshot
    res = g4(Snapshot().bars)
    _write_gate("G4", res)
    typer.echo(json.dumps(res, indent=1))


@gate_app.command("g0")
def gate_g0() -> None:
    """Post-cutoff knowledge check for both evaluated models (80 small calls)."""
    from dlo.analysis import g0
    from dlo.data.snapshot import Snapshot

    async def run():
        c = _client()
        try:
            return await g0.run(c, Snapshot().bars, ["terra", "luna"])
        finally:
            await c.aclose()

    res = asyncio.run(run())
    _write_gate("G0", res)
    typer.echo(json.dumps({"pass": res["pass"], **{m: {k: v for k, v in r.items() if k != "answers"}
                                                   for m, r in res["models"].items()}}, indent=1))


run_app = typer.Typer(help="M7 replay runner")
app.add_typer(run_app, name="run")


@run_app.command("window")
def run_window(window: str = typer.Argument(..., help="trial | dev | test"),
               run_id: str = typer.Option(None, help="defaults to <window>-v1; reuse it to resume"),
               models: str = typer.Option("terra,luna"), probes: bool = True,
               max_days: int = typer.Option(None), i_have_frozen_the_preregistration: bool = False) -> None:
    """Replay a window (resumable). The test window refuses to start without an explicit freeze flag."""
    from dlo import config
    from dlo.data.snapshot import Snapshot, verify_hashes
    from dlo.orchestrator.replay import Replay, RunConfig, default_trajectories
    from dlo.store import RunStore
    from dlo.synth import pregenerate

    if window == "test" and not i_have_frozen_the_preregistration:
        raise typer.BadParameter("the test window runs exactly once, after the pre-registration freeze; "
                                 "pass --i-have-frozen-the-preregistration to start it")
    if not all(verify_hashes().values()):
        raise typer.BadParameter("snapshot hashes do not match the manifest")
    revise_every = config.arms()["revise_every_days"]
    if window == "trial":
        revise_every = config.arms()["trial_overrides"]["revise_every_days"]
    budget = {"trial": "trial", "dev": "dev", "test": "test"}[window]
    synth, specs = pregenerate.load()
    cfg = RunConfig(run_id=run_id or f"{window}-v1", window=window, budget=budget,
                    trajectories=default_trajectories(models.split(",")), revise_every=revise_every,
                    probes=probes, max_days=max_days)

    async def run():
        c = _client()
        try:
            return await Replay(cfg, c, RunStore(), Snapshot(), synth, specs).run()
        finally:
            await c.aclose()

    typer.echo(json.dumps(asyncio.run(run())))


analyze_app = typer.Typer(help="M9 analysis")
app.add_typer(analyze_app, name="analyze")


@analyze_app.command("report")
def analyze_report(run_id: str, judge: bool = True) -> None:
    from datetime import date as _date
    from dlo.analysis import metrics, report
    from dlo.data.snapshot import Snapshot
    from dlo.data.universe_view import UniverseView
    from dlo.store import RunStore
    from dlo.synth import pregenerate

    store = RunStore()
    synth, specs = pregenerate.load()
    snap = Snapshot()
    cfg = json.loads(store.query("SELECT config_json FROM runs WHERE run_id=?", (run_id,))[0]["config_json"])
    view = UniverseView(snap, synth, specs, window_start=_date.fromisoformat(cfg["days"][0]))
    rd = metrics.RunData(store, run_id, view, synth, specs)
    rd.view_snap = snap

    async def run():
        c = _client() if judge else None
        try:
            return await report.build(store, run_id, rd, c)
        finally:
            if c:
                await c.aclose()

    asyncio.run(run())
    typer.echo((report.REPORT_DIR / f"{run_id.replace('/', '_')}.md").read_text())
