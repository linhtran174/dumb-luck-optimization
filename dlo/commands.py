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
