"""Scrub every selected news item for a list of simulated days (idempotent, resumable, bounded concurrency)."""
from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date

from dlo import logs
from dlo.anon.scrub import scrub_item
from dlo.anon.select import select_for_day
from dlo.clients.openrouter_chat import LLMClient
from dlo.data.snapshot import Snapshot
from dlo.store import RunStore

log = logs.get("anon")


async def scrub_days(client: LLMClient, store: RunStore, snap: Snapshot, days: list[date],
                     concurrency: int = 16) -> Counter:
    sem = asyncio.Semaphore(concurrency)
    stats: Counter = Counter()

    async def one(item: dict, subject: str | None, kind: str, day: date):
        async with sem:
            row = await scrub_item(client, store, item, subject, kind, day.isoformat())
            stats[row["status"]] += 1

    jobs = []
    for d in days:
        sel = select_for_day(snap, d)
        for t, items in sel["assets"].items():
            jobs += [one(it, t, "asset", d) for it in items]
        jobs += [one(it, None, "market", d) for it in sel["market"]]
    for i in range(0, len(jobs), 200):
        await asyncio.gather(*jobs[i:i + 200])
        logs.event(log, "scrub_progress", done=min(i + 200, len(jobs)), total=len(jobs), **stats)
    return stats


def scrubbed_for_day(store: RunStore, snap: Snapshot, day: date) -> dict:
    """Scrubbed, accepted texts for a day, in selection order. Missing items are reported, never fetched live."""
    sel = select_for_day(snap, day)
    out: dict = {"assets": {}, "market": [], "missing": 0}

    def get(kind: str, subject: str | None, item: dict) -> str | None:
        nid = f"{kind}:{subject or 'MKT'}:{item['id']}"
        r = store.query("SELECT status, text_tokenized FROM news_scrubbed WHERE news_id=?", (nid,))
        if not r:
            out["missing"] += 1
            return None
        return r[0]["text_tokenized"] if r[0]["status"] == "ok" else None

    for t, items in sel["assets"].items():
        out["assets"][t] = [x for x in (get("asset", t, it) for it in items) if x]
    out["market"] = [x for x in (get("market", None, it) for it in sel["market"]) if x]
    return out
