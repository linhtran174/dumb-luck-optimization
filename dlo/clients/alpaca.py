"""Alpaca Market Data v2 (bars) and News API client. Read-only; no brokerage endpoints exist here."""
from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import Any, Protocol

import httpx

from dlo import config, logs

log = logs.get("alpaca")
BASE = "https://data.alpaca.markets"

# Alpaca's News API tags Berkshire as BRK (sometimes also BRK/B); bars use BRK.B.
NEWS_SYMBOL = {"BRK.B": "BRK"}
NEWS_ALIASES = {"BRK.B": {"BRK", "BRK/B", "BRK.B", "BRK.A", "BRK/A"}}


class MarketDataSource(Protocol):
    async def bars(self, symbols: list[str], start: date, end: date) -> list[dict[str, Any]]: ...
    async def news(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]: ...


class _RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class AlpacaClient:
    def __init__(self, per_minute: int = 180):
        self.http = httpx.AsyncClient(
            base_url=BASE, timeout=60.0,
            headers={"APCA-API-KEY-ID": config.secret("ALPACA_API_KEY_ID"),
                     "APCA-API-SECRET-KEY": config.secret("ALPACA_API_SECRET_KEY")},
        )
        self.rl = _RateLimiter(per_minute)

    async def aclose(self) -> None:
        await self.http.aclose()

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(8):
            await self.rl.wait()
            try:
                r = await self.http.get(path, params=params)
            except httpx.HTTPError as e:
                log.warning(f"alpaca transport error {e}; retry {attempt}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"alpaca {path} {r.status_code}: {r.text[:300]}")
        raise RuntimeError(f"alpaca {path}: retries exhausted")

    async def bars(self, symbols: list[str], start: date, end: date) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        token = None
        while True:
            params = {"symbols": ",".join(symbols), "timeframe": "1Day", "start": start.isoformat(),
                      "end": end.isoformat(), "feed": "sip", "adjustment": "all", "limit": 10000}
            if token:
                params["page_token"] = token
            d = await self._get("/v2/stocks/bars", params)
            for sym, rows in (d.get("bars") or {}).items():
                for b in rows:
                    out.append({"symbol": sym, **b})
            token = d.get("next_page_token")
            if not token:
                return out

    async def news(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        token = None
        while True:
            params = {"symbols": symbol, "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "sort": "asc",
                      "include_content": "false", "exclude_contentless": "false"}
            if token:
                params["page_token"] = token
            d = await self._get("/v1beta1/news", params)
            out.extend(d.get("news") or [])
            token = d.get("next_page_token")
            if not token:
                return out
