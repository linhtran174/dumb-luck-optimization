"""Persistent stores.

* `LLMLedger` (SQLite, WAL): every LLM request/response, verbatim, written before the caller sees the
  result. Successful responses double as a cache keyed by request hash, so no paid call is ever repeated.
* `RunStore` (SQLite): trajectory checkpoints, decisions, fills, memos, run log.
* Market snapshot lives in Parquet (see dlo.data.snapshot) and is queried with DuckDB / polars.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dlo.config import DATA_DIR


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class _SQLite:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def executescript(self, sql: str) -> None:
        with self._lock:
            self.conn.executescript(sql)

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
  call_id TEXT PRIMARY KEY,
  request_hash TEXT NOT NULL,
  purpose TEXT NOT NULL,          -- decision | decision_retry | memo | scrub | scrub_strict | entail | alias | synth_news | probe | g0 | judge | smoke
  budget TEXT NOT NULL,           -- spend cap bucket: trial | dev | test | infra
  charged INTEGER NOT NULL,       -- 1 if debited from a trajectory's paper cash
  trajectory_id TEXT,
  sim_day TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,
  model_key TEXT NOT NULL,
  model_slug TEXT NOT NULL,
  response_model TEXT,
  provider TEXT,
  request_json TEXT NOT NULL,
  response_json TEXT,
  http_status INTEGER,
  ok INTEGER NOT NULL,
  error TEXT,
  latency_s REAL,
  prompt_tokens INTEGER,
  completion_tokens INTEGER,
  reasoning_tokens INTEGER,
  cached_tokens INTEGER,
  cost REAL NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_calls_hash ON llm_calls(request_hash, ok);
CREATE INDEX IF NOT EXISTS ix_calls_budget ON llm_calls(budget);
CREATE INDEX IF NOT EXISTS ix_calls_traj ON llm_calls(trajectory_id, sim_day);
"""


class LLMLedger(_SQLite):
    def __init__(self, path: Path | None = None, jsonl: Path | None = None):
        super().__init__(path or DATA_DIR / "db" / "llm_calls.sqlite")
        self.executescript(LEDGER_SCHEMA)
        self.jsonl = jsonl if jsonl is not None else DATA_DIR / "logs" / "llm_calls.jsonl"
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def cached(self, request_hash: str) -> sqlite3.Row | None:
        rows = self.query(
            "SELECT * FROM llm_calls WHERE request_hash=? AND ok=1 ORDER BY created_at LIMIT 1", (request_hash,)
        )
        return rows[0] if rows else None

    def record(self, row: dict[str, Any]) -> str:
        row = dict(row)
        row.setdefault("call_id", str(uuid.uuid4()))
        row.setdefault("created_at", now_utc())
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        with self._lock:
            self.conn.execute(f"INSERT INTO llm_calls ({cols}) VALUES ({qs})", tuple(row.values()))
            with open(self.jsonl, "a") as f:
                f.write(canonical_json(row) + "\n")
                f.flush()
        return row["call_id"]

    def spent(self, budget: str) -> float:
        r = self.query("SELECT COALESCE(SUM(cost),0) AS c FROM llm_calls WHERE budget=?", (budget,))
        return float(r[0]["c"])

    def summary(self) -> list[dict]:
        rows = self.query(
            """SELECT budget, purpose, model_key, COUNT(*) n, SUM(ok) ok, ROUND(SUM(cost),4) cost,
                      SUM(prompt_tokens) pin, SUM(completion_tokens) pout, SUM(reasoning_tokens) preason
               FROM llm_calls GROUP BY budget, purpose, model_key ORDER BY budget, purpose, model_key"""
        )
        return [dict(r) for r in rows]


RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, window TEXT, started_at TEXT, finished_at TEXT, status TEXT,
  git_commit TEXT, config_json TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id TEXT, trajectory_id TEXT, sim_day TEXT, state_json TEXT, created_at TEXT,
  PRIMARY KEY (run_id, trajectory_id, sim_day)
);
CREATE TABLE IF NOT EXISTS day_done (
  run_id TEXT, sim_day TEXT, created_at TEXT, PRIMARY KEY (run_id, sim_day)
);
CREATE TABLE IF NOT EXISTS decisions (
  run_id TEXT, trajectory_id TEXT, sim_day TEXT, day_index INTEGER, status TEXT, n_attempts INTEGER,
  decision_json TEXT, state_hash TEXT, state_bytes INTEGER, cost REAL, call_ids TEXT, error TEXT,
  created_at TEXT, PRIMARY KEY (run_id, trajectory_id, sim_day)
);
CREATE TABLE IF NOT EXISTS fills (
  run_id TEXT, trajectory_id TEXT, sim_day TEXT, fill_json TEXT, forced_sale INTEGER, created_at TEXT,
  PRIMARY KEY (run_id, trajectory_id, sim_day)
);
CREATE TABLE IF NOT EXISTS equity (
  run_id TEXT, trajectory_id TEXT, sim_day TEXT, cash REAL, positions_value REAL, net_equity REAL,
  gross_equity REAL, compute_cost_cum REAL, trading_cost_cum REAL, weights_json TEXT, ruined INTEGER,
  PRIMARY KEY (run_id, trajectory_id, sim_day)
);
CREATE TABLE IF NOT EXISTS memos (
  run_id TEXT, trajectory_id TEXT, round INTEGER, sim_day TEXT, memo TEXT, changes TEXT, report_json TEXT,
  shuffled INTEGER, cost REAL, call_ids TEXT, created_at TEXT, PRIMARY KEY (run_id, trajectory_id, round)
);
CREATE TABLE IF NOT EXISTS events (
  run_id TEXT, sim_day TEXT, trajectory_id TEXT, kind TEXT, payload_json TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS news_scrubbed (
  news_id TEXT PRIMARY KEY, kind TEXT, ticker TEXT, published_at TEXT, sim_day TEXT, status TEXT,
  text_tokenized TEXT, attempts INTEGER, regex_hits TEXT, entail_ok INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS synth_news (
  item_id TEXT PRIMARY KEY, asset TEXT, sim_day TEXT, kind TEXT, event_type TEXT, sentiment REAL,
  text_tokenized TEXT, status TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS probes (
  probe_id TEXT PRIMARY KEY, run_id TEXT, week INTEGER, sim_day TEXT, model_key TEXT, asset_key TEXT,
  truth TEXT, is_synthetic INTEGER, answer TEXT, answer_synthetic TEXT, call_id TEXT, created_at TEXT
);
"""


class RunStore(_SQLite):
    def __init__(self, path: Path | None = None):
        super().__init__(path or DATA_DIR / "db" / "runs.sqlite")
        self.executescript(RUN_SCHEMA)

    def put(self, table: str, row: dict[str, Any], replace: bool = True) -> None:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        self.execute(f"{verb} INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))

    def event(self, run_id: str, sim_day: str | None, trajectory_id: str | None, kind: str, payload: Any) -> None:
        self.put(
            "events",
            {"run_id": run_id, "sim_day": sim_day, "trajectory_id": trajectory_id, "kind": kind,
             "payload_json": canonical_json(payload), "created_at": now_utc()},
            replace=False,
        )
