"""Structured JSON logging to stderr and data/logs/dlo.jsonl."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from dlo.config import DATA_DIR


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            out.update(extra)
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


_configured = False


def get(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        root = logging.getLogger("dlo")
        root.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(_JsonFormatter())
        root.addHandler(h)
        try:
            (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(DATA_DIR / "logs" / "dlo.jsonl")
            fh.setFormatter(_JsonFormatter())
            root.addHandler(fh)
        except OSError:
            pass
        root.propagate = False
        _configured = True
    return logging.getLogger(f"dlo.{name}")


def event(logger: logging.Logger, msg: str, **fields) -> None:
    logger.info(msg, extra={"fields": fields})
