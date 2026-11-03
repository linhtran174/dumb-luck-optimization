"""Config loader. All YAML under config/; secrets only from the environment (.env is loaded if present)."""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("DLO_CONFIG_DIR", ROOT / "config"))
DATA_DIR = Path(os.environ.get("DLO_DATA_DIR", ROOT / "data"))

load_dotenv(ROOT / ".env", override=False)


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache
def models() -> dict[str, Any]:
    return _load("models.yaml")["models"]


@lru_cache
def universe() -> dict[str, Any]:
    return _load("universe.yaml")


@lru_cache
def windows() -> dict[str, Any]:
    return _load("windows.yaml")


@lru_cache
def arms() -> dict[str, Any]:
    return _load("arms.yaml")


@lru_cache
def costs() -> dict[str, Any]:
    return _load("costs.yaml")


def secret(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"missing secret {name}: set it in the environment or .env")
    return val


def real_tickers() -> list[str]:
    return [c["ticker"] for s in universe()["sectors"].values() for c in s["real"]]


def sector_of() -> dict[str, str]:
    return {c["ticker"]: sec for sec, s in universe()["sectors"].items() for c in s["real"]}


def donor_tickers() -> dict[str, list[str]]:
    return {sec: s["donors"] for sec, s in universe()["sectors"].items()}


def window(name: str) -> tuple[date, date]:
    w = windows()["windows"][name]
    return _d(w["start"]), _d(w["end"])


def _d(x: Any) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))
