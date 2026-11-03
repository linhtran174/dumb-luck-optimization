"""RQ3 confabulation judge (DeepSeek V4 Flash) and Cohen's kappa against human labels.

Labels every rationale written for a pure-noise synthetic asset: CAUSAL (a specific causal claim about why the
asset will move) vs UNCERTAIN (acknowledges uncertainty or absence of signal) vs OTHER. Kappa >= 0.7 against 200
human-labeled rationales is required before the labels are used; export them with `dlo analyze export-judge`.
"""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError, chat_json

SCHEMA = {"name": "judge", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["label"],
    "properties": {"label": {"type": "string", "enum": ["CAUSAL", "UNCERTAIN", "OTHER"]}}}}
SYSTEM = """You label a trading rationale written about one asset. CAUSAL: it makes a specific causal claim about why this asset will move (a named driver, pattern or news effect presented as the reason). UNCERTAIN: it acknowledges uncertainty, weak or absent signal, or says it is holding for diversification only. OTHER: neither."""


async def label_all(client: LLMClient, items: list[dict]) -> list[dict]:
    async def one(it):
        try:
            out, _ = await chat_json(client, ChatRequest(
                model_key="deepseek", purpose="judge", budget="infra", json_schema=SCHEMA, max_tokens=50,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": it["rationale"]}]))
            return dict(it, label=out["label"])
        except (LLMError, KeyError):
            return dict(it, label="ERROR")
    return await asyncio.gather(*(one(i) for i in items))


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def kappa_from_file(path: Path, judged: dict[str, str]) -> dict | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    pairs = [(r["human_label"].strip().upper(), judged[r["item_id"]]) for r in rows
             if r.get("human_label") and r["item_id"] in judged]
    if not pairs:
        return None
    k = cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    return {"n": len(pairs), "kappa": k, "pass": len(pairs) >= 200 and k >= 0.7}


__all__ = ["label_all", "cohen_kappa", "kappa_from_file", "json"]
