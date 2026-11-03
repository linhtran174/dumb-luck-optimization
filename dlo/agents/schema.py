"""Decision output schema (strict JSON schema for OpenRouter) and validation/normalization."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from dlo import config

N_ASSETS = 20
LABELS = [f"ASSET_{i:02d}" for i in range(1, N_ASSETS + 1)]
REVIEW_CHOICES = [1, 2, 3, 5]
RATIONALE_WORDS = 40
VIEW_WORDS = 60


def decision_json_schema() -> dict:
    num = {"type": "number"}
    return {"name": "portfolio_decision", "strict": True, "schema": {
        "type": "object", "additionalProperties": False,
        "required": ["scores", "weights", "cash", "confidence", "next_review_in_days", "rationale", "market_view"],
        "properties": {
            "scores": {"type": "object", "additionalProperties": False, "required": LABELS,
                       "properties": {k: num for k in LABELS}},
            "weights": {"type": "object", "additionalProperties": False, "required": LABELS,
                        "properties": {k: num for k in LABELS}},
            "cash": num,
            "confidence": num,
            "next_review_in_days": {"type": "integer", "enum": REVIEW_CHOICES},
            "rationale": {"type": "object", "additionalProperties": False, "required": LABELS,
                          "properties": {k: {"type": "string"} for k in LABELS}},
            "market_view": {"type": "string"},
        }}}


@dataclass
class Decision:
    scores: dict[str, float]
    weights: dict[str, float]
    cash: float
    confidence: float
    next_review_in_days: int
    rationale: dict[str, str]
    market_view: str
    flags: list[str] = field(default_factory=list)   # soft issues fixed during normalization

    def to_json(self) -> dict:
        return {"scores": self.scores, "weights": self.weights, "cash": self.cash, "confidence": self.confidence,
                "next_review_in_days": self.next_review_in_days, "rationale": self.rationale,
                "market_view": self.market_view, "flags": self.flags}


class ValidationFailed(ValueError):
    pass


def _words(s: str) -> int:
    return len(s.split())


def _trim(s: str, n: int) -> str:
    w = s.split()
    return " ".join(w[:n])


def validate(content: str) -> Decision:
    """Parse and validate. Hard failures raise ValidationFailed (the agent gets a retry);
    soft issues (sum within tolerance, over-long text) are normalized and flagged."""
    cap = config.costs()["max_weight"]
    tol = config.costs()["weight_sum_tolerance"]
    try:
        d = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValidationFailed(f"output is not valid JSON: {e}") from None
    errs: list[str] = []
    flags: list[str] = []
    for k in ("scores", "weights", "rationale"):
        if not isinstance(d.get(k), dict):
            errs.append(f"'{k}' must be an object keyed by asset label")
    if errs:
        raise ValidationFailed("; ".join(errs))
    missing = [k for k in LABELS if k not in d["scores"] or k not in d["weights"]]
    extra = [k for k in set(d["scores"]) | set(d["weights"]) if k not in LABELS]
    if missing:
        errs.append(f"missing assets in scores/weights: {missing}")
    if extra:
        errs.append(f"unknown asset labels: {sorted(extra)}")
    if errs:
        raise ValidationFailed("; ".join(errs))
    try:
        scores = {k: float(d["scores"][k]) for k in LABELS}
        weights = {k: float(d["weights"][k]) for k in LABELS}
        cash = float(d["cash"])
        conf = float(d["confidence"])
        nr = int(d["next_review_in_days"])
    except (TypeError, ValueError, KeyError) as e:
        raise ValidationFailed(f"non-numeric or missing field: {e}") from None
    bad_s = [k for k, v in scores.items() if not -1.0 <= v <= 1.0]
    if bad_s:
        errs.append(f"scores must be in [-1, 1]: {bad_s}")
    neg = [k for k, v in weights.items() if v < 0]
    if neg or cash < 0:
        errs.append(f"weights and cash must be >= 0 (long-only): {neg + (['cash'] if cash < 0 else [])}")
    over = [k for k, v in weights.items() if v > cap + 1e-6]
    if over:
        errs.append(f"weights above the {cap:.0%} cap: {over}")
    total = sum(weights.values()) + cash
    if abs(total - 1.0) > tol:
        errs.append(f"weights plus cash sum to {total:.4f}; must be 1 +/- {tol}")
    if not 0.0 <= conf <= 1.0:
        errs.append("confidence must be in [0, 1]")
    if nr not in REVIEW_CHOICES:
        errs.append(f"next_review_in_days must be one of {REVIEW_CHOICES}")
    rationale = {k: str(d["rationale"].get(k, "") or "") for k in LABELS}
    need = [k for k in LABELS if (weights[k] > 0 or abs(scores[k]) > 0.5) and not rationale[k].strip()]
    if need:
        errs.append(f"rationale required (weight > 0 or |score| > 0.5) for: {need}")
    if errs:
        raise ValidationFailed("; ".join(errs))

    if abs(total - 1.0) > 1e-9:
        weights = {k: v / total for k, v in weights.items()}
        cash = cash / total
        flags.append(f"renormalized_sum_{total:.4f}")
        excess = sum(max(0.0, v - cap) for v in weights.values())
        if excess > 0:
            weights = {k: min(v, cap) for k, v in weights.items()}
            cash += excess
            flags.append("clipped_cap_after_renorm")
    for k, v in rationale.items():
        if _words(v) > RATIONALE_WORDS:
            rationale[k] = _trim(v, RATIONALE_WORDS)
            flags.append(f"rationale_truncated_{k}")
    mv = str(d.get("market_view", "") or "")
    if _words(mv) > VIEW_WORDS:
        mv = _trim(mv, VIEW_WORDS)
        flags.append("market_view_truncated")
    return Decision(scores, weights, cash, conf, nr, rationale, mv, flags)
