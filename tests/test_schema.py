import json

import pytest

from dlo.agents.schema import LABELS, ValidationFailed, validate


def base(**kw):
    d = {"scores": {k: 0.0 for k in LABELS}, "weights": {k: 0.0 for k in LABELS}, "confidence": 0.5,
         "next_review_in_days": 1, "rationale": {k: "" for k in LABELS}, "market_view": "flat"}
    d.update(kw)
    return d


def test_all_cash_valid():
    d = validate(json.dumps(base()))
    assert d.cash == 1.0 and d.next_review_in_days == 1


def test_renormalize_within_tolerance():
    w = {k: 0.0 for k in LABELS}
    w["ASSET_01"] = 0.3
    r = {k: "" for k in LABELS}
    r["ASSET_01"] = "trend"
    d = validate(json.dumps(base(weights=w, rationale=r)))
    assert abs(d.cash - 0.7) < 1e-12 and not d.flags          # cash is derived, no renormalization needed
    w = {k: 0.0 for k in LABELS}
    for k in LABELS[:4]:
        w[k], r[k] = 0.2525, "trend"                           # sums to 1.01: within tolerance
    d = validate(json.dumps(base(weights=w, rationale=r)))
    assert abs(sum(d.weights.values()) - 1) < 1e-12 and d.cash == 0.0
    assert any(f.startswith("renormalized") for f in d.flags)


@pytest.mark.parametrize("mut,msg", [
    (lambda d: d["weights"].__setitem__("ASSET_02", 0.31), "cap"),
    (lambda d: [d["weights"].__setitem__(k, 0.3) for k in LABELS[:4]], "sum"),
    (lambda d: d["scores"].__setitem__("ASSET_03", 1.5), "scores"),
    (lambda d: d.__setitem__("next_review_in_days", 4), "next_review"),
])
def test_hard_failures(mut, msg):
    d = base()
    mut(d)
    with pytest.raises(ValidationFailed, match=msg):
        validate(json.dumps(d))


def test_rationale_required():
    s = {k: 0.0 for k in LABELS}
    s["ASSET_05"] = 0.8
    with pytest.raises(ValidationFailed, match="rationale"):
        validate(json.dumps(base(scores=s)))


def test_not_json():
    with pytest.raises(ValidationFailed, match="JSON"):
        validate("{oops")
