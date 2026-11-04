"""Scrubber v2 control flow with a mock LLM: attack flags, strict retry, drop, token rules and versioned ids."""
import asyncio
import json

from dlo.anon import scrub
from dlo.clients.openrouter_chat import MockLLMClient
from dlo.store import RunStore

ITEM = {"id": 1, "headline": "Exxon raises capex", "summary": "Exxon Mobil lifts spending on offshore wells.",
        "created_at": "2025-12-01T15:00:00Z"}
NEUTRAL = {"industry_probability": {k: 0.2 for k in scrub._IND_KEYS.values()}, "company_guess": "",
           "company_probability": 0.0, "cue": ""}
ENERGY = dict(NEUTRAL, industry_probability={**NEUTRAL["industry_probability"], "energy": 0.6, "technology": 0.0},
              cue="wells")
OK_ENTAIL = {"supported": True, "unsupported_sentences": [], "named_entities": []}


def _responder(scrub_texts: list[str], attacks: list[dict]):
    s_iter, a_iter = iter(scrub_texts), iter(attacks)

    def f(req):
        if req.purpose.startswith("scrub"):
            return json.dumps({"text": next(s_iter), "drop": False})
        if req.purpose == "attack":
            return json.dumps(next(a_iter))
        if req.purpose == "entail":
            return json.dumps(OK_ENTAIL)
        raise AssertionError(req.purpose)
    return f


def _run(client, subject="XOM", kind="asset", item_id=1):
    return asyncio.run(scrub.scrub_item(client, RunStore(), dict(ITEM, id=item_id), subject, kind, "2025-12-01"))


def test_industry_leak_triggers_strict_rescrub_then_ok():
    c = MockLLMClient(_responder(["[[XOM]] raised spending on wells.", "[[XOM]] raised spending on a project."],
                                 [ENERGY, NEUTRAL]))
    row = _run(c)
    assert row["status"] == "ok" and row["attempts"] == 2
    assert row["news_id"] == f"{scrub.SCRUB_VERSION}:asset:XOM:1"
    assert [r.purpose for r in c.calls] == ["scrub", "attack", "scrub_strict", "attack", "entail"]
    assert "industry energy" in c.calls[2].messages[0]["content"]          # the cue is fed back to the re-scrub


def test_second_leak_drops_item():
    c = MockLLMClient(_responder(["[[XOM]] drilled wells.", "[[XOM]] drilled more wells."], [ENERGY, ENERGY]))
    row = _run(c, item_id=2)
    assert row["status"] == "dropped_leak" and row["text_tokenized"] == ""


def test_missing_subject_token_is_a_leak_and_subject_phrase_is_tokenized():
    c = MockLLMClient(_responder(["A company raised spending.", "The subject company raised spending."], [NEUTRAL]))
    row = _run(c, item_id=3)
    assert row["status"] == "ok" and row["text_tokenized"] == "[[XOM]] raised spending."


def test_market_items_never_carry_company_tokens_and_skip_attack():
    c = MockLLMClient(_responder(["Stocks fell as [[NVDA]] slid."], []))
    row = _run(c, subject=None, kind="market", item_id=4)
    assert row["status"] == "ok" and "[[" not in row["text_tokenized"]
    assert "attack" not in [r.purpose for r in c.calls]


def test_company_guess_matching():
    assert scrub._names_company("NVIDIA Corporation", "NVDA")
    assert not scrub._names_company("Apple", "NVDA")
    assert not scrub._names_company("cannot tell", "NVDA")
    assert scrub._names_company("Apple", "S03")       # any confident name for a synthetic subject is a leak


def test_identified_assets_rule():
    from dlo.analysis.metrics import identified_assets
    store = RunStore()
    rows = []
    for w in range(30):
        rows.append(("NVDA", 0, "NVIDIA Corp", "NVIDIA Corp" if w < 9 else "cannot tell", "real"))       # 30%: dropped
        rows.append(("JPM", 0, "JPMorgan Chase & Co", "JPMorgan Chase & Co" if w < 4 else "Apple Inc", "real"))  # 13%: kept
        rows.append(("S01", 1, "synthetic (not a real company)", "cannot tell", "synthetic" if w < 22 else "real"))
        rows.append(("S02", 1, "synthetic (not a real company)", "cannot tell", "synthetic" if w < 16 else "real"))
        for k, syn, truth, ans, ros in rows[-4:]:
            store.put("probes", {"probe_id": f"t:{w}:{k}", "run_id": "t-ident", "week": w, "sim_day": "2026-03-06",
                                 "model_key": "terra", "asset_key": k, "truth": truth, "is_synthetic": syn,
                                 "answer": ans, "answer_synthetic": ros, "call_id": None, "created_at": "x"})
    assert identified_assets(store, "t-ident")["terra"]["identified"] == ["NVDA", "S01"]
