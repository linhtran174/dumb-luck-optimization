"""Assemble the run report (JSON + Markdown) from persisted data. Free except for the confabulation judge."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

import numpy as np

from dlo import config
from dlo.analysis import judge, metrics
from dlo.clients.openrouter_chat import LLMClient
from dlo.store import LLMLedger, RunStore

REPORT_DIR = config.DATA_DIR / "derived" / "reports"


def _news_counts(store: RunStore, rd: metrics.RunData) -> dict[tuple[str, date], int]:
    from dlo.anon.pipeline import scrubbed_for_day
    out = {}
    for d in rd.days:
        sc = scrubbed_for_day(store, rd.view_snap, d)
        for k, v in sc["assets"].items():
            out[(k, d)] = len(v)
    from dlo.anon.scrub import SCRUB_VERSION
    for r in store.query("SELECT asset, sim_day, COUNT(*) n FROM synth_news WHERE status='ok' AND item_id LIKE ? "
                         "GROUP BY 1, 2", (f"{SCRUB_VERSION}:%",)):
        out[(r["asset"], date.fromisoformat(r["sim_day"]))] = r["n"]
    return out


def _mcs(rd: metrics.RunData) -> dict | None:
    by_t = defaultdict(list)
    for e in rd.equity:
        by_t[e["trajectory_id"]].append(e["net_equity"])
    ts = sorted(by_t)
    L = min(len(v) for v in by_t.values()) if by_t else 0
    if len(ts) < 2 or L < 4:
        return None
    from arch.bootstrap import MCS
    cap = config.costs()["capital_usd"]
    losses = np.column_stack([-np.diff(np.concatenate([[cap], by_t[t][:L]])) for t in ts])
    m = MCS(losses, size=0.10, block_size=5, method="R", seed=0)
    m.compute()
    return {"included": [ts[i] for i in m.included], "pvalues": {ts[i]: float(p) for i, p in
                                                                  zip(m.pvalues.index, m.pvalues.iloc[:, 0])}}


async def build(store: RunStore, run_id: str, rd: metrics.RunData, client: LLMClient | None) -> dict:
    news_counts = _news_counts(store, rd)
    comp = metrics.comparators(rd, news_counts)
    ew_final = comp["baselines"]["ew_daily"]
    port = metrics.portfolio_metrics(rd, ew_final)
    rand = comp.pop("random_final")
    for t, p in port.items():
        p["random_agent_percentile_gross"] = float((rand < p["final_gross_equity"]).mean() * 100)
    rep = {"run_id": run_id, "days": [str(rd.days[0]), str(rd.days[-1])], "n_days": len(rd.days),
           "ic_and_detection": metrics.ic_metrics(rd), "portfolio": port, "behavior": metrics.behavior(rd),
           "comparators": comp, "gates": metrics.gate_checks(rd),
           "random_agents": {"n": int(len(rand)), "p5": float(np.percentile(rand, 5)),
                             "p50": float(np.percentile(rand, 50)), "p95": float(np.percentile(rand, 95))}}
    try:
        rep["mcs"] = _mcs(rd)
    except Exception as e:  # small samples can make the MCS bootstrap degenerate
        rep["mcs"] = {"error": str(e)}
    rep["memos"] = [dict(r) for r in store.query(
        "SELECT trajectory_id, round, sim_day, shuffled, cost, substr(memo,1,400) memo, changes FROM memos "
        "WHERE run_id=? ORDER BY trajectory_id, round", (run_id,))]
    # RQ3 confabulation judge on pure-noise rationales
    noise_keys = {k for k, s in rd.specs.items() if s["group"] == "noise"}
    items = []
    for d in rd.decisions:
        if not d["decision_json"]:
            continue
        st = rd.states[d["trajectory_id"]]
        for lab, text in json.loads(d["decision_json"])["rationale"].items():
            if text.strip() and st.label_to_key()[lab] in noise_keys:
                items.append({"item_id": f"{d['trajectory_id']}|{d['sim_day']}|{lab}", "rationale": text})
    if client is not None and items:
        labels = await judge.label_all(client, items)
        by_t = defaultdict(list)
        for x in labels:
            by_t[x["item_id"].split("|")[0]].append(x["label"])
        rep["confabulation"] = {t: {"n": len(v), "causal_rate": v.count("CAUSAL") / len(v)} for t, v in by_t.items()}
        rep["judge_kappa"] = judge.kappa_from_file(REPORT_DIR / f"{run_id.replace('/', '_')}_human_labels.csv",
                                                   {x["item_id"]: x["label"] for x in labels}) or \
            "pending: needs >= 200 human-labeled rationales (dlo analyze export-judge)"
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / f"{run_id.replace('/', '_')}_judge_items.json").write_text(json.dumps(labels, indent=1))
    led = LLMLedger()
    rep["spend"] = {"by_budget": {b: led.spent(b) for b in ("trial", "dev", "test", "infra")},
                    "run_decision_cost": float(sum(d["cost"] or 0 for d in rd.decisions)),
                    "run_memo_cost": float(sum(m["cost"] or 0 for m in rep["memos"]))}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    base = REPORT_DIR / run_id.replace("/", "_")
    base.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    base.with_suffix(".md").write_text(markdown(rep))
    return rep


def _f(x, nd=4):
    return "na" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def markdown(rep: dict) -> str:
    L = [f"# Run report: {rep['run_id']}", f"Days {rep['days'][0]} to {rep['days'][1]} ({rep['n_days']} sessions).", ""]
    g = rep["gates"]
    L += ["## Gates", f"- G5 output validity (first try): {_f(g['G5']['valid_first_share'], 3)} over {g['G5']['n']} "
          f"decisions; pass={g['G5']['pass']}; by model {json.dumps({k: round(v, 3) for k, v in g['G5']['by_model'].items()})}",
          f"- State size: max {g['state_bytes']['max']} bytes, mean {_f(g['state_bytes']['mean'], 0)}; all under 30 KB: "
          f"{g['state_bytes']['all_under_30kb']}",
          f"- G3 regex hits in accepted scrubbed text: {g['G3']['regex_hits_in_accepted_text']}"]
    for m, v in g["G2"].items():
        L.append(f"- G2 {m}: real-vs-synthetic accuracy {_f(v['real_vs_synth_accuracy'], 3)} (n={v['n']}), pass={v['pass']}")
    for m, v in g["G3"]["probe"].items():
        L.append(f"- G3 {m}: re-identification {_f(v['reidentification_rate'], 3)} (n={v['n']}), pass={v['pass']}")
    L += ["", "## Portfolio (RQ1, RQ4)", "| trajectory | net equity | gross ret | compute cost | trading cost | "
          "decisions | sleep share | random pctile | break-even K |", "|---|---|---|---|---|---|---|---|---|"]
    for t, p in sorted(rep["portfolio"].items()):
        L.append(f"| {t} | {p['final_net_equity']:.2f} | {p['gross_return'] * 100:.2f}% | {p['compute_cost']:.4f} | "
                 f"{p['trading_cost']:.4f} | {p['decisions']} | {p['sleep_share']:.2f} | "
                 f"{p['random_agent_percentile_gross']:.0f} | {_f(p['break_even_capital'], 0)} |")
    L += ["", "Comparators (final net equity): " + ", ".join(f"{k} {v:.2f}" for k, v in rep["comparators"]["baselines"].items()),
          f"Random agents (n={rep['random_agents']['n']}): p5 {rep['random_agents']['p5']:.2f}, "
          f"p50 {rep['random_agents']['p50']:.2f}, p95 {rep['random_agents']['p95']:.2f}", ""]
    L += ["## Skill and detection (RQ1, RQ2)", "| trajectory | IC1 | IC5 | top3-bottom3 | detection (score vs planted) |",
          "|---|---|---|---|---|"]
    for t, r in sorted(rep["ic_and_detection"].items()):
        det = "; ".join(f"{k[4:]} {_f(v['corr'], 2)} (n={v['n']})" for k, v in sorted(r.items()) if k.startswith("det:"))
        L.append(f"| {t} | {_f(r['ic1']['mean'], 3)} (n={r['ic1']['n']}) | {_f(r['ic5']['mean'], 3)} | "
                 f"{_f(r['spread']['mean'], 4)} | {det} |")
    L += ["", "## Behavior (RQ3)", "| trajectory | mean cash | turnover | index-reversion share | momentum corr | "
          "noise-asset weight | drawdown days |", "|---|---|---|---|---|---|---|"]
    for t, b in sorted(rep["behavior"].items()):
        L.append(f"| {t} | {b['mean_cash']:.2f} | {b['mean_turnover']:.3f} | {b['index_reversion_share']:.2f} | "
                 f"{_f(b['momentum_chasing_corr'], 2)} | {b['noise_asset_mean_weight']:.3f} | {b['drawdown_days']} |")
    if "confabulation" in rep:
        L += ["", "Confabulation (share of pure-noise rationales making a causal claim; judge not yet validated): " +
              ", ".join(f"{t} {v['causal_rate']:.2f} (n={v['n']})" for t, v in sorted(rep["confabulation"].items())),
              f"Judge kappa: {rep['judge_kappa']}"]
    L += ["", "## Revise loop (RQ5)"]
    for m in rep["memos"]:
        L.append(f"- {m['trajectory_id']} round {m['round']} ({m['sim_day']}, shuffled={bool(m['shuffled'])}, "
                 f"cost {_f(m['cost'])}): {(m['changes'] or '')[:200]}")
    L += ["", f"MCS (10%): {json.dumps(rep.get('mcs'))[:600]}", "",
          "## Spend", json.dumps(rep["spend"], indent=1)]
    return "\n".join(L) + "\n"
