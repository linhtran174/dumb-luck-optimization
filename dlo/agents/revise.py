"""Self-revision loop (RQ5): performance report, shuffled control, memo rewrite (charged), snapshots."""
from __future__ import annotations

import json

import numpy as np

from dlo.agents import prompts
from dlo.agents.state import TrajState
from dlo.clients.openrouter_chat import ChatRequest, LLMClient, LLMError


def build_report(equity_rows: list[dict], panel_close: dict[tuple[str, str], float], st: TrajState,
                 prev_close_lookup) -> dict:
    """equity_rows: this trajectory's last N daily rows (sim_day, net_equity, compute_cost_cum, weights_json by key),
    oldest first, including the row just before the period as the baseline."""
    base, rows = equity_rows[0], equity_rows[1:]
    total_pnl = rows[-1]["net_equity"] - base["net_equity"]
    compute = rows[-1]["compute_cost_cum"] - base["compute_cost_cum"]
    net_ret = total_pnl / base["net_equity"] if base["net_equity"] else 0.0
    per_asset: dict[str, dict] = {}
    per_day = []
    prev_w = json.loads(base["weights_json"])
    for i, r in enumerate(rows):
        day_contrib = 0.0
        for key, w in prev_w.items():
            if w <= 0:
                continue
            c1 = panel_close[(key, r["sim_day"])]
            c0 = prev_close_lookup(key, r["sim_day"])
            ret = c1 / c0 - 1
            a = per_asset.setdefault(st.labels[key], {"w_sum": 0.0, "contrib": 0.0, "held": 0, "hits": 0})
            a["w_sum"] += w
            a["contrib"] += w * ret * base["net_equity"]
            a["held"] += 1
            a["hits"] += int(ret > 0)
            day_contrib += w * ret * base["net_equity"]
        per_day.append({"day_offset": i + 1 - len(rows), "pnl_from_positions": day_contrib})
        prev_w = json.loads(r["weights_json"])
    n = len(rows)
    table = [{"asset": lab, "avg_weight": a["w_sum"] / n, "return_contribution": a["contrib"],
              "hit_rate": a["hits"] / a["held"] if a["held"] else None} for lab, a in sorted(per_asset.items())]
    explained = sum(t["return_contribution"] for t in table)
    return {"days": n, "total_pnl": total_pnl, "compute_cost": compute, "net_return": net_ret,
            "attribution": table, "daily": per_day,
            "other_pnl": total_pnl - explained}  # trading costs, inference costs, intraday rebalancing


def shuffle_report(report: dict, rng: np.random.Generator) -> dict:
    """Control arm: permute attribution asset labels and day order; totals stay true."""
    r = json.loads(json.dumps(report))
    labels = [t["asset"] for t in r["attribution"]]
    perm = rng.permutation(len(labels))
    for t, j in zip(r["attribution"], perm):
        t["asset"] = labels[j]
    r["attribution"].sort(key=lambda t: t["asset"])
    vals = [d["pnl_from_positions"] for d in r["daily"]]
    order = rng.permutation(len(vals))
    for d, j in zip(r["daily"], order):
        d["pnl_from_positions"] = vals[j]
    return r


def render_report(rep: dict) -> str:
    lines = [f"PERFORMANCE REPORT, last {rep['days']} trading days",
             f"total P&L {rep['total_pnl']:.2f}; inference cost {rep['compute_cost']:.4f}; "
             f"net return {rep['net_return'] * 100:.2f}%",
             "Attribution by asset (average weight, P&L contribution, share of held days with a gain):"]
    for t in rep["attribution"]:
        hr = "na" if t["hit_rate"] is None else f"{t['hit_rate']:.2f}"
        lines.append(f"- {t['asset']}: avg_weight {t['avg_weight']:.3f}, contribution {t['return_contribution']:.2f}, "
                     f"hit_rate {hr}")
    lines.append("Daily P&L from positions: " + ", ".join(f"D{d['day_offset']} {d['pnl_from_positions']:.2f}"
                                                         for d in rep["daily"]))
    lines.append(f"Other P&L (trading costs, inference costs, intraday rebalancing): {rep['other_pnl']:.2f}")
    return "\n".join(lines)


async def rewrite_memo(client: LLMClient, st: TrajState, report_text: str, recent: list[dict], sim_day: str,
                       budget: str) -> tuple[dict | None, float, list[str], str | None]:
    dec = "\n".join(f"- D{h['day_index']}: weights {json.dumps({k: round(v, 3) for k, v in h['weights'].items() if v > 0})}"
                    f" cash {h['cash']:.2f}; view: {h['market_view']}" for h in recent[-10:]) or "- none"
    msg = (f"CURRENT MEMO\n{st.memo or '(empty: this is your first memo)'}\n\n{report_text}\n\n"
           f"YOUR LAST DECISIONS (day indexes)\n{dec}\n\nReturn JSON with memo and changes.")
    try:
        out = await client.chat(ChatRequest(
            model_key=st.model_key, purpose="memo", budget=budget, charged=True, trajectory_id=st.traj_id,
            sim_day=sim_day, json_schema=prompts.memo_schema(),
            messages=[{"role": "system", "content": prompts.memo_system()}, {"role": "user", "content": msg}]))
    except LLMError as e:
        return None, 0.0, [], str(e)
    try:
        d = json.loads(out.content)
        memo = " ".join(str(d["memo"]).split()[:300])
        changes = " ".join(str(d["changes"]).split()[:80])
        return {"memo": memo, "changes": changes}, out.cost, [out.call_id], None
    except (json.JSONDecodeError, KeyError) as e:
        return None, out.cost, [out.call_id], f"invalid memo json: {e}"
