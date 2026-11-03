"""Offline gates.

G1  Gradient-boosted classifier on ~30 stylized-fact features, real vs synthetic, grouped CV AUC <= 0.60.
    Uses only pre-test data: history before the dev window plus the dev window. Synthetic replicates are
    generated over the same sessions with independent seeds. CV folds are grouped by asset.
G4  500 simulated paths per synthetic asset: realized strength within +/-0.02 of target.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from dlo import config
from dlo.data import calendar
from dlo.synth import donor_pool, generator

CHUNK = 60


def _ac(x: np.ndarray, k: int) -> float:
    if len(x) <= k + 2 or np.std(x) == 0:
        return np.nan
    return float(np.corrcoef(x[:-k], x[k:])[0, 1])


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def features(r, gap, rng_, rv, mret, news) -> dict[str, float]:
    a = np.abs(r)
    sd = np.std(r)
    beta = np.cov(r, mret)[0, 1] / np.var(mret) if np.var(mret) > 0 else np.nan
    cum = np.cumsum(r)
    f = {
        "mean": np.mean(r), "std": sd, "skew": stats.skew(r), "kurt": stats.kurtosis(r),
        "ac1": _ac(r, 1), "ac2": _ac(r, 2), "ac5": _ac(r, 5),
        "ac1_abs": _ac(a, 1), "ac5_abs": _ac(a, 5), "ac10_abs": _ac(a, 10), "ac1_sq": _ac(r**2, 1),
        "beta": beta, "corr_mkt": _corr(r, mret), "idio": np.std(r - beta * mret) if np.isfinite(beta) else np.nan,
        "mdd": float(np.max(np.maximum.accumulate(cum) - cum)), "max_abs": np.max(a),
        "tail2": np.mean(a > 2 * sd) if sd > 0 else np.nan, "leverage": _corr(r[:-1], a[1:]),
        "gap_mean": np.mean(gap), "gap_std": np.std(gap), "gap_r": _corr(gap, r), "gap_share": _corr(np.abs(gap), a),
        "rng_mean": np.mean(rng_), "rng_std": np.std(rng_), "rng_abs": _corr(rng_, a),
        "rv_mean": np.nanmean(rv), "rv_std": np.nanstd(rv), "rv_abs": _corr(rv, a), "rv_ac1": _ac(np.nan_to_num(rv), 1),
        "news_mean": np.nan, "news_std": np.nan, "news_zero": np.nan, "news_abs_next": np.nan,
    }
    if news is not None and np.isfinite(news).all():
        f.update({"news_mean": np.mean(news), "news_std": np.std(news), "news_zero": np.mean(news == 0),
                  "news_abs_next": _corr(news[:-1], a[1:])})
    return {k: float(v) if v is not None else np.nan for k, v in f.items()}


def _real_panel(bars: pl.DataFrame, sym: str) -> pl.DataFrame:
    return bars.filter(pl.col("symbol") == sym).sort("session").with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("r"),
        (pl.col("open") / pl.col("close").shift(1)).log().alias("gap"),
        ((pl.col("high") - pl.col("low")) / pl.col("close").shift(1)).alias("rng"),
        (pl.col("volume") / pl.col("volume").shift(1).rolling_median(60)).alias("rv"),
    )


def g1(bars: pl.DataFrame, real_counts: pl.DataFrame, n_replicates: int = 3, seed: int = 7) -> dict:
    dev_start, dev_end = config.window("dev")
    pools = donor_pool.build(bars)
    specs = generator.make_specs(pools, config.universe()["synthetic"]["seed"])
    start = date(2024, 1, 2)
    sessions = calendar.sessions(start, dev_end)
    mkt = donor_pool.market_returns(bars)
    mret_map = dict(zip(mkt["session"].to_list(), mkt["mret"].to_list()))
    counts_map = {(t, d): n for t, d, n in real_counts.select("ticker", "session", "n").iter_rows()}
    rc = real_counts.filter(pl.col("session") <= dev_end)["n"].to_numpy()
    chunks = [sessions[i:i + CHUNK] for i in range(len(sessions) % CHUNK, len(sessions), CHUNK)]
    X, y, groups = [], [], []

    def add(sym_or_key, df: pl.DataFrame, label: int, news_fn):
        for ch in chunks:
            sub = df.filter(pl.col("session").is_in(ch)).drop_nulls(["r"])
            if sub.height < CHUNK * 0.9:
                continue
            m = np.array([mret_map.get(d, 0.0) for d in sub["session"].to_list()])
            news = news_fn(sub["session"].to_list())
            X.append(features(sub["r"].to_numpy(), sub["gap"].to_numpy(), sub["rng"].to_numpy(),
                              sub["rv"].to_numpy(), m, news))
            y.append(label)
            groups.append(sym_or_key)

    for t in config.real_tickers():
        def nf(ds, t=t):
            return np.array([counts_map[(t, d)] for d in ds], float) if all((t, d) in counts_map for d in ds) else None
        add(t, _real_panel(bars, t), 0, nf)
    for rep in range(n_replicates):
        for s in specs:
            sp = replace(s, seed=s.seed + 1000 * (rep + 1))
            df = generator.generate(sp, pools[s.sector], sessions, mret_map, rc)
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1)).log().alias("r"),
                (pl.col("open") / pl.col("close").shift(1)).log().alias("gap"),
                ((pl.col("high") - pl.col("low")) / pl.col("close").shift(1)).alias("rng"),
                (pl.col("volume") / pl.col("volume").shift(1).rolling_median(60)).alias("rv"),
            )
            cmap = dict(zip(df["session"].to_list(), df["news_count"].to_list()))

            def nf(ds, cmap=cmap):
                return np.array([cmap[d] for d in ds], float) if all(d >= dev_start for d in ds) else None
            add(f"{s.key}", df, 1, nf)

    names = sorted(X[0])
    Xm = np.array([[row[k] for k in names] for row in X], dtype=float)
    Xm[~np.isfinite(Xm)] = np.nan
    keep = [i for i in range(Xm.shape[1]) if np.isfinite(Xm[:, i]).sum() >= 10 and np.nanstd(Xm[:, i]) > 0]
    names, Xm = [names[i] for i in keep], Xm[:, keep]
    y_arr, g_arr = np.array(y), np.array(groups)
    oof = np.zeros(len(y_arr))
    for tr, te in GroupKFold(n_splits=5).split(Xm, y_arr, g_arr):
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed,
                                             class_weight="balanced")
        clf.fit(Xm[tr], y_arr[tr])
        oof[te] = clf.predict_proba(Xm[te])[:, 1]
    auc = float(roc_auc_score(y_arr, oof))
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed,
                                         class_weight="balanced").fit(Xm, y_arr)
    uni = {}
    for i, k in enumerate(names):
        m = np.isfinite(Xm[:, i])
        if len(set(y_arr[m])) == 2:
            uni[k] = float(roc_auc_score(y_arr[m], Xm[m, i]))
    real_means = {k: float(np.nanmean(Xm[y_arr == 0, i])) for i, k in enumerate(names)}
    syn_means = {k: float(np.nanmean(Xm[y_arr == 1, i])) for i, k in enumerate(names)}
    return {"gate": "G1", "auc": auc, "pass": auc <= 0.60, "n_real": int((y_arr == 0).sum()),
            "n_synth": int((y_arr == 1).sum()), "features": names, "chunk_sessions": CHUNK,
            "feature_means_real": real_means, "feature_means_synth": syn_means,
            "univariate_auc": dict(sorted(uni.items(), key=lambda kv: -abs(kv[1] - 0.5)))}


def g4(bars: pl.DataFrame, n_paths: int = 500, n_days: int = 230) -> dict:
    pools = donor_pool.build(bars)
    specs = generator.make_specs(pools, config.universe()["synthetic"]["seed"])
    out = []
    p_event = config.universe()["synthetic"]["event_day_prob"]
    for s in specs:
        realized = []
        rng = np.random.default_rng(s.seed + 99)
        for _ in range(n_paths):
            eta, _, _ = generator.bootstrap_eta(pools[s.sector], n_days, rng)
            events = rng.random(n_days) < p_event
            z = rng.standard_normal(n_days)
            group = s.group if s.group != "noise" else "momentum_probe"
            if s.group == "noise":
                eps, _ = generator.plant("noise", 0.0, eta, events, z)
                # measure against the momentum construction to confirm no predictable part
                sig = np.array([np.sqrt(5) * eps[t - 5:t].mean() if t >= 5 else np.nan for t in range(n_days)])
            else:
                eps, sig = generator.plant(s.group, s.strength, eta, events, z)
            m = np.isfinite(sig)
            # The planted process is mean-zero by construction, so strength is the uncentered correlation pooled
            # over paths. Per-path demeaned correlations carry a small-sample bias of about -5/n (~ -0.02 here).
            realized.append((float(np.sum(eps[m] * sig[m])), float(np.sum(eps[m] ** 2)), float(np.sum(sig[m] ** 2)),
                             float(np.corrcoef(eps[m], sig[m])[0, 1])))
            _ = group
        num, de, ds_ = (sum(x[i] for x in realized) for i in range(3))
        mean = num / np.sqrt(de * ds_)
        per_path = np.array([x[3] for x in realized])
        out.append({"asset": s.key, "group": s.group, "target": s.strength, "realized": float(mean),
                    "per_path_demeaned_mean": float(per_path.mean()),
                    "se": float(per_path.std() / np.sqrt(n_paths)), "pass": abs(mean - s.strength) <= 0.02})
    return {"gate": "G4", "pass": all(o["pass"] for o in out), "assets": out, "paths": n_paths, "days": n_days}
