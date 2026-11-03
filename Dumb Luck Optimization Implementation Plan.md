# Dumb Luck Optimization: Implementation Plan

Nov 3, 2026 · @Linh

## Purpose and scope

Build a paper-traded benchmark that replays about eight months of post-cutoff market history one simulated day at a time. Two LLMs (GPT-5.6 Terra and GPT-5.6 Luna) trade on anonymized data, pay their own inference costs, and only ever see data timestamped up to the simulated day. Every LLM call goes through OpenRouter; all market data is pre-fetched from Alpaca into a frozen snapshot.

| RQ | Question | Component that answers it |
| --- | --- | --- |
| RQ1 | Skill vs. luck on real assets | Real assets in the main arm, random-agent null |
| RQ2 | Detection of planted signals | Synthetic planted-signal assets |
| RQ3 | Fallback behavior under noise | Pure-noise assets, drawdown episodes, trace annotation |
| RQ4 | Economic survival | Cost ledger charged to each portfolio |
| RQ5 | Self-improvement under noise | Revise arm vs. shuffled-feedback arm |

Handover notes for Claude Code: build in the milestone order below, keep every external call behind an interface so it can be mocked, never let a decision see data after its simulated day, and never let an agent see a real ticker or real date. The test window is run exactly once, after the pre-registration freeze. Paper trading only; no brokerage order is ever sent.

## Locked decisions

Two evaluated models, one infrastructure model, 20 assets (10 real across five sectors, 10 synthetic), daily decisions, and a 170-trading-day historical replay from March 2 to October 30, 2026. Pin every model to the exact slug below; never use `latest` aliases.

| Model | Role | OpenRouter slug | List price in / out (USD per 1M tokens) | Knowledge cutoff |
| --- | --- | --- | --- | --- |
| GPT-5.6 Terra | Evaluated | `openai/gpt-5.6-terra-20260709` | 2 / 12 | Feb 16, 2026 |
| GPT-5.6 Luna | Evaluated | `openai/gpt-5.6-luna-20260709` | 0.20 / 1.20 | Feb 16, 2026 |
| DeepSeek V4 Flash | Infrastructure | `deepseek/deepseek-v4-flash` (pin a dated snapshot in M1) | 0.14 / 0.28 | not published; released Apr 24, 2026 |

Sources: [GPT-5.6 Terra](https://openrouter.ai/models/openai/gpt-5.6-terra-20260709), [GPT-5.6 Luna](https://openrouter.ai/models/openai/gpt-5.6-luna-20260709), [DeepSeek V4 Flash pricing](https://computeprices.com/providers/openrouter/models/deepseek-v4-flash) and [release date](https://www.llmreference.com/compare/deepseek-v4-flash/kimi-k2-7-code). Prices are list prices as of November 2026; the simulator charges the actual `usage.cost` returned per call. Earlier candidates (Claude Opus 5.5, Sonnet 5.5, GPT-6.1 Sol and Terra, Jev) were dropped for cost or because later cutoffs would shorten the window.

**DeepSeek V4 Flash does every non-evaluated job:** news scrubbing, alias-dictionary expansion, synthetic news generation, the entailment check and the RQ3 judge. It comes from a different family than the evaluated models, so its writing style gives them no home advantage. Its training cutoff is unpublished and it may know early-window events, which is why the entailment check exists (see Risks).

**Real universe: two companies from each of five sectors with different return drivers.** Pairs matter for anonymization: if each sector had one company, any sector cue in the news (crude prices, a drug trial) would identify it outright. With pairs, and with synthetic assets also given sector-flavored news, a sector cue narrows the field to four candidates, two of them fake.

| Sector | Main driver | Provisional pair |
| --- | --- | --- |
| Information Technology | AI capex, growth multiples | NVDA, MSFT |
| Financials | Rates, credit, payment volumes | JPM, V |
| Health Care | Drug pipelines, policy | LLY, JNJ |
| Energy | Oil and gas prices | XOM, CVX |
| Consumer Staples | Defensive consumer demand | WMT, COST |

Selection rule: the two largest companies by S&P 500 weight in each sector, taken from the last SPY holdings disclosure before March 2, 2026. Selecting on later weights would leak the window's winners into the universe. One share class per company. SPY is fetched as the market series but is not tradable. Confirm the pairs from that disclosure in M2.

**Synthetic universe.** 10 assets: 4 pure noise, 3 planted momentum, 3 planted news-lead, two assigned to each sector at random (see the synthetic section).

**Window and cadence.**

- Test window: about 170 trading days, March 2 to October 30, 2026, replayed exactly once after the pre-registration freeze. The start leaves a two-week buffer after both models' February 16 cutoff. Any rerun (for a bug) is logged and reported.
- Development window: December 1, 2025 to February 13, 2026, used only to build and pass the gates. It predates the cutoff, so models may have seen it; that is acceptable because its results are never reported.
- One decision per trajectory per simulated trading day, using only data timestamped up to 18:00 ET that day; fills at the next day's open.
- Paper capital: USD 1,000 per trajectory; long-only plus cash; max 30% per asset.
- Gate G0 checks empirically that neither model knows test-window events.

## Data sources

Use Alpaca for both prices and news, on its free Basic plan with a paper account; Binance is not suitable. Binance is a crypto exchange: it has no consolidated feed for US-listed S&P 500 equities, and any tokenized-stock listings would be thin proxies trading outside US market hours.

| Need | Source | Plan limits that matter |
| --- | --- | --- |
| Daily OHLCV bars, split- and dividend-adjusted | Alpaca Market Data API v2, `feed=sip` | Free; history since 2016; the latest 15 minutes are blocked; 200 calls/min ([docs](https://docs.alpaca.markets/us/docs/about-market-data-api.md)) |
| Company and market news | Alpaca News API (Benzinga content) | Historical and real-time, back to 2015; queried by symbol and date ([announcement](https://benzinga.com/z/25468501)) |
| Trading calendar | `exchange_calendars` (XNYS) | Holidays, half days, DST |
| Donor pool for synthetic assets | Alpaca bars for same-sector S&P 500 constituents outside the universe | 2 years of history, pulled once |

**Frozen snapshot.** In M2, fetch everything once: bars from August 2025 onward (enough for the 60-day lookback before the development window), news from November 28, 2025 to October 30, 2026, and the donor pool. Store it as Parquet, record SHA-256 hashes, and make the replay read only from the snapshot, never from the live API. The News API's `start` and `end` parameters give each simulated day exactly the items published in its window, so the replay sees news as it arrived.

Two checks for M2: confirm the News API is still free on the Basic plan (the announcement describes a beta), and confirm Benzinga's terms allow storing headline and summary text for research. Store only headline, summary, timestamp and symbols, never full article bodies, and release only derived, rewritten text.

## Architecture

A single Python package replays one simulated trading day at a time from the frozen snapshot. Each stage writes its output before the next starts, so an interrupted replay resumes from the last completed day.

&#91;embedded content: daily pipeline · 9 components\]

The simulator's fills and equity feed the next day's state, and the revise loop reads stored decision history to write a new memo every 10 trading days.

**Replay loop, per simulated day (times are simulated America/New\_York)**

1. Load the day's bars from the snapshot; apply the previous day's target weights at the open; mark to close.
2. Select news published up to 18:00 ET; anonymize it; compute the day's synthetic bars and render their news.
3. Build states and run all due trajectories (those not asleep), with bounded concurrency and per-model rate limits. Calls are real-time API calls even though the market day is simulated.
4. Validate, write logs, and checkpoint the day. A whole 170-day test replay of 14 trajectories is about 2,400 decision calls, which should finish in under a day.

**Stack.** Python 3.12, httpx, Pydantic v2, polars, DuckDB with Parquet, exchange\_calendars, numpy, scipy, statsmodels, arch (block bootstrap), tenacity for retries, Typer for the CLI, pytest.

**Repo layout**

```text
dlo-bench/
  config/        models.yaml, universe.yaml, arms.yaml, costs.yaml, aliases/*.yaml
  dlo/
    clients/     openrouter_chat.py, alpaca.py
    data/        ingest.py, snapshot.py, calendar.py, validate.py
    synth/       donor_pool.py, bootstrap.py, signals.py, candles.py, news_gen.py, pregenerate.py
    anon/        scrub.py, leakcheck.py, probe.py
    agents/      state.py, schema.py, prompts/, llm_agent.py, revise.py
    sim/         portfolio.py, execution.py, ledger.py, baselines.py, random_agents.py
    orchestrator/ replay.py, checkpoint.py, run_log.py
    analysis/    metrics.py, bootstrap.py, mcs.py, behavior.py, judge.py, figures.py
  tests/
  data/          raw/, derived/, pregen/, logs/  (gitignored)
```

## Anonymization pipeline

Anonymization is one-way: every news item is scrubbed once into blind text, and nothing is ever mapped back to real names. Leakage is checked by an alias regex on every item and by a weekly probe in which the evaluated models try to identify assets.

**Asset identity**

- Blind labels are `ASSET_01` to `ASSET_20`, assigned by a random permutation per trajectory seed and fixed for the whole window. The permutation table is stored outside the prompt path.
- No sector, market cap, exchange or company description is shown to agents. The market series is labeled `MARKET`.
- Dates become a day index (`D-59` to `D0`) in every arm.

**Prices and volume (all arms)**

- Prices are indexed so each asset's close on the window's first day equals 100. The scale factor is fixed for the whole window, so the series stays continuous.
- Each asset row per day carries close-to-close return (%), overnight gap (%), intraday range (%), indexed close, and relative volume (volume divided by the median of the 60 pre-window days).
- Lookback is 60 trading days. The full state for 20 assets plus news must stay under 30 KB of text, which keeps prompts cheap and gives both models the same input.

**News**

1. **Fetch.** From the snapshot, take news per real ticker published between the previous simulated day's 18:00 ET cutoff and 18:00 ET on the simulated day. Keep items whose symbol list includes the ticker and has at most 3 symbols, deduplicate near-identical headlines, and keep at most 3 items per asset per day. Also keep up to 5 market-wide headlines (macro, Fed, indices).
2. **Scrub.** DeepSeek V4 Flash (temperature 0) rewrites each item into 1 to 3 neutral wire-style sentences. The company becomes its asset label; products, executives and partners become generic phrases ("a flagship product", "the chief executive"); a competitor in the universe becomes its asset label, otherwise "a competitor". Absolute dollar figures become relative terms (for example, "revenue up 18% year over year, above consensus"), since absolute sizes identify megacaps.
3. **Leak check.** A regex against the alias dictionary runs on every scrubbed item: any hit triggers a stricter re-scrub, and a second hit drops the item. Log hit and drop rates per company per day.

**Alias dictionary.** One YAML file per real company with ticker, legal and common names, products, executives, subsidiaries and major partners. Build it with DeepSeek V4 Flash in M4, review it by hand, and expect 50 to 150 terms per company. Because the snapshot is fixed, build it from all raw news in both windows before the run, then freeze it.

**Weekly probe of the evaluated models.** Outside any trading trajectory, show each model one asset's blind data and news and ask which of the 10 companies it is, whether it is synthetic, or "cannot tell". This yields the re-identification rate, the main leak test for prices and news together, and real-vs-synthetic accuracy (RQ2 validity).

## Synthetic asset generation

Each synthetic asset is the same day's historical SPY return times a beta, plus an idiosyncratic shock bootstrapped from real same-sector residuals. Signal is planted only in that shock, at a known strength. Because the market part is the real SPY return, synthetic assets co-move with the real ten on every day of the window.

```latex
r_{i,t} = \beta_i \, r_{M,t} + \sigma_i \, \varepsilon_{i,t}
```

| Group | Assets | Shock rule | Target strength |
| --- | --- | --- | --- |
| Pure noise | 4 | ε = η (no predictable part) | 0 |
| Planted momentum | 3 | ε = a · s(t−1) + √(1 − a²) · η, where s is the standardized trailing 5-day mean of past ε | a = 0.05, 0.10, 0.20 |
| Planted news-lead | 3 | On event days, next-day ε = c · z + √(1 − c²) · η, where z is the standardized sentiment of that day's generated event | c = 0.05, 0.10, 0.20 |

η comes from a stationary block bootstrap (mean block length 10 days) of standardized residuals, regressed on SPY, of donor stocks in the asset's assigned sector. This keeps fat tails and volatility clustering. β and σ are drawn near the assigned sector's real pair, so an energy-flavored synthetic asset behaves like an energy stock.

**News for synthetic assets.** Every synthetic asset gets the same daily item count as a randomly drawn real asset-day, so news volume never reveals which assets are fake. Events occur on 20% of days for every synthetic asset, noise assets included. For news-lead assets the event sentiment z predicts the next day; for all others z is independent of future returns. The other items are neutral filler, written in the voice of the asset's assigned sector (refinery margins for Energy, payment volumes for Financials). DeepSeek V4 Flash writes each item in the same scrubbed wire style as real news (event types: earnings, guidance, analyst action, product, contract, regulatory, management, filler), so style matches.

**Candles.** Each simulated day borrows the candle shape (open gap, high and low excursions) from the donor day that supplied η. Shift the open by half the difference between synthetic and donor log returns, shift high, low and close by the full difference, then enforce low ≤ open, close ≤ high. Volume is the donor day's relative volume.

**Pre-generation and audit.** Generate all shocks, events and news templates for both windows plus their 60-day lookbacks, with a fixed seed, before the pre-registration freeze. Combine them with historical SPY returns, and record SHA-256 hashes of the generated files in the pre-registration.

**Measure detection against the planted signal, not returns.** The analysis knows s and z exactly, so RQ2 correlates each agent's daily score with the planted signal. That removes return noise from the measurement. Over 170 days the standard error is about 0.08 per asset and seed, so pooling across the 3 seeds matters.

## Agent protocol

Every trajectory makes one call per review day and returns per-asset scores, target weights and a next-review interval. Both LLMs answer in validated JSON.

**System prompt (fixed, cached).** The agent manages a paper portfolio of assets resembling US large-cap equities, with identities hidden. It is long-only with a 30% cap per asset; trades fill at the next open with 5 bps cost per side; and every call's inference cost is deducted from its cash. The prompt never mentions synthetic assets (blinding is pre-registered) and never states a goal beyond maximizing net equity.

**User message (daily state).**

- Day index; portfolio (cash, positions, weights, gross equity, cumulative compute cost, net equity); last fills and trading costs.
- Its own last 5 decisions (weights plus market view).
- Per-asset 60-day table (see anonymization), then news per asset, then market headlines.
- REVISE arms only: the current strategy memo.

**Output schema (LLMs).**

```json
{
  "scores": {"ASSET_01": 0.4, "...": 0.0},
  "weights": {"ASSET_01": 0.15, "...": 0.0},
  "cash": 0.25,
  "confidence": 0.6,
  "next_review_in_days": 1,
  "rationale": {"ASSET_01": "max 40 words"},
  "market_view": "max 60 words"
}
```

- `scores`: every asset, in \[−1, 1\], defined as expected performance relative to the universe over the next 5 trading days. These drive the IC and detection metrics.
- `rationale`: required for every asset with weight above 0 or absolute score above 0.5. This is the text RQ3 annotates.
- `next_review_in_days`: one of 1, 2, 3 or 5. While asleep the trajectory makes no calls, pays nothing and holds its shares. This makes saving compute a real choice and "retreat" an observable behavior.

**Validation.** Use OpenRouter structured outputs (`response_format` with a strict JSON schema) where the model supports it, otherwise parse and validate with Pydantic. Weights plus cash must sum to 1 ± 0.01 (renormalize within tolerance) and respect the cap. On failure, retry up to 2 times with the validation error appended; then hold previous weights and log `invalid`. Retries are charged to the trajectory.

**Model settings.** Set `reasoning.effort = "medium"` for both LLMs, leave temperature at the provider default, set `max_tokens` to 16,000, and request `usage` so each response carries its cost. Record reasoning-token counts and, where returned, reasoning summaries.

**Logging.** Store every request and response verbatim (JSONL), with model slug, latency, tokens, `usage.cost`, retry count and a state hash.

## Experimental arms and seeds

Three arms give 14 trajectories, 7 per LLM. A seed is an independent trajectory with its own asset-label permutation; it is not an API seed parameter.

| Arm | Universe | Labels and news | Strategy memo | Models | Seeds | Serves |
| --- | --- | --- | --- | --- | --- | --- |
| BLIND-FULL | 10 real + 10 synthetic | Asset labels, scrubbed news | None | Both | 3 | RQ1, RQ2, RQ3, RQ4 |
| REVISE | 10 real + 10 synthetic | Asset labels, scrubbed news | Rewritten every 10 trading days from true P&L | Both | 2 | RQ5 |
| REVISE-SHUFFLED | 10 real + 10 synthetic | Asset labels, scrubbed news | Rewritten every 10 trading days from shuffled P&L | Both | 2 | RQ5 control |

BLIND-FULL is the primary arm for skill, detection, fallback behavior and survival. Every agent sees the same 20-asset universe, so all decision calls share one prompt size.

**Non-LLM comparators, run free inside the simulator on each universe:** 1,000 random agents (daily Dirichlet weights, with sleep intervals drawn from the LLMs' pooled distribution), equal-weight rebalanced daily, equal-weight buy-and-hold, 20-day momentum top 3, 5-day reversal top 3, and a ridge model on 5-day and 20-day returns plus news count, refit monthly on past data.

## Self-funding accounting

Every trajectory starts with USD 1,000 of paper cash, and the exact OpenRouter `usage.cost` of each of its calls is debited from that cash at call time. The agent sees cumulative compute cost and net equity in every prompt.

- **Charged:** decision calls, their retries, and REVISE memo rewrites.
- **Not charged:** infrastructure calls (scrubber, generator, re-identification classifier, weekly probe, RQ3 judge). These are logged to a separate project ledger.
- **Cost can force selling.** If cash falls below the next expected call cost, the simulator sells pro rata at the next open to raise it, and logs the forced sale.
- **Ruin.** Net equity at or below zero ends the trajectory; record the ruin day. At this capital, ruin is unlikely within 170 days, so survival is reported mainly as compute drag and break-even capital.

**Post-hoc capital sweep.** Compute costs are a fixed dollar stream, so net equity at other capital levels (USD 250, 500, 5,000) can be recomputed from the same decisions. Report this as a sensitivity analysis, noting that agents saw only the USD 1,000 framing.

**Break-even capital** is the starting capital at which compute exactly eats the gross excess return over equal-weight:

```latex
K^{*} = \frac{C_{\text{total}}}{R_{\text{gross}} - R_{\text{EW}}}
```

It is undefined (reported as "never breaks even") when gross excess return is zero or negative.

## Self-revision loop (RQ5)

Every 10 trading days each REVISE trajectory rewrites its own strategy memo from a performance report, giving 17 revision rounds in the window. The control arm gets the same report with credit assignment scrambled.

1. **Report.** For the last 10 trading days: total P&L, compute cost and net return (always true), plus a per-asset, per-day attribution table showing average weight, return contribution and hit rate.
2. **Shuffle (control only).** Permute the attribution table's asset labels and day order, keeping the true total. The agent still learns how much it made, but not which assets or days made it.
3. **Rewrite call.** Input: current memo, the report, and the last 10 decisions. Output JSON: `memo` (max 300 words) and `changes` (max 80 words). The new memo appears in every later decision prompt. The call is charged to the trajectory.
4. **Snapshot.** Store each memo version with its round number, for text analysis and for splitting RQ2 and RQ3 metrics by round.

Known limitation, to state in the paper: shuffled agents can still see true prices in their daily prompts and could reconstruct attribution themselves. The manipulation weakens the explicit learning signal rather than removing all information, so RQ5 tests whether agents use the feedback they are handed.

## Metrics and analysis

The pre-registered primary metric is mean daily rank IC of scores against next-day open-to-open returns, on the real assets in BLIND-FULL. Everything else is secondary. With only 10 real assets, that IC has a standard error of about 0.02 across 170 days and 3 seeds, so only effects above about 0.045 are detectable; most statistical weight sits on the synthetic assets, where truth is known.

| RQ | Main metric | Inference |
| --- | --- | --- |
| RQ1 | Rank IC (1-day and 5-day), top-3 minus bottom-3 spread, net Sharpe vs. equal-weight | Stationary block bootstrap over days (block 5); percentile against the 1,000 random agents; Model Confidence Set across models; Holm correction |
| RQ2 | Correlation of score with the planted signal, per strength level; detection threshold = lowest level significantly above zero | Bootstrap over days, pooled across seeds; logistic "psychometric" fit over strength |
| RQ3 | Behavior profile on pure-noise assets and after drawdowns of 3% or more; confabulation rate | Mixed-effects regressions with trajectory random effects |
| RQ4 | Net equity, compute drag (cost ÷ capital), break-even capital, sleep share, tokens per decision | Descriptive with bootstrap intervals |
| RQ5 | Change in RQ2 detection and RQ3 profile per revision round, REVISE vs. REVISE-SHUFFLED | Difference-in-differences across rounds |

**Behavior taxonomy (RQ3), coded from actions with thresholds fixed in the pre-registration:**

- **Retreat:** cash at or above 70%, or a sleep of 3 or more days, within 5 days of a drawdown.
- **Index reversion:** weights within 0.1 total-variation distance of equal-weight.
- **Momentum chasing:** weight changes positively correlated with the last 5-day return.
- **Gambler's fallacy:** weight increases after 3 or more same-sign down days.
- **Doubling down:** gross exposure rises after a drawdown.
- **Overtrading:** daily turnover above twice the equal-weight benchmark's.

**Confabulation rate:** the share of rationales for pure-noise assets that make a specific causal claim, rather than acknowledging uncertainty or the absence of signal. DeepSeek V4 Flash labels every rationale as judge. Validate it against 200 human-labeled rationales and require Cohen's kappa of at least 0.7 before using the labels.

## Pilot and validation gates

The test window is replayed only after all seven gates pass. Every gate runs on the development window (December 1, 2025 to February 13, 2026), whose results are never reported, so the test window stays unseen until its single run.

| Gate | Test | Pass criterion |
| --- | --- | --- |
| G0 Post-cutoff check | Ask each model about 20 verifiable test-window events (index moves, earnings outcomes, major company news) mixed with 20 fabricated but plausible events | Accuracy on real events within 10 points of the fabricated-event baseline; a clear hit blocks that model |
| G1 Indistinguishable prices | Gradient-boosted classifier on about 30 stylized-fact features (volatility clustering, kurtosis, autocorrelation, beta, volume and news-count statistics), real vs. synthetic | Cross-validated AUC at or below 0.60 |
| G2 Indistinguishable to the models | Weekly probe, both models: real vs. synthetic | Accuracy within 10 points of chance |
| G3 Anonymization | Alias regex on all scrubbed text; weekly probe asking both models to name each real asset | Zero regex hits; probe re-identification at or below 20% (chance is 10%) |
| G4 Planted signal | 500 simulated paths per synthetic asset | Realized strength within ±0.02 of target |
| G5 Output validity | Development-window decisions, both models | At least 98% valid on the first try |
| G6 Dress rehearsal | Full replay of the development window, all arms, no manual fixes | All artifacts written; cost within ±25% of estimate |

If G1 or G2 fails, adjust the generator (donor pool, block length, news counts) and re-run. If G3 fails, tighten the scrubber prompt and alias dictionary. Never relax a criterion after seeing results.

**Pre-registration (OSF), frozen before the test replay starts:** hypotheses H1 to H5, primary and secondary metrics, behavior thresholds, model slugs and settings, the universe and its selection rule, window dates, generator code commit, and SHA-256 hashes of the snapshot and the pre-generated synthetic files. The test replay then runs once.

## Build order and milestones

Build in 10 milestones over about five weeks, one pull request each with its own tests, then clear all gates on the development window and run the test replay once in mid-December. Each milestone has a hard "done when" check, so Claude Code can work through them in order.

&#91;embedded content: build and run timeline · 10 milestones\]

The test replay takes about a day of compute, so analysis can start as soon as it finishes.

| Milestone | Deliverable | Done when |
| --- | --- | --- |
| M0 Scaffold | Repo, config loader, secrets from environment, structured logging, DuckDB and Parquet store, exchange calendar | `dlo --help` runs; CI runs tests |
| M1 Clients | OpenRouter chat and Decisions clients with retries, rate limits, cost capture; model registry with pinned slugs | Smoke test returns valid output and `usage.cost` from both models and DeepSeek V4 Flash |
| M2 Snapshot | Universe selection from the pre-window SPY disclosure; frozen bars, news and donor pool; validation | Snapshot hashed; poison-row look-ahead test passes |
| M3 Synthetic | Bootstrap, signals, candles, sector-flavored news generator, pre-generation with hashes | G1 and G4 pass on the development window |
| M4 Anonymizer | Scrubber, alias dictionaries, regex leak check, entailment check, weekly probe | G3 passes |
| M5 Agents | State builder, prompts, schema, LLM agent | G5 passes; every state under 30 KB |
| M6 Simulator | Execution, ledger, sleep, ruin, baselines, random agents | Hand-computed P&L on a 5-day toy case matches exactly |
| M7 Replay runner | Day loop, checkpoint and resume, run log | 5 consecutive development days replayed unattended |
| M8 Revise loop | Report builder, shuffle, memo rewrite, snapshots | 2 rounds on the development window, memos stored |
| M9 Analysis | Metrics, bootstrap, Model Confidence Set, behavior coding, judge, G0 cutoff check | Full report generated from development-window data; judge kappa at least 0.7 |

## Budget estimate

Plan for about USD 260 in API spend over the whole project, including a 20% margin. GPT-5.6 Terra is about 90% of decision cost; Luna and DeepSeek V4 Flash together cost under USD 25. Paper capital is notional and costs nothing.

| Item | Estimate (USD) | Basis |
| --- | --- | --- |
| Test-window decision calls, 170 days | 115 | Terra 105, Luna 10; assumes no sleep days |
| Memo rewrites and retries | 8 | 17 rounds × 4 trajectories × 2 LLMs; about 3% retries |
| Weekly probes and G0 cutoff check | 18 | 34 simulated weeks × 40 calls, plus 80 G0 questions |
| DeepSeek V4 Flash, all infrastructure jobs | 12 | About 40M tokens across both windows |
| Development window: gates, dress rehearsal, debugging | 65 | 50-day dress rehearsal (about 34) plus development calls |
| Hosting | 0 | A laptop or any small VM is enough |
| **Total, including about 20% margin** | **260** |  |

Token assumptions per decision: about 20,000 input and 4,000 output tokens (reasoning included); every arm uses the 20-asset universe. At list prices that is about USD 0.09 per call for Terra and under 0.01 for Luna. If reasoning doubles average output, test-window decision calls rise to about USD 180, so G6 re-checks these numbers against real `usage.cost`.

## Risks and open items

The biggest threat in a historical replay is contamination. Both evaluated models were released on July 9, 2026, in the middle of the test window, so the guarantee rests on their February 16 cutoff; the two-week buffer and gate G0 guard it. Pipeline leakage and learnable synthetic assets are the next threats.

**Risks and mitigations**

- **Look-ahead inside the pipeline.** A unit test injects "poison" rows timestamped after the simulated day's cutoff and fails if any reach a prompt. News is filtered by publication time, bars by session date.
- **Hindsight from infrastructure models.** DeepSeek V4 Flash (released April 24, 2026, cutoff unpublished) may know about early-window events. Restrict the scrubber to rephrasing, and run an entailment check that flags any rewritten sentence not supported by its source item.
- **Forking paths.** Any tuning on the test window would invalidate it. All tuning happens on the development window; the test replay runs once, and reruns are logged and reported.
- **Silent model changes.** Pin slugs, log the serving provider returned on every call, and disable OpenRouter fallbacks to hosts serving different weights. Running the test replay in one sitting keeps this risk small.
- **API failures.** Retry with backoff; a call that still fails holds prior weights and is logged.
- **Corporate actions.** Use adjusted bars and derive the index factor from adjusted data, so splits never create jumps.
- **Synthetic assets learned over time.** The weekly probe tracks real-vs-synthetic accuracy by simulated week. Pre-register a robustness analysis that drops any asset a model identifies above chance.
- **Frozen universe.** The 10 names never change during the window.

**Open items to confirm during the build**

- [ ] Alpaca News API still free on the Basic plan; Benzinga terms allow storing headlines and summaries for research (M2)
- [ ] Sector pairs confirmed from the last SPY holdings disclosure before March 2, 2026 (M2)
- [ ] A dated DeepSeek V4 Flash snapshot pinned, preferring the earliest available (M1)
- [ ] `reasoning.effort` honored for both LLMs through OpenRouter (M1 smoke test)
- [ ] Ethics note for the paper: paper trading only, no orders sent, blinding of agents to synthetic assets pre-registered
