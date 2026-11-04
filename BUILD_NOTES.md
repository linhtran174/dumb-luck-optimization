# Build notes: decisions and deviations from the plan

Written during the unattended build (Nov 3-4, 2026). Each entry says what changed and why.

## Scope agreed with the principal

- Gates run on a **5-day trial** (Dec 1-5, 2025) instead of the 50-day dress rehearsal (G6). The trial runs all
  14 trajectories. It uses `revise_every_days: 2` (trial only) so the revise loop gets 2 rounds inside 5 days.
- **No OSF pre-registration.** Settings, hashes and the commit are recorded in the repo and the run store instead.
- The unattended session stops after the trial. The test window is never started automatically; the CLI refuses
  without `--i-have-frozen-the-preregistration`.

## Data and universe

- **Universe pairs from the SPY N-PORT-P** filed 2026-02-26 (holdings as of 2025-12-31), the last disclosure
  before Mar 2, 2026. Two pairs differ from the provisional list: IT is NVDA + AAPL (MSFT ranks 3rd), Financials
  is BRK.B + JPM (V ranks 3rd). Fetched via Firecrawl because SEC requires a contact email in the User-Agent.
- Alpaca **requires API keys** for stock bars and news (HTTP 401 without them).
- Alpaca news tags Berkshire as `BRK` (sometimes `BRK/B`); bars use `BRK.B`. Handled in `clients/alpaca.py`.
- Snapshot `bars_end` is Nov 2, 2026: the free plan blocks recent SIP data and Nov 3 was still trading. The last
  test-window decision (Oct 30) fills Nov 2; its open-to-open forward return (Nov 2 to Nov 3) is unavailable, so
  that one day drops out of IC metrics.

## Models

- Pinned slugs work as sent: `openai/gpt-5.6-terra-20260709`, `openai/gpt-5.6-luna-20260709` (provider OpenAI,
  no fallbacks), and `deepseek/deepseek-v4-flash-20260423`, the earliest dated DeepSeek snapshot.
- DeepSeek runs with `reasoning.enabled=false`. StreamLake was removed from its providers: it spent the whole
  token budget on hidden reasoning and returned empty content.

## Anonymization (scrub v2, Nov 4)

v1 (used by trial-v1) failed G3 at 70-80%: the scrubbed text still carried sector and business-model detail
(weight-loss drugs, memberships, dairy plants, offshore basins). v2 replaced it before the test run:

- **Industry-neutral rewrite.** Products, industry metrics, industry regulators/events, places and function-specific
  roles become generic business terms. The company's own share-price moves are removed (agents have the prices).
  Market items carry no company tokens at all. Ids are versioned (`v2:...`); v1 rows stay in the store for audit.
- **Per-item attacker.** DeepSeek V4 Flash with reasoning (`deepseek_r`, same pinned snapshot) sees only the rewrite
  and gives a probability per industry plus a company guess. A leak is p >= 0.40 on the true industry (chance 0.20)
  or p >= 0.25 on the right company; a leak triggers the strict re-scrub, a second one drops the item. Attack-call
  errors count as leaks (14 items in the dev window).
- **Synthetic news goes through the same scrubber and checks.** It is written by mirroring randomly drawn real v2
  items from the dev window only (same kind and direction, new specifics), so real and synthetic share one style,
  and test-window news is never shown early. The plan's "neutral filler in the sector's voice" is dropped: sector
  voice would mark synthetic assets once real news is industry-neutral. Template direction is independent of the
  synthetic asset's future returns, so no signal is planted. Dropped synthetic items are regenerated (3 rounds).
- Result: dev window 1,061 items, 581 kept (55%); test window 3,331 items, 1,799 kept (54%). v1 kept about 67%.
- Synthetic news counts were re-matched to the v2 post-scrub real distribution (`dlo synth pregen`). Prices,
  events, sentiments and signals are byte-identical to before (verified); only `news_count` changed. G1 and G4 are
  unchanged.
- Alias regex, generic date/amount patterns and the entailment + leftover-name check are unchanged.

### G3 outcome: failed, leak accepted by the principal

`dlo gate g3` probes many weeks outside any trajectory (variants: full, data-only, news-only).

| Probe | Luna re-ID | Terra re-ID | Real-vs-synth accuracy (Luna / Terra) |
|---|---|---|---|
| trial-v1, v1 scrub, 1 week | 80% | 70% | 50% / 80% |
| dev window, v2, 6 weeks | 28% | 55% | 61% / 63% |
| test window, v2, 4 weeks (post-cutoff) | 40% | 52% | 66% / 72% |
| dev window, prices only, 6 weeks | 10% | 13% | n/a |

Prices alone are near chance except NVDA (83% for both). The remaining leak is the models' knowledge of these ten
megacaps' storylines (direct-to-consumer drug pricing, chip export limits, project stakes, off-cycle earnings), which
industry-neutral wording cannot remove. It persists after the Feb 16, 2026 cutoff. Further scrubbing would mainly
discard news. The principal chose to **accept the leak** (Nov 4) over changing the universe or removing company news.
G3 and G2 are reported as failed; the criteria were not relaxed.

**Pre-registered robustness rule** (`metrics.identified_assets`, fixed before the test run): per model, an asset is
dropped from the robustness analysis when that model's weekly probes identify it above chance (one-sided binomial,
5%): a real asset named correctly above 10%, a synthetic one called synthetic above 50%. The report shows IC and
planted-signal detection with and without the dropped assets (IC needs at least 5 real assets left). On the 4
test-window probe weeks, both models would drop AAPL, LLY and NVDA; 34 weeks give more power, so expect more.

## Synthetic generator (M3)

- Donor residuals are **pre-whitened with AR(5)** before the block bootstrap. Raw residuals carried about -0.03
  short-term reversal, which planted an unintended signal in the "pure noise" assets and failed G4.
- The momentum signal is divided by its exact stationary scale c(a), so corr(eps, s) = a. Without this it
  overshoots, because eps feeds back into s.
- G4 measures strength as the uncentered correlation pooled over 500 paths. The process has mean zero by
  construction; per-path demeaned correlations are biased by about -5/n (about -0.02).
- Calibrated against G1 on pre-test data only (Jan 2024 to the dev window end): block length 20 (plan: 10).
  beta/sigma come from one real company of the sector, with jitter, instead of the pair average. Borrowed candles
  are rescaled to the asset's volatility. Range and volume scale with the synthetic day's move (elasticities
  0.5 and 0.3), and log relative volume is shrunk by 0.92. The plan's candle rule is the special case where
  both elasticities are 0.
- News counts for synthetic assets match the **post-scrub** distribution of real asset-days (what agents
  actually see), with event days forced to at least one item and the marginal kept exact.

## Agent state

- The full 5-metric, 60-day table was 28.6 KB on its own, so states would exceed 30 KB. ret% and indexed close
  (rounded) keep 60 days; gap%, rng% and rvol show the last 20 days. States are now about 21-23 KB, including a
  300-word memo.
- Agents see only the current day's news, even after sleeping.

## Gate status (see data/derived/gates/*.json and the trial report)

- G0 pass: both models answer "don't know" to all 40 statements. In a forced yes/no variant, the yes rate is
  0% vs 0% (Terra) and 5% vs 10% (Luna) on real vs fabricated events.
- G1 **fail**: AUC 0.626 against a 0.60 threshold, after calibration. No single dominant feature remains; the
  top separators are mild (univariate AUC 0.57-0.62).
- G4 pass: all ten assets within +/-0.011 of target.
- G2 (probe, real vs synthetic): Luna 50% (pass), Terra **80% (fail)**, n=20 each.
- G3: zero regex hits in accepted text (pass). Probe re-identification **fails** (trial-v1: 80% Luna, 70% Terra;
  after scrub v2: 40% and 52% on test-window weeks). Leak accepted by the principal; see Anonymization above.
- G2 after scrub v2: 66% (Luna) and 72% (Terra) on test-window weeks, against 50% +/- 10 (fail).
- G5: 82.8% first-try valid in trial-v1 (Terra 71%, Luna 93%); 92.6% in trial-v2-g5 after a prompt self-check.
  Then (principal-approved protocol change) `cash` was **removed from the output**: it is derived as 1 minus the
  asset weights, and the only sum rule is weights <= 1 + 0.01. Re-validating all 56 earlier first-try responses
  offline: 96.4%. Confirmation trial trial-v3-g5 (28 decisions, v2 news): **96.4% (27/28; Terra 100%, Luna 93%)**,
  still below 98%. The one failure is a missing rationale for an asset with |score| > 0.5; the charged retry fixed it.
  At n=28 a single failure fails a 98% gate. Not relaxed. The principal accepted G5 at 96% (Nov 4): no further
  protocol change; the charged retry handles the rare missing rationale.
- G1 and G2 accepted as failed by the principal (Nov 4) and the test run launched: synthetic price realism
  (AUC 0.626) and the real-vs-synthetic probe (66% / 72%) are reported as failed; criteria not relaxed. The
  synthetic-asset results (planted-signal detection, noise confabulation) carry that caveat.
- States: max 24.0 KB, mean 22.2 KB (pass).
- Judge kappa: needs at least 200 human-labeled rationales: `dlo analyze export-judge trial-v1`.

## Trial cost (actuals)

- trial-v1 (14 trajectories, 5 days, 29 decisions + 16 memos): $0.94. trial-v2-g5: $0.85.
- Terra is about $0.047 per decision (about 15k prompt tokens, 1k output); Luna is about $0.005.
- Scrub v2 work (both windows re-scrubbed, synthetic news, 680 probe calls): about $7.50 infra.
- Infrastructure through the trial: $0.83 (scrub both windows, aliases, synthetic news, 40 probes, G0 twice,
  judge).
- Projected test window: 7 Terra trajectories x 170 days x $0.047 is about $56 with no sleep. Agents slept 60%
  of trial days, so realistically about $25 for Terra and $3 for Luna, plus memos and probes (about $15).
