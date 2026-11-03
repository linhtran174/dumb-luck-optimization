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

## Anonymization

- Alias dictionaries were built with DeepSeek, then hand-reviewed. Removed: countries, basins, index names, media
  outlets, commentators and generic concepts, which would flag macro headlines and other companies' news. NVDA's
  ~200 "partners" (every co-mentioned entity) were replaced with a curated list.
- Regex matching is case-sensitive for proper nouns ("Chase" vs "chase", "COST" vs "cost").
- The entailment call also lists leftover proper names (people, firms, products). Any name triggers the strict
  re-scrub; a second hit drops the item. Government bodies and countries are allowed.
- Scrubbed text stores companies as `[[KEY]]` tokens, mapped to per-trajectory labels only at render time.
- Result for both windows: 4,392 selected items, 2,945 kept, 1,230 dropped as uninformative (listicles, options
  activity), 159 failed entailment, 58 leaked twice. DeepSeek cost for this was about $0.30.
- Known residual leak no regex can catch: semantic giveaways such as a bank CEO's widely reported remarks. The
  weekly probe (G3) measures this.

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
- G3: zero regex hits in accepted text (pass). Probe re-identification is **80% (Luna) and 70% (Terra)**, n=10
  each, against a 20% threshold (fail). Both models name the right company for AAPL, COST, CVX, JNJ, LLY, NVDA
  and WMT. They also assign sector-matching real names to most synthetic assets. Pairs don't hide identity,
  because business-model cues in scrubbed news (membership, weight-loss drugs, GPUs, devices) separate the two
  companies in each sector. Caveat: the trial is in December 2025, before both models' cutoff, so remembered
  price paths may inflate G2 and G3 relative to the test window.
- G5: 82.8% first-try valid in trial-v1 (Terra 71%, Luna 93%). After adding a self-check line to the prompt,
  92.6% in trial-v2-g5 (27 decisions). The remaining failures are all weights plus cash summing to 1.03-1.12.
  Recommended fix (not applied; it changes the output protocol): drop `cash` from the schema and derive it as
  1 minus the weights.
- States: max 24.0 KB, mean 22.2 KB (pass).
- Judge kappa: needs at least 200 human-labeled rationales: `dlo analyze export-judge trial-v1`.

## Trial cost (actuals)

- trial-v1 (14 trajectories, 5 days, 29 decisions + 16 memos): $0.94. trial-v2-g5: $0.85.
- Terra is about $0.047 per decision (about 15k prompt tokens, 1k output); Luna is about $0.005.
- Infrastructure through the trial: $0.83 (scrub both windows, aliases, synthetic news, 40 probes, G0 twice,
  judge).
- Projected test window: 7 Terra trajectories x 170 days x $0.047 is about $56 with no sleep. Agents slept 60%
  of trial days, so realistically about $25 for Terra and $3 for Luna, plus memos and probes (about $15).
