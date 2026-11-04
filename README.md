# dumb-luck-optimization (dlo-bench)

Paper-traded, post-cutoff LLM trading benchmark: GPT-5.6 Terra and Luna trade 10 anonymized real S&P 500
stocks plus 10 synthetic assets with planted signals, one simulated day at a time, paying their own inference
costs. See `Dumb Luck Optimization Implementation Plan.md` for the design.

No brokerage order is ever sent. Alpaca is used only for read-only market data and news.

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
# .env needs OPENROUTER_API_KEY, ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY
.venv/bin/pytest -q
```

## Every LLM call is persisted

`data/db/llm_calls.sqlite` (mirrored to `data/logs/llm_calls.jsonl`) stores every request and response verbatim,
written before the caller sees it. Successful responses are reused for identical requests, so a crash, resume
or rerun never pays twice. Hard spend caps per bucket live in `config/costs.yaml` and are checked against this
ledger before every uncached call. `dlo ledger` prints spend by bucket, purpose and model.

Back up `data/` (it is gitignored): it holds the frozen snapshot, pre-generated synthetic data, scrubbed news,
run state and the LLM ledger.

## Pipeline (already run once; all steps are idempotent)

```bash
dlo smoke                      # M1: one structured call per pinned model
dlo snapshot fetch             # M2: frozen Alpaca bars + news (refuses to overwrite)
dlo snapshot validate
dlo anon build-aliases         # M4: alias dictionaries (then hand-reviewed and frozen in config/aliases/)
dlo anon scrub all             # scrub every selected news item of both windows (DeepSeek)
dlo synth pregen               # M3: synthetic assets for both windows, hashed
dlo synth news all             # synthetic news text, generated once
dlo gate g1 | g4 | g0 [--forced]
dlo gate g3 --window dev --weeks 6 --variant full|data|news   # multi-week re-identification probe
dlo run window trial --run-id trial-v1      # 5-day trial, all 14 trajectories
dlo analyze report trial-v1
```

## The real run (test window, Mar 2 to Oct 30, 2026)

Runs once. It refuses to start without an explicit flag:

```bash
dlo run window test --run-id test-v1 --i-have-frozen-the-preregistration
dlo analyze report test-v1
```

If it stops for any reason (network, budget cap, crash), rerun the same command: it resumes after the last
completed day and answers repeated requests from the ledger. Raise `spend_caps_usd.test` in
`config/costs.yaml` only deliberately.

## Layout

```text
config/   models.yaml (pinned slugs), universe.yaml, windows.yaml, arms.yaml, costs.yaml, aliases/*.yaml
dlo/      clients/ data/ synth/ anon/ agents/ sim/ orchestrator/ analysis/  (cli.py, commands.py)
tests/    look-ahead poison test, hand-computed P&L, schema, ledger cache/budget
data/     raw/snapshot, pregen, db, derived (gates, reports), logs   (gitignored)
```

See `BUILD_NOTES.md` for decisions and deviations from the plan made during the build.
