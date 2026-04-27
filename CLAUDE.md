# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Backtest — compares strategies (factor_only, factor_macd, factor_kdj, factor_decision_tree, factor_xgboost, factor_rl, factor_hybrid)
python main.py backtest
python main.py backtest --strategy factor_hybrid       # single strategy, faster
python main.py backtest --factor-module both           # alpha101 + alpha158
python main.py backtest --review                       # auto-run codex_review after
python main.py backtest --llm-filter mock              # apply rule-based LLM filter (matches live)
python main.py backtest --llm-filter real              # apply actual Claude API (slow, expensive)

# Live trading (KIS virtual/real, config/.env required)
python main.py live

# ML training / retraining
python main.py train --model xgboost                   # options: decision_tree, xgboost, lightgbm, lstm, transformer, rl
python main.py retrain --model xgboost                 # replaces only if F1 improves

# Account status
python main.py status

# Grid search: Hybrid buy_threshold (uses pool cache if present)
python scripts/threshold_sweep.py

# PIT universe metadata bootstrap (run once, then monthly)
python scripts/build_universe_meta.py            # produces data/universe_meta.csv (~5,700 rows)
python scripts/build_universe_meta.py --force    # rebuild even if file exists

# SAPPO pipeline
python scripts/fetch_daily_news.py                     # today's news + sentiment (cron 16:00)
python scripts/fetch_daily_news.py --date 20260419     # backfill
python scripts/fetch_daily_news.py --codes 005930 035420
python scripts/weekly_sappo_report.py --slack          # weekly IC / training-run comparison (cron Sat 07:00)

# Codex CLI review pipeline (3 modes)
python scripts/codex_review.py daily                   # today's trading results → reports/codex_daily_YYYY-MM-DD.md
python scripts/codex_review.py backtest                # latest backtest log
python scripts/codex_review.py code --base main        # diff vs main branch

# LLM decision IC analysis
python scripts/llm_ic_analysis.py --forward 5 --days 30

# Alpha101 vs Alpha158 A/B (factor_only backtest each, side by side)
python scripts/compare_factors.py

# Windows Python via WSL — always use this executable
/mnt/e/SuperTrader/venv/Scripts/python.exe
```

Primary config: `config/settings.yaml` (+ `config/.env` for KIS / Anthropic / Slack / Notion keys).

No test suite exists (`tests/` directory is empty).

## Architecture

SuperTrader is a Korean stock (KOSPI/KOSDAQ) auto-trading system combining **factor-based stock selection**, **ML/RL timing signals**, **LLM signal validation (with rule-based post-check)**, and **news-based SAPPO sentiment augmentation** for RL retraining.

### Five execution modes
- **backtest**: Runs strategies on historical data with optional LLM filter
- **live**: Scheduled auto-trading via KIS broker API
- **train**: Fits a fresh timing model from scratch
- **retrain**: Trains a candidate, replaces the live model only if F1 improves (auto-backup)
- **status**: Account holdings and P&L

### Live Trading Pipeline

```
startup:
  _restore_pool_from_disk()  → reads data/current_pool.json, skip rebalance if ≥10 stocks
    └─ fallback: rebalance_pool() builds 30-stock pool via factor scoring

every 5 min (market hours):
  get_balance() → sync held_codes
  stop_loss check (LLM-free, immediate)
  force-sell pool exits (held_codes - pool_codes)
  for each stock in pool:
    strategy.generate_signal() → BUY/SELL/HOLD
    if actionable: LLM validate → rule post-check (rule wins if disagree)
    if confirmed: RiskManager.validate_order → OrderManager → KIS API
  check_filled() → emit ✅ confirmation after ~5min
  save_daily_pnl snapshot

daily 15:40: daily_report (Slack + Notion)
daily 16:00: SAPPO fetch (news + sentiment)
monthly day=1 08:30: rebalance_pool
weekly Sat 06:00: ML retrain
weekly Sat 07:00: SAPPO weekly report (Slack)
```

### Data Pipeline (backtest mode — uses same factor pipeline as live)

```
market_data.get_universe()                  → KOSPI stocks filtered by cap/volume (+ fallback if <30 result)
market_data.get_ohlcv_batch()              → {code: OHLCV DataFrame}
market_data.filter_by_listing_date(start)   → drops post-IPO stocks (partial survivorship fix)
    ↓
main._build_pool_history_factor_based()
    pool cache hit?  → load data/pool_cache/pool_<hash>.json (config-hash keyed)
    else:
      per rebalance date:
        compute factors → neutralize → rolling IC report (ic_lookback=12) → top-30
      save cache
    ↓
main._train_models_if_needed()              → walk-forward prefix (train_ratio=0.5) if models missing
    ↓
comparison.run_strategy_comparison()        → per strategy in PortfolioBacktestEngine
    ↓
report.print_comparison_table()
```

### Strategy Layer (`src/strategy/`)

All strategies inherit `BaseStrategy` and implement `generate_signal(code, df) → TradeSignal`.

- **FactorOnlyStrategy**: `commit_pool()` + `update_pool()` — rebalance-only. Engine triggers auto BUY on new entries.
- **FactorMACDStrategy**, **FactorKDJStrategy**: Classic technical signals.
- **FactorMLStrategy**: Wraps any ML model via `TimingPredictor`.
- **FactorRLStrategy**: PPO agent with business-day holding tracking. Live `sync_positions()` uses DB → chart estimate → today fallback for entry dates.
- **FactorHybridStrategy**: XGBoost (alpha) + RL (timing/risk). BUY requires both; SELL on XGBoost alone. As of 2026-04-20: **+63.3% return, Sharpe 0.45, MDD -21.8%** with `buy_action_threshold=0.07`. These figures came from `scripts/threshold_sweep.py` against the universe at that time — re-run the sweep if the universe, factor module, or RL model changes before trusting them.

**Important**: `FactorRLStrategy` and `FactorHybridStrategy` accept `current_date` kwarg in `generate_signal()` and `sync_positions()` (with `entry_dates` dict) so backtest uses correct holding days. Live mode omits these and falls back to `datetime.now()`.

### Pool Exit Handling

- **Backtest engine** (`portfolio_engine.py:216-237`): `old_set - new_set` force-SELL always runs (not gated on `commit_pool`). Only new-entry auto-BUY is restricted to `commit_pool` strategies (FactorOnly).
- **Live `check_signals()`**: `held_codes - pool_codes` get force-SELL with `signal_type="pool_exit"` logged, `save_holding` removed. The "not in self._pool → SELL" branch in non-FactorOnly strategies is intentionally absent — pool-exit selling is handled centrally by the engine and live loop, not per-strategy.

### Timing Models (`src/timing/`)

- **features.py**: ~30 features from OHLCV (RSI, MACD, KDJ, BB, ATR, ADX, MFI + derivatives)
- **labels.py**: Forward N-day returns → 1/-1/0 with ±2% thresholds
- **trainer.py**: Dispatch. RL path loads sentiment_map from DB when `config.timing.rl.sentiment_lambda > 0`
- **predictor.py**: `TimingPredictor(model_type, path).predict(df) → 1/0/-1`
- **retrain.py**: Trains new, compares F1/accuracy, replaces only if improved, backs up old
- **llm_validator.py**:
  - `SignalValidator` — Claude Sonnet 4.6 (`claude-sonnet-4-6`), JSON-schema-ish 확정/보류
  - Prompt: conservative. BUY holds if (RSI≥80 AND 1d≥+7%) OR RSI≥90 OR 1d≥+15% OR 5d≥+25%. SELL holds only if MA20-up AND MA5-up AND RSI≥50 AND 1d≥-1% AND 5d≥+1% (all five). Strict interpretation clause forbids loose phrasing.
  - `apply_rule_check()` post-check: re-evaluates rules deterministically. If LLM disagrees, **rule wins** (logged as WARNING). On LLM API error, falls back to rule.
  - `MockSignalValidator` — same rules, no API call. Used by backtest `--llm-filter mock`.
- **sentiment_generator.py**: Claude Haiku + strict JSON output → `{score: -1~+1, confidence, rationale}`. Stores to `sappo_sentiment_scores`. Cache-control on system prompt for 90% cost cut.
- **rl_env.py**: State(29 features + 3 position), Action(BUY/HOLD/SELL), Reward = DSR + drawdown penalty + holding cost ramp + opportunity weight. **SAPPO**: `reset(df, sentiment_series)` + `step()` adds `sentiment_lambda × sentiment(date)` to reward.
- **rl_agent.py**: PPO Actor-Critic (shared backbone + LayerNorm) + GAE (λ=0.95). GRPO legacy auto-convert. `collect_episode(reset_env=False)` option for SAPPO pre-reset env.
- **rl_trainer.py**: Multi-stock ThreadPoolExecutor rollout → PPO batch update (GPU). Auto-saves training-run record to `sappo_training_runs` DB (lambda, Sharpe, return, MDD, run_name).

### RL Training-Live Alignment

RL trains on daily bars (1 step = 1 trading day). Live checks every 5 min but `holding_days` uses **actual business days** since entry (via `_business_days_held(entry_date, reference_date)`). Backtest passes `current_date` so holding is computed against backtest time, not `datetime.now()`.

**Reward v3 (current)**: `holding_cost=0.0002` + ramp `0.00005/day`, `opportunity_weight=0.35`. v4 (anti-churn) was reverted.

### RL Action Thresholds (`config/settings.yaml` → `timing.rl`)

- `buy_action_threshold: 0.07` — optimized via grid search after 2026-04-19 fixes
- `sell_action_threshold: 0.06`
- `sentiment_lambda: 0.0` — SAPPO off by default, 0.1 when news data ≥6 months accumulated
- `sentiment_source: "off"` — off / news / mock / xgb_proxy

Flow: `factor_rl.py` → `predictor.py` → `rl_agent.py`. Live `sync_positions()` passes `avg_prices` for DB → chart estimate → today fallback buy_date resolution.

### Backtest Engine (`backtest/`)

`PortfolioBacktestEngine.run()`:
1. Rebalance: pool diff → force-SELL exits (always) + auto-BUY new entries (only for `commit_pool` strategies)
2. Stop loss check (T-day close → T+1 open sell)
3. Daily timing signals: `strategy.generate_signal(current_date=date)` → pending_orders
4. Optional `llm_validator` (if passed): applies to each actionable signal
5. `strategy.sync_positions(held, prices, avg_prices, entry_dates, current_date)` at day end

**llm_validator param**: `--llm-filter mock` → `MockSignalValidator` (deterministic, free). `--llm-filter real` → `SignalValidator` (Claude API, ~$5-10/run).

### Factor System (`src/factors/`)

- **alpha101.py** + **alpha158.py** → 220 factors (prefix `a158_` to avoid collision)
- Pipeline: `calculator.py` → `neutralizer.py` (industry/mcap) → `validity.py` (IC analysis) → `composite.py` (IC-weighted or equal) → `stock_pool.py` (top-N)
- **`build_stock_pool(..., return_factors=True)`** returns `(pool, factor_df)` for trainer to accumulate factor_history
- `_build_pool_history_factor_based()` accumulates rolling (factor_history, return_history) and rebuilds `factor_report` every rebalance → enables **actual** IC-weighted composite (was falling back to equal-weight before 2026-04-19 fix)
- **pool_cache.py** (new): `data/pool_cache/pool_<sha1:12>.json` keyed on 14 config fields. Saves ~60min on reruns.

Rebalancing: monthly (day 1). Live restart uses `_restore_pool_from_disk()` to skip rebalance — prevents pool-diff mass trades on restart.

### Database (`src/db/`)

SQLAlchemy + SQLite (`data/trading.db`, WAL mode). Two module sets coexist in same DB:

**Core tables (`src/db/models.py`)**:
- `trade_logs`, `daily_pnl`, `position_logs`, `holding_positions`, `runtime_status`
- `signal_logs` — `signal_type ∈ {llm, summary, pool_exit}`. Used by web dashboard and `llm_ic_analysis.py`.

**SAPPO tables (`src/db/sappo_models.py`, `expire_on_commit=False`)**:
- `sappo_news` — raw articles (stock_code, date, url UNIQUE)
- `sappo_sentiment_scores` — (stock_code, date) PK, score/confidence/rationale
- `sappo_training_runs` — PPO/SAPPO run history (lambda, Sharpe, return, MDD, model_path)
- `sappo_sentiment_ic` — IC measurement snapshots
- `sappo_weekly_metrics` — week-start PK, aggregates for verification reports

`HoldingPosition.buy_date` (YYYYMMDD) solves holding_days=0 bug on restart.

### Web Dashboard (`web/`)

```bash
/mnt/e/SuperTrader/venv/Scripts/python.exe web/app.py
```

Sections: Summary cards, Cumulative Return chart, Holdings table (click for indicators), Factor Pool (30 stocks), Recent Signals (RL + LLM).

API: `/api/status`, `/api/pool` (reads `data/current_pool.json` with encoding fallback), `/api/signals`, `/api/returns`, `/api/indicators/<code>`.

### Live Trading (`src/broker/`, `src/risk/`, `src/notification/`)

- **kis_client.py**: KIS REST API, virtual/real switch via `is_virtual`. 10 calls/sec rate limit.
- **order.py**: `Order.reference_price` (signal-time price) for market-order notification display. `check_filled()` updates filled_qty/price post-fill.
- **account.py**: `AccountSummary` with `total_deposit` (D+2) and `available_cash` (unsettled-aware).
- **risk/manager.py**: 5%/position, -5% daily loss limit, -7% stop loss, kill switch (3 consecutive errors).
- **notification/slack_bot.py**: **Dual channels** — `#supertrader` (system/reports/feedback), `#super_trader_buy_sell` (signals/fills/fails/stop-loss). `SLACK_TRADE_CHANNEL` env override supported.
- **notification/notion_reporter.py**: Daily trading log to Notion DB.

### Live Trading Schedules (APScheduler, registered in `main.py::run_live`)

- Signal check: every `check_interval_sec` (300s) from market_open (09:00)
- Daily report: 15:40
- **Daily SAPPO fetch: 16:00** → `scripts/fetch_daily_news.py`
- Monthly rebalance: day 1 at pre_market (08:30)
- Weekly retrain: Saturday 06:00
- **Weekly SAPPO report: Saturday 07:00** → `scripts/weekly_sappo_report.py --slack`

### SAPPO (Sentiment-Augmented PPO) Pipeline

External news → LLM sentiment scalar → RL training reward (`reward += λ × sentiment(code, date)`). Preserves XGBoost↔RL peer structure; sentiment is external info (no circularity). Inference doesn't use sentiment.

**Data flow**: Google News RSS → `sappo_news` → Claude Haiku → `sappo_sentiment_scores` → `TradingEnv.reset(sentiment_series=)` during training.

**Activation conditions** (until met, keep `sentiment_lambda=0.0`):
1. ≥6 months of daily sentiment per stock accumulated
2. `sappo_weekly_metrics.sentiment_ic_5d > 0` with p < 0.05
3. Baseline (λ=0) Sharpe recorded in `sappo_training_runs` for comparison

Then: set `sentiment_lambda=0.1, sentiment_source="news"` → `python main.py train --model rl` → compare new run's Sharpe vs baseline.

### Point-in-Time Universe

Universe membership is reconstructed per-date from `data/universe_meta.csv` (built by `scripts/build_universe_meta.py` from FDR `KRX-DESC` + `KRX-DELISTING`). This file is the source of truth for who was tradable when, including delisted stocks.

- `get_universe(date, ohlcv_dict=None)` in `src/data/market_data.py`:
  - `date is None` → legacy path (today's FDR `StockListing` snapshot). Live mode no-op.
  - `date` set → filters meta by `listed_date <= date AND (delisted_date is NaT OR delisted_date >= date)`.
  - Volume filter uses `ohlcv_dict[code].volume.rolling(20).mean().asof(date)` when `ohlcv_dict` is passed; otherwise volume column is left at 0 (filter skipped).
  - **Cap filter known limit**: today's `Marcap` from FDR `KRX` listing is used (no historical shares-outstanding source). Delisted stocks are exempted from the cap filter (cap_map has no entry → would otherwise drop all historical members). This is a residual lookahead (Tier 3 unfixed).
  - Result cached at `data/universe_cache/uni_<sha1:12>.csv` keyed on `(date, market, min_market_cap, min_avg_volume, meta_signature, vol_mode)`. Bypass with `SUPER_TRADER_DISABLE_UNIVERSE_CACHE=1`.
- `main.py:run_backtest` seeds `get_ohlcv_batch()` with the **union** of `get_universe(d)` over all rebalance dates so delisted codes' OHLCV is also pulled.
- `pool_cache._payload()` includes `pit_universe: True` and `meta_signature` (mtime+size SHA-1 of `universe_meta.csv`), so rebuilding the meta or upgrading from a pre-PIT cache automatically invalidates old `data/pool_cache/pool_*.json` (orphaned files can be deleted manually).
- `filter_by_listing_date()` is preserved as a defensive net (no-op when meta is present; safety for legacy fallback).

**Known PIT limits after Tier 1+2**:
- **Cap filter is today-cap, not PIT.** A stock that crossed 5,000억 in 2023 may appear in 2020 universe. Tier 3 would require historical shares-outstanding (pykrx broken; needs KRX MDC scraper or paid feed).
- **Sector/industry mapping is snapshot-only.** `get_sector_info(date)` returns today's sector. Acceptable for industry-neutralization, not for KOSPI200 membership reconstruction.
- **Delisted-stock OHLCV depends on FDR retention.** Last ~10 years generally OK; pre-2014 coverage is thin.
- **`KRX-DELISTING` Korean names are encoding-garbled.** Filtering uses code, so harmless. The `name` column in `universe_meta.csv` is unreliable for delisted rows.

### Codex CLI Review Pipeline (`scripts/codex_review.py`)

Three modes writing to `reports/codex_<mode>_<YYYY-MM-DD>.md`:
- `daily` — DB snapshot → `codex exec --sandbox read-only`
- `backtest` — latest `logs/*backtest*.log` + optional CSV → `codex exec`
- `code` — `codex review` with `--base`, `--uncommitted`, `--commit`

Auto-triggers: `--review` flag on backtest, `config.codex.enabled=true` for daily.

### Key Design Decisions

- **Pool restore on restart** (`_restore_pool_from_disk`) — prevents mass pool-diff trades when restarting intraday. Only rebuilds if <10 stocks saved.
- **Rule post-check** for LLM — deterministic rules override LLM if they disagree. Prevents LLM's loose interpretations (e.g. "RSI near 85" = hold) from breaking strategy intent.
- **Sentiment is learning-only** — SAPPO injects into reward during training; inference path unchanged.
- **Duplicate buy prevention** via `held_codes` set before every BUY order.
- **Model retraining** only replaces if F1 improves by >0.5%p; old model auto-backed up.
- **LLM validation** falls back to rule-based when API key missing or 404.
- **KIS order flow**: submit (no fill info) → next cycle `check_filled()` → emit ✅ confirmation with real filled_qty/price. Immediate notification uses `reference_price` (signal-time estimate).

### Environment Gotchas

- **WSL + Windows venv**: use `/mnt/e/SuperTrader/venv/Scripts/python.exe`, not Linux python.
- **pykrx cross-sectional API broken in this environment**: `get_market_cap`, `get_market_ticker_list`, `get_market_fundamental` return empty DataFrame due to KRX endpoint changes. Verified broken under both WSL Linux python and the Windows venv (`venv/Scripts/python.exe`) — it is not a WSL-only issue. OHLCV per-ticker still works. PIT membership is reconstructed via FDR `KRX-DESC` + `KRX-DELISTING` instead (see "Point-in-Time Universe" section below).
- **FDR `Volume` column is intraday today**: empty at market open → universe shrinks to 1-2 stocks. `get_universe()` has fallback: if filtered result <30 stocks, skip volume filter.
- **Trained models in `models/`**: `.pkl` (sklearn), `.pt` (torch). Not checked into git. RL backups: `rl_timing.backup_grpo.pt`, `backup_ppo_v1.pt`, `backup_ppo_v2.pt`.
- **PyTorch CUDA**: `torch 2.11.0+cu126` for GTX 1660 SUPER 6GB. Rollout CPU-bound; GPU only helps PPO batch updates. Mini-batch auto-scales (6GB → 384).
- **Backtest duration**: Pool build ~90min (84 monthly × 220 factors). Single strategy ~40min. `factor_only` seconds. Full comparison ~5 hours. Pool cache hit cuts first stage to ~1s.
- **Universe filter**: KOSPI, min cap 5,000억, min 20-day volume 300K. Produces ~123 large caps.
- **File encoding**: Windows Python writes EUC-KR/CP949 for Korean. JSON files (`data/current_pool.json`) and logs need multi-encoding fallback from WSL/Flask.
- **Korean in terminal logs**: often garbled in WSL display (cp949 pipe); files are fine. Use stock codes for reliable identification.
- **Google News RSS instead of Naver Finance**: Naver's per-stock news page is dynamically loaded (non-scrapable HTML). News collector uses `news.google.com/rss/search` with Korean ticker name.
- **Legacy/scratch files at repo root**: `run_dt_rl_backtest.py` and `*_result.txt` / `validation_result.txt` / `account_status.txt` are one-off scratch outputs, not part of the supported entry points. Don't take dependencies on them; treat as candidates for cleanup.
