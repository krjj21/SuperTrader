# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run backtest (compares strategies: factor_only, factor_macd, factor_kdj, factor_decision_tree, factor_xgboost, factor_rl, factor_hybrid)
python main.py backtest

# Run live trading (requires KIS API keys in config/.env)
python main.py live

# Train a specific ML timing model
python main.py train --model xgboost          # options: decision_tree, xgboost, lightgbm, lstm, transformer, rl

# Retrain model with latest data (replaces only if new model is better)
python main.py retrain --model xgboost

# Check virtual trading account status
python main.py status

# Run specific strategy only (faster)
python main.py backtest --strategy factor_rl

# Custom config file
python main.py backtest --config config/settings.yaml

# Select factor module (alpha101, alpha158, both)
python main.py backtest --factor-module both

# Threshold sweep (buy_threshold grid search for hybrid strategy)
python scripts/threshold_sweep.py

# Codex CLI review pipeline (3 modes)
python scripts/codex_review.py daily                       # today's trading results
python scripts/codex_review.py daily --date 20260419       # specific date
python scripts/codex_review.py backtest                    # latest backtest log
python scripts/codex_review.py backtest --comparison reports/backtest_comparison_*.csv
python scripts/codex_review.py code                        # uncommitted changes (default)
python scripts/codex_review.py code --base main            # diff vs main branch

# Auto-trigger codex review after backtest completes
python main.py backtest --review

# Python executable (Windows python via WSL)
/mnt/e/SuperTrader/venv/Scripts/python.exe
```

No test suite exists yet (`tests/` directory is empty).

## Architecture

SuperTrader is a Korean stock (KOSPI/KOSDAQ) trading system that combines **factor-based stock selection** with **ML-based timing signals** and **LLM-based signal validation**.

### Five execution modes
- **backtest**: Runs 6 strategies on historical data, outputs comparison table
- **live**: Scheduled auto-trading via KIS broker API with risk management
- **train**: Trains ML timing models on historical OHLCV data
- **retrain**: Retrains with latest data, replaces model only if F1 improves
- **status**: Displays current account holdings and P&L

### Live Trading Pipeline

```
startup: build_stock_pool() → 30 stocks via factor scoring
    ↓
every 5 min: RL (PPO) generate_signal() → BUY/SELL/HOLD
    ↓
LLM validation (Claude Sonnet 4.6) → confirm or reject signal
    ↓
if confirmed: RiskManager validates → OrderManager executes via KIS API
    ↓
monthly: rebalance_pool() re-selects top 30 stocks
    ↓
weekly (Saturday): retrain model if new one beats old F1/Sharpe
```

### Data Pipeline (backtest mode)

Backtest and live use the **same factor pipeline** for stock selection:

```
market_data.get_universe() → KOSPI stocks (filtered by market cap/volume)
    ↓
market_data.get_ohlcv_batch() → {code: OHLCV DataFrame}
    ↓
main._build_pool_history_factor_based() → calls build_stock_pool() per rebalance date
    (220 factors [alpha101+alpha158] → neutralization → IC-weighted composite → top 30)
    ↓
main._train_models_if_needed() → trains DT/XGBoost if models/ is empty
    ↓
comparison.run_strategy_comparison() → runs all strategies through PortfolioBacktestEngine
    ↓
metrics.calculate_metrics() + report.print_comparison_table()
```

### Strategy Layer (`src/strategy/`)

All strategies inherit `BaseStrategy` (in `base.py`) and implement `generate_signal(code, df) → TradeSignal`.

- **FactorOnlyStrategy**: Rebalancing-only, no intra-period timing. Uses `commit_pool` / `update_pool` for the backtest engine to detect.
- **FactorMACDStrategy**: MACD golden/death cross signals with RSI confirmation.
- **FactorKDJStrategy**: KDJ overbought/oversold + J-line crossover signals.
- **FactorMLStrategy**: Wraps any ML model via `TimingPredictor`. Loads model by type string + path.
- **FactorRLStrategy**: PPO RL agent with **business-day holding tracking**. Uses `_business_days_held()` to convert entry_date to actual holding days, matching the daily-bar training environment. `.pt` file.
- **FactorHybridStrategy**: XGBoost(alpha generator) + RL(risk/timing filter). BUY requires both models to agree; SELL executes on XGBoost alone (risk management priority). Backtest: +96.5% return, Sharpe 0.81, MDD -11.8% with `buy_threshold=0.09`.

`Signal` enum: `BUY`, `SELL`, `HOLD`. `TradeSignal` dataclass includes strength (0-1), reason, stop_loss, take_profit.

### Timing Models (`src/timing/`)

- **features.py**: Builds ~30 features from OHLCV (RSI, MACD, KDJ, Bollinger Bands, ATR, ADX, MFI and their derivatives)
- **labels.py**: Forward N-day returns → 1 (BUY if > +2%), -1 (SELL if < -2%), 0 (HOLD)
- **trainer.py**: Training pipeline, dispatches to model-specific modules
- **predictor.py**: Unified inference interface — `TimingPredictor(model_type, model_path).predict(df) → 1/0/-1`
- **retrain.py**: Auto-retraining — trains new model, compares F1/accuracy against existing, replaces only if improved, backs up old model
- **llm_validator.py**: LLM signal validation — sends ML signal + technical context to Claude Sonnet 4.6 (configurable model), confirms or rejects. Relaxed prompt: BUY blocked only when RSI≥85 AND 1-day return≥+10% simultaneously; SELL always confirmed. RL model signals are respected by default.
- Model implementations: `decision_tree.py`, `gradient_boost.py` (XGBoost/LightGBM), `lstm_model.py`, `transformer_model.py`
- **rl_env.py**: Trading Gym environment — State(30 features + 3 position) / Action(BUY/HOLD/SELL) / Reward: Differential Sharpe Ratio + drawdown penalty + transaction cost. Log-return based risk-adjusted reward
- **rl_agent.py**: PPO Actor-Critic (shared backbone + LayerNorm) + GAE (λ=0.95) + mini-batch update. Legacy GRPO model auto-conversion supported. Enables `cudnn.benchmark` when CUDA available.
- **rl_trainer.py**: Multi-stock batch rollout → PPO update, walk-forward validation, Sharpe-based model selection, early stopping. Rollout collection uses **ThreadPoolExecutor** (8 workers, CPU inference) for ~2x speedup. GPU used only for PPO batch update. Mini-batch size auto-scales by VRAM (6GB → 384).

sklearn models save as `.pkl`, torch models as `.pt`.

### RL Training-Live Alignment

The RL model trains on **daily bars** (1 step = 1 trading day). In live mode, signals are checked every 5 minutes, but `holding_days` is computed as **actual business days** since entry (via `_business_days_held()`), not cycle count. This ensures the model's learned holding behavior matches real-world time.

**Reward v3 (current)**: `holding_cost=0.0002` + ramp `0.00005/day`, `opportunity_weight=0.35`. Produces ~5-17 trades/episode with Sharpe ~0.34, win rate ~78%. Higher holding cost or additional anti-churn penalties cause trades to collapse to near zero — v4 experiment (short_hold_penalty, roundtrip_penalty, churn_penalty) was reverted back to v3.

### RL Action Thresholds

The RL model outputs action probabilities P(HOLD), P(BUY), P(SELL). Configurable thresholds in `config/settings.yaml` → `timing.rl`:
- `buy_action_threshold: 0.09` — P(BUY) > 9% triggers BUY (optimized via grid search from [0.05-0.15])
- `sell_action_threshold: 0.06` — P(SELL) > 6% triggers SELL (model average level)

These flow through `factor_rl.py` → `predictor.py` → `rl_agent.py`. Live trading calls `sync_positions(held_codes, prices, avg_prices)` each cycle to pass actual holdings to the RL strategy. Buy date resolution order: **DB (`holding_positions`) → daily chart estimation (closest close to avg_price) → today (fallback)**. Estimated dates are auto-saved to DB for future lookups.

### Backtest Engine (`backtest/`)

`PortfolioBacktestEngine.run()` iterates every trading day across all dates:
1. On rebalancing dates: updates stock pool, sells exited stocks, buys new entries
2. Daily: calls `strategy.generate_signal()` for each stock in pool
3. Tracks positions, cash, equity curve with commission (0.015%) and tax (0.23%)

**Performance**: Engine pre-builds date→price dict caches and uses `bisect` + O(1) dict lookup for price retrieval. Date strings are converted once during initialization. MACD/KDJ/ML strategies still take ~30 min each for 7-year backtests due to `generate_signal()` per stock per day.

### Configuration (`src/config.py`)

Pydantic models loaded from `config/settings.yaml`. `load_config(path)` sets a module-level singleton; `get_config()` retrieves it. Key sections: `universe`, `factors`, `timing`, `strategy`, `risk`, `backtest`, `schedule`.

Secrets (KIS API keys, Slack token, Anthropic API key) loaded from `config/.env` via pydantic-settings.

### Data Sources (`src/data/`)

- **market_data.py**: `pykrx` for universe/market cap, `FinanceDataReader` for OHLCV
- **indicators.py**: Technical indicator calculations (pure pandas/numpy)
- **factor_data.py**: Fundamental data (PER, PBR, dividend yield)

### Factor System (`src/factors/`)

- **alpha101.py**: ~62 quantitative factors (momentum, volatility, volume, fundamental)
- **alpha158.py**: 158 Qlib-style factors — candlestick(9), raw price(4), rolling time-series(115: 23 types × 5 windows [5,10,20,30,60d]), volume(30: 6 types × 5 windows). All prefixed `a158_` to avoid name collision.
- **calculator.py** → **validity.py** (IC analysis) → **neutralizer.py** (industry/market-cap) → **composite.py** (IC-weighted scoring) → **stock_pool.py** (top-N selection)
- `_get_factor_functions()` dispatches to alpha101/alpha158/both based on `config.factors.factor_module`
- `build_stock_pool()` accepts optional `ohlcv_dict` for backtest data reuse

Rebalancing frequency: **monthly** (매월 첫 거래일). Both backtest and live use the same pipeline. Changed from biweekly based on academic research showing monthly rebalancing reduces turnover by ~50% with negligible impact on returns.

### Database (`src/db/`)

SQLAlchemy ORM with SQLite. `init_db(path)` creates tables (guarded against duplicate calls). Models: `TradeLog`, `DailyPnL`, `PositionLog`, `SignalLog`, `RuntimeStatus`, `HoldingPosition`. `save_signal_log()` records LLM validation results; `get_recent_signals(limit, days)` queries them for the web dashboard. `save_runtime_status()` writes a singleton row for live pipeline state.

**HoldingPosition** (`holding_positions` table): Tracks current positions with `buy_date` (YYYYMMDD). Written on BUY fill (`save_holding()`), deleted on SELL/stop-loss fill (`remove_holding()`). `get_holding_buy_date(code)` returns the stored buy date. This solves the holding_days=0 bug where system restarts would lose the actual entry date.

### Web Dashboard (`web/`)

Flask app (`web/app.py`) for real-time account monitoring. Run separately from main process. Auto-refreshes every 30 seconds.

```bash
# Run dashboard (from project root)
/mnt/e/SuperTrader/venv/Scripts/python.exe web/app.py
```

**Dashboard sections**: Summary cards (total value, asset change, realized P&L, unrealized P&L, available cash, today volume), Cumulative Return chart, Holdings table (clickable for technical indicator charts), Factor Pool table (30 stocks with composite scores, scrollable), Recent Signals log (RL + LLM validation).

**API endpoints**: `/api/status` (account + realized P&L), `/api/pool` (factor pool from `data/current_pool.json`), `/api/signals` (DB or log fallback), `/api/returns` (cumulative return from DailyPnL), `/api/indicators/<code>` (technical charts).

**Factor Pool JSON**: Live trading writes `data/current_pool.json` on each rebalance with stock codes, names, and composite scores. Dashboard reads this file (handles EUC-KR/UTF-8 encoding from Windows Python).

### Live Trading (`src/broker/`, `src/risk/`, `src/notification/`)

- **kis_client.py**: Korean Investment & Securities REST API (supports virtual/real via `is_virtual` config). Rate limited to 10 calls/sec with auto-retry.
- **order.py** / **account.py**: Order execution and balance queries. Virtual/real trading uses different tr_id prefixes (VTTC vs TTTC). `AccountSummary` has both `total_deposit` (D+2 settled cash) and `available_cash` (order-ready amount including unsettled sell proceeds). `get_realized_pnl()` queries daily realized P&L via KIS API.
- **risk/manager.py**: Position sizing (5% per stock, ATR-based), daily loss limit (-5%), stop loss (-7%), kill switch (3 consecutive errors). Kill switch monitors all scheduled jobs (signal check, rebalance, retrain).
- **notification/slack_bot.py**: Trade alerts and daily reports to `#supertrader` channel (requires `slack-sdk`)
- **notification/notion_reporter.py**: Daily trading log to Notion database (positions table, trade list, AI feedback)

### Live Trading Schedules (APScheduler)

- **Signal check**: every `check_interval_sec` (default 300s) starting at market_open (09:00)
- **Daily report**: cron at post_market (15:40)
- **Monthly rebalance**: cron day=1 at pre_market (08:30)
- **Weekly retrain**: cron on Saturday at 06:00

### Codex CLI Review Pipeline (`scripts/codex_review.py`)

Single entry point that delegates analysis to the `codex` CLI. Writes review outputs to `reports/codex_<mode>_<YYYY-MM-DD>.md`.

Three modes:
- **daily** — queries `DailyPnL`, `TradeLog`, `HoldingPosition`, `PositionLog`, `SignalLog` for a date and pipes JSON to `codex exec`
- **backtest** — tails the latest `logs/*backtest*.log` (+ optional comparison CSV) and pipes to `codex exec`
- **code** — delegates to `codex review` (supports `--base`, `--uncommitted`, `--commit`)

Auto-triggers:
- `python main.py backtest --review` saves `reports/backtest_comparison_<stamp>.csv` and runs `codex_review.py backtest`
- Live `daily_report()` calls `codex_review.py daily` when `codex.enabled=true` in `config/settings.yaml`

Codex runs with `--sandbox read-only` (for `exec`) so the reviewer cannot mutate the repo. Config flags in `config/settings.yaml` → `codex`: `enabled`, `daily_review`, `model`.

### Key Design Decisions

- Duplicate buy prevention: `held_codes` set checked before BUY orders
- LLM validation gates all BUY/SELL signals; falls back to ML-only if API key missing or API error
- Model retraining only replaces if F1 improves by >0.5%p; old model auto-backed up
- Terminal encoding issues with Korean stock names in WSL — use stock code for reliable identification
- Windows Python writes files in EUC-KR by default; web dashboard handles multi-encoding reads

### Environment Gotchas

- **WSL + Windows venv**: The project uses a Windows Python venv from WSL. Always use `/mnt/e/SuperTrader/venv/Scripts/python.exe`, not a Linux python.
- **Trained models in `models/`**: `.pkl` (sklearn) and `.pt` (torch) files. Not checked into git — must be trained locally via `python main.py train`. RL model backups: `rl_timing.backup_grpo.pt` (legacy), `backup_ppo_v1.pt`, `backup_ppo_v2.pt` (v3 reward).
- **PyTorch CUDA**: `torch 2.11.0+cu126` installed for GTX 1660 SUPER 6GB. Rollout is CPU-bound (env simulation); GPU only helps PPO batch updates.
- **Backtest duration**: Pool build ~90min (220 factors × 84 monthly rebalances). Single strategy ~40 min each. `factor_only` runs in seconds. Full comparison ~5 hours.
- **Universe filter**: KOSPI, min market cap 5000억, min avg volume 300K. Produces ~123 large-cap stocks.
- **File encoding**: Windows Python outputs EUC-KR/CP949 for Korean text. JSON files (e.g. `data/current_pool.json`) and logs need multi-encoding fallback when reading from WSL/Flask.
