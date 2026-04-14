# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run backtest (compares 6 strategies: factor_only, factor_macd, factor_kdj, factor_decision_tree, factor_xgboost, factor_rl)
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
LLM validation (Claude Haiku) → confirm or reject signal
    ↓
if confirmed: RiskManager validates → OrderManager executes via KIS API
    ↓
biweekly: rebalance_pool() re-selects top 30 stocks
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
    (70+ factors → neutralization → IC-weighted composite → top 30)
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

`Signal` enum: `BUY`, `SELL`, `HOLD`. `TradeSignal` dataclass includes strength (0-1), reason, stop_loss, take_profit.

### Timing Models (`src/timing/`)

- **features.py**: Builds ~30 features from OHLCV (RSI, MACD, KDJ, Bollinger Bands, ATR, ADX, MFI and their derivatives)
- **labels.py**: Forward N-day returns → 1 (BUY if > +2%), -1 (SELL if < -2%), 0 (HOLD)
- **trainer.py**: Training pipeline, dispatches to model-specific modules
- **predictor.py**: Unified inference interface — `TimingPredictor(model_type, model_path).predict(df) → 1/0/-1`
- **retrain.py**: Auto-retraining — trains new model, compares F1/accuracy against existing, replaces only if improved, backs up old model
- **llm_validator.py**: LLM signal validation — sends ML signal + technical context to Claude Haiku (configurable model), confirms or rejects based on RSI/MA/volume analysis
- Model implementations: `decision_tree.py`, `gradient_boost.py` (XGBoost/LightGBM), `lstm_model.py`, `transformer_model.py`
- **rl_env.py**: Trading Gym environment — State(30 features + 3 position) / Action(BUY/HOLD/SELL) / Reward: Differential Sharpe Ratio + drawdown penalty + transaction cost. Log-return based risk-adjusted reward
- **rl_agent.py**: PPO Actor-Critic (shared backbone + LayerNorm) + GAE (λ=0.95) + mini-batch update. Legacy GRPO model auto-conversion supported
- **rl_trainer.py**: Multi-stock batch rollout → PPO update, walk-forward validation, Sharpe-based model selection, early stopping

sklearn models save as `.pkl`, torch models as `.pt`.

### RL Training-Live Alignment

The RL model trains on **daily bars** (1 step = 1 trading day). In live mode, signals are checked every 5 minutes, but `holding_days` is computed as **actual business days** since entry (via `_business_days_held()`), not cycle count. This ensures the model's learned holding behavior matches real-world time.

**Known issue**: With `holding_cost=0.0` in rl_env.py, the model tends to converge to a buy-and-hold strategy (avg_trades ~0.5/episode). Reward structure adjustment (holding cost, opportunity cost) is needed to encourage active trading.

### RL Action Thresholds

The RL model outputs action probabilities P(HOLD), P(BUY), P(SELL). Configurable thresholds in `config/settings.yaml` → `timing.rl`:
- `buy_action_threshold: 0.13` — P(BUY) > 13% triggers BUY (model average level)
- `sell_action_threshold: 0.06` — P(SELL) > 6% triggers SELL (model average level)

These flow through `factor_rl.py` → `predictor.py` → `rl_agent.py`. Live trading calls `sync_positions()` each cycle to pass actual holdings to the RL strategy so it can generate SELL signals for held stocks.

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

- **alpha101.py**: ~70 quantitative factors (momentum, volatility, volume, fundamental)
- **calculator.py** → **validity.py** (IC analysis) → **neutralizer.py** (industry/market-cap) → **composite.py** (IC-weighted scoring) → **stock_pool.py** (top-N selection)
- `compute_cross_sectional_factors()` accepts optional `ohlcv_dict` for backtest (avoids re-fetching)
- `build_stock_pool()` accepts optional `ohlcv_dict` for backtest data reuse

Rebalancing frequency: **biweekly** (14-day intervals). Both backtest and live use the same pipeline.

### Database (`src/db/`)

SQLAlchemy ORM with SQLite. `init_db(path)` creates tables (guarded against duplicate calls). Models: `TradeLog`, `DailyPnL`, `PositionLog`, `SignalLog`, `RuntimeStatus`. `save_signal_log()` records LLM validation results; `get_recent_signals(limit, days)` queries them for the web dashboard. `save_runtime_status()` writes a singleton row for live pipeline state.

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
- **Biweekly rebalance**: interval `weeks=2` at pre_market (08:30)
- **Weekly retrain**: cron on Saturday at 06:00

### Key Design Decisions

- Duplicate buy prevention: `held_codes` set checked before BUY orders
- LLM validation gates all BUY/SELL signals; falls back to ML-only if API key missing or API error
- Model retraining only replaces if F1 improves by >0.5%p; old model auto-backed up
- Terminal encoding issues with Korean stock names in WSL — use stock code for reliable identification
- Windows Python writes files in EUC-KR by default; web dashboard handles multi-encoding reads

### Environment Gotchas

- **WSL + Windows venv**: The project uses a Windows Python venv from WSL. Always use `/mnt/e/SuperTrader/venv/Scripts/python.exe`, not a Linux python.
- **Trained models in `models/`**: `.pkl` (sklearn) and `.pt` (torch) files. Not checked into git — must be trained locally via `python main.py train`.
- **Backtest duration**: Full 6-strategy comparison (2018–2024) takes ~3 hours. Single strategy via `--strategy factor_rl` takes ~40 min. `factor_only` runs in seconds.
- **File encoding**: Windows Python outputs EUC-KR/CP949 for Korean text. JSON files (e.g. `data/current_pool.json`) and logs need multi-encoding fallback when reading from WSL/Flask.
