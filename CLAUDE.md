# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run backtest (compares all strategies: factor_only, factor_macd, factor_kdj, factor_decision_tree, factor_xgboost)
python main.py backtest

# Run live trading (requires KIS API keys in config/.env)
python main.py live

# Train a specific ML timing model
python main.py train --model xgboost          # options: decision_tree, xgboost, lightgbm, lstm, transformer, rl

# Retrain model with latest data (replaces only if new model is better)
python main.py retrain --model xgboost

# Check virtual trading account status
python main.py status

# Custom config file
python main.py backtest --config config/settings.yaml

# Python executable (Windows python via WSL)
/mnt/e/SuperTrader/venv/Scripts/python.exe
```

No test suite exists yet (`tests/` directory is empty).

## Architecture

SuperTrader is a Korean stock (KOSPI/KOSDAQ) trading system that combines **factor-based stock selection** with **ML-based timing signals** and **LLM-based signal validation**.

### Five execution modes
- **backtest**: Runs 5 strategies on historical data, outputs comparison table
- **live**: Scheduled auto-trading via KIS broker API with risk management
- **train**: Trains ML timing models on historical OHLCV data
- **retrain**: Retrains with latest data, replaces model only if F1 improves
- **status**: Displays current account holdings and P&L

### Live Trading Pipeline

```
startup: build_stock_pool() → 30 stocks via factor scoring
    ↓
every 5 min: XGBoost generate_signal() → BUY/SELL/HOLD
    ↓
LLM validation (Claude Haiku) → confirm or reject signal
    ↓
if confirmed: RiskManager validates → OrderManager executes via KIS API
    ↓
monthly: rebalance_pool() re-selects top 30 stocks
    ↓
weekly (Saturday): retrain XGBoost if new model beats old F1
```

### Data Pipeline (backtest mode)

```
market_data.get_universe() → KOSPI stocks (filtered by market cap/volume)
    ↓
market_data.get_ohlcv_batch() → {code: OHLCV DataFrame}
    ↓
main._build_pool_history_from_ohlcv() → momentum-ranked top 30 stocks per month
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
- **FactorRLStrategy**: PPO (Proximal Policy Optimization) RL agent. Actor-Critic + GAE + Differential Sharpe Ratio reward. `.pt` 파일.

`Signal` enum: `BUY`, `SELL`, `HOLD`. `TradeSignal` dataclass includes strength (0-1), reason, stop_loss, take_profit.

### Timing Models (`src/timing/`)

- **features.py**: Builds ~30 features from OHLCV (RSI, MACD, KDJ, Bollinger Bands, ATR, ADX, MFI and their derivatives)
- **labels.py**: Forward N-day returns → 1 (BUY if > +2%), -1 (SELL if < -2%), 0 (HOLD)
- **trainer.py**: Training pipeline, dispatches to model-specific modules
- **predictor.py**: Unified inference interface — `TimingPredictor(model_type, model_path).predict(df) → 1/0/-1`
- **retrain.py**: Auto-retraining — trains new model, compares F1/accuracy against existing, replaces only if improved, backs up old model
- **llm_validator.py**: LLM signal validation — sends ML signal + technical context to Claude Haiku, confirms or rejects based on RSI/MA/volume analysis
- Model implementations: `decision_tree.py`, `gradient_boost.py` (XGBoost/LightGBM), `lstm_model.py`, `transformer_model.py`
- **rl_env.py**: Trading Gym environment — State(30 features + 3 position) / Action(BUY/HOLD/SELL) / Reward: Differential Sharpe Ratio + drawdown penalty + transaction cost. Log-return 기반 risk-adjusted reward
- **rl_agent.py**: PPO Actor-Critic (shared backbone + LayerNorm) + GAE (λ=0.95) + mini-batch update. Legacy GRPO 모델 자동 변환 로드 지원
- **rl_trainer.py**: Multi-stock batch rollout → PPO update, walk-forward validation, Sharpe-based model selection, early stopping

sklearn models save as `.pkl`, torch models as `.pt`.

Current XGBoost hyperparameters (tuned to prevent overfitting): max_depth=4, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=10, learning_rate=0.03, n_estimators=300.

### Backtest Engine (`backtest/`)

`PortfolioBacktestEngine.run()` iterates every trading day across all dates:
1. On rebalancing dates: updates stock pool, sells exited stocks, buys new entries
2. Daily: calls `strategy.generate_signal()` for each stock in pool
3. Tracks positions, cash, equity curve with commission (0.015%) and tax (0.23%)

**Performance bottleneck**: The inner loop does `df[df["date"].astype(str) <= date]` and `df[df["date"].astype(str) == date]` on every iteration — O(n) string conversion per lookup. MACD/KDJ/ML strategies take 30-60 min each for 7-year backtests.

### Configuration (`src/config.py`)

Pydantic models loaded from `config/settings.yaml`. `load_config(path)` sets a module-level singleton; `get_config()` retrieves it. Key sections: `universe`, `factors`, `timing`, `strategy`, `risk`, `backtest`, `schedule`.

Secrets (KIS API keys, Slack token, Anthropic API key) loaded from `config/.env` via pydantic-settings.

### Data Sources (`src/data/`)

- **market_data.py**: `pykrx` for universe/market cap, `FinanceDataReader` for OHLCV
- **indicators.py**: Technical indicator calculations (pure pandas/numpy)
- **factor_data.py**: Fundamental data (PER, PBR, dividend yield)

### Factor System (`src/factors/`)

- **alpha101.py**: ~100 quantitative factors (momentum, volatility, volume, fundamental)
- **calculator.py** → **validity.py** (IC analysis) → **neutralizer.py** (industry/market-cap) → **composite.py** (IC-weighted scoring) → **stock_pool.py** (top-N selection)

In live mode, `build_stock_pool()` runs at startup and monthly via APScheduler cron job (rebalance_day in config).

### Database (`src/db/`)

SQLAlchemy ORM with SQLite. `init_db(path)` creates tables. Models: `TradeLog`, `DailyPnL`, `Position`. `save_daily_pnl()` records daily P&L snapshots for historical tracking.

### Web Dashboard (`web/`)

Flask app (`web/app.py`) for real-time account monitoring. Run separately from main process. Auto-refreshes every 30 seconds. Uses `config/settings.yaml` for KIS credentials.

```bash
# Run dashboard (from project root)
/mnt/e/SuperTrader/venv/Scripts/python.exe web/app.py
```

### Live Trading (`src/broker/`, `src/risk/`, `src/notification/`)

- **kis_client.py**: Korean Investment & Securities REST API (supports virtual/real via `is_virtual` config). Rate limited to 10 calls/sec with auto-retry.
- **order.py** / **account.py**: Order execution and balance queries. Virtual/real trading uses different tr_id prefixes (VTTC vs TTTC).
- **risk/manager.py**: Position sizing (5% per stock, ATR-based), daily loss limit (-3%), stop loss (-7%), kill switch (3 consecutive errors)
- **notification/slack_bot.py**: Trade alerts and daily reports

### Live Trading Schedules (APScheduler)

- **Signal check**: every `check_interval_sec` (default 300s) starting at market_open (09:00)
- **Daily report**: cron at post_market (15:40)
- **Monthly rebalance**: cron on rebalance_day (1st) at pre_market (08:30)
- **Weekly retrain**: cron on Saturday at 06:00

### Key Design Decisions

- Duplicate buy prevention: `held_codes` set checked before BUY orders
- LLM validation gates all BUY/SELL signals; falls back to ML-only if API key missing or API error
- Model retraining only replaces if F1 improves by >0.5%p; old model auto-backed up
- Terminal encoding issues with Korean stock names in WSL — use stock code for reliable identification

### Environment Gotchas

- **WSL + Windows venv**: The project uses a Windows Python venv from WSL. Always use `/mnt/e/SuperTrader/venv/Scripts/python.exe`, not a Linux python.
- **Missing from requirements.txt**: `anthropic` (for LLM validator) and `flask` (for web dashboard) are used but not listed in `requirements.txt`. Install separately if needed.
- **Trained models in `models/`**: `.pkl` (sklearn) and `.pt` (torch) files. Not checked into git — must be trained locally via `python main.py train`.
