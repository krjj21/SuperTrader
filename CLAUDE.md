# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run backtest (compares all strategies: factor_only, factor_macd, factor_kdj, factor_decision_tree, factor_xgboost)
python main.py backtest

# Run live trading (requires KIS API keys in config/.env)
python main.py live

# Train a specific ML timing model
python main.py train --model xgboost          # options: decision_tree, xgboost, lightgbm, lstm, transformer

# Custom config file
python main.py backtest --config config/settings.yaml

# Activate venv (Windows python via WSL)
# The venv uses Windows python: venv/Scripts/python.exe
```

No test suite exists yet (`tests/` directory is empty).

## Architecture

SuperTrader is a Korean stock (KOSPI/KOSDAQ) trading system that combines **factor-based stock selection** with **ML-based timing signals**.

### Three execution modes
- **backtest**: Runs 5 strategies on historical data, outputs comparison table
- **live**: Scheduled auto-trading via KIS broker API with risk management
- **train**: Trains ML timing models on historical OHLCV data

### Data Pipeline (backtest mode)

```
market_data.get_universe() → 311 KOSPI stocks
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

`Signal` enum: `BUY`, `SELL`, `HOLD`. `TradeSignal` dataclass includes strength (0-1), reason, stop_loss, take_profit.

### Timing Models (`src/timing/`)

- **features.py**: Builds ~30 features from OHLCV (RSI, MACD, KDJ, Bollinger Bands, ATR, ADX, MFI and their derivatives)
- **labels.py**: Forward N-day returns → 1 (BUY if > +2%), -1 (SELL if < -2%), 0 (HOLD)
- **trainer.py**: Walk-forward training pipeline, dispatches to model-specific modules
- **predictor.py**: Unified inference interface — `TimingPredictor(model_type, model_path).predict(df) → 1/0/-1`
- Model implementations: `decision_tree.py`, `gradient_boost.py` (XGBoost/LightGBM), `lstm_model.py`, `transformer_model.py`

sklearn models save as `.pkl`, torch models as `.pt`.

### Backtest Engine (`backtest/`)

`PortfolioBacktestEngine.run()` iterates every trading day across all dates:
1. On rebalancing dates: updates stock pool, sells exited stocks, buys new entries
2. Daily: calls `strategy.generate_signal()` for each stock in pool
3. Tracks positions, cash, equity curve with commission (0.015%) and tax (0.23%)

**Performance bottleneck**: The inner loop does `df[df["date"].astype(str) <= date]` and `df[df["date"].astype(str) == date]` on every iteration — O(n) string conversion per lookup. MACD/KDJ/ML strategies take 30-60 min each for 7-year backtests.

### Configuration (`src/config.py`)

Pydantic models loaded from `config/settings.yaml`. `load_config(path)` sets a module-level singleton; `get_config()` retrieves it. Key sections: `universe`, `factors`, `timing`, `strategy`, `risk`, `backtest`, `schedule`.

Secrets (KIS API keys, Slack token) loaded from `config/.env` via python-dotenv.

### Data Sources (`src/data/`)

- **market_data.py**: `pykrx` for universe/market cap, `FinanceDataReader` for OHLCV
- **indicators.py**: Technical indicator calculations (pure pandas/numpy)
- **factor_data.py**: Fundamental data (PER, PBR, dividend yield)

### Factor System (`src/factors/`)

- **alpha101.py**: ~100 quantitative factors (momentum, volatility, volume, fundamental)
- **calculator.py** → **validity.py** (IC analysis) → **neutralizer.py** (industry/market-cap) → **composite.py** (IC-weighted scoring) → **stock_pool.py** (top-N selection)

### Live Trading (`src/broker/`, `src/risk/`, `src/notification/`)

- **kis_client.py**: Korean Investment & Securities REST API (supports virtual/real)
- **order.py** / **account.py**: Order execution and balance queries
- **risk/manager.py**: Position sizing, daily loss limit (-3%), stop loss (-7%), kill switch
- **notification/slack_bot.py**: Trade alerts and daily reports
