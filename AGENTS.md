# Repository Guidelines

## Project Structure & Module Organization

SuperTrader is a Python auto-trading system for Korean equities. The main CLI entrypoint is `main.py`. Core application code lives in `src/`, split by responsibility: `strategy/` for trading strategies, `factors/` for alpha factor construction, `timing/` for ML/RL timing models, `risk/` for order sizing and trading guards, `broker/` for KIS API integration, `db/` for SQLAlchemy models, and `regime/` for market regime detection. Backtesting code is in `backtest/`. Operational scripts are in `scripts/`, Flask dashboard code is in `web/`, configuration is in `config/settings.yaml`, and tests live in `tests/`.

## Build, Test, and Development Commands

Use the project virtualenv on this machine:

```bash
./venv/Scripts/python.exe -m unittest tests.test_strategy_invariants -v
python3 -m compileall -q main.py backtest src scripts tests
./venv/Scripts/python.exe main.py backtest --strategy factor_hybrid
./venv/Scripts/python.exe main.py train --model xgboost
./venv/Scripts/python.exe web/app.py
```

`unittest` runs the current invariant suite. `compileall` catches syntax errors without requiring full runtime dependencies. Backtests may be slow and can use cached pools/models.

## Coding Style & Naming Conventions

Use Python 3.12+ style with type hints where practical. Prefer small modules with clear boundaries matching the existing package layout. Use `snake_case` for functions, variables, and module names; `PascalCase` for classes; and explicit strategy names such as `factor_hybrid`. Keep comments concise and explain trading assumptions, timing semantics, or safety decisions rather than restating code.

## Testing Guidelines

Tests use the standard library `unittest`; `pytest` is not installed in the current virtualenv. Add tests under `tests/` with names like `test_*.py`. Focus on strategy invariants: no lookahead, `T` signal to `T+1` execution, forced pool exits, risk sizing, and live/backtest alignment. Mock broker, model, and network dependencies.

## Commit & Pull Request Guidelines

Recent history uses concise imperative messages, often with a subsystem prefix or Korean summary, for example `Add strategy invariant tests` or `Regime Detector ... 반영`. Keep commits scoped and avoid including generated files such as `logs/`, `__pycache__/`, `data/*.db*`, or local `.env` files. Pull requests should include the purpose, main behavioral changes, verification commands, and any trading-risk impact.

## Security & Configuration Tips

Do not commit secrets from `config/.env`, token caches, databases, model backups, or account logs. Live trading depends on KIS credentials and market-hour guards; changes to `risk/`, `broker/`, `strategy/`, or order execution paths require explicit test evidence before deployment.
