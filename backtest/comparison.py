"""
전략 비교 프레임워크
- 동일 데이터/종목풀에서 여러 전략 동시 실행
- 성과 지표 비교 테이블
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import get_config
from src.strategy.factor_only import FactorOnlyStrategy
from src.strategy.factor_macd import FactorMACDStrategy
from src.strategy.factor_kdj import FactorKDJStrategy
from backtest.portfolio_engine import PortfolioBacktestEngine


def run_strategy_comparison(
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str] | None = None,
) -> pd.DataFrame:
    """여러 전략을 동일 조건에서 비교합니다.

    Args:
        ohlcv_dict: {종목코드: OHLCV DataFrame}
        pool_history: {리밸런싱일: 종목코드 리스트}
        rebalance_dates: 리밸런싱 날짜 리스트
        model_paths: {모델타입: 모델경로} (ML 전략용)

    Returns:
        전략별 성과 지표 비교 테이블
    """
    config = get_config()
    model_paths = model_paths or {}

    strategies = {
        "factor_only": FactorOnlyStrategy(),
        "factor_macd": FactorMACDStrategy(params=config.strategy.params.model_dump()),
        "factor_kdj": FactorKDJStrategy(params=config.strategy.params.model_dump()),
    }

    # ML 모델이 있으면 추가
    for model_type in ["decision_tree", "xgboost", "lightgbm", "lstm", "transformer"]:
        if model_type in model_paths:
            from src.strategy.factor_ml import FactorMLStrategy
            strategies[f"factor_{model_type}"] = FactorMLStrategy(
                model_type=model_type,
                model_path=model_paths[model_type],
            )

    results = {}
    equity_curves = {}

    for name, strategy in strategies.items():
        logger.info(f"전략 실행: {name}")

        engine = PortfolioBacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
            max_positions=config.risk.max_total_positions,
        )

        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)

        if "error" not in result:
            results[name] = result["metrics"]
            equity_curves[name] = result["equity_curve"]
        else:
            logger.warning(f"전략 실패: {name} - {result['error']}")

    if not results:
        return pd.DataFrame()

    comparison = pd.DataFrame(results).T
    comparison.index.name = "strategy"

    # 주요 지표만 선택
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    available = [m for m in key_metrics if m in comparison.columns]
    comparison = comparison[available]

    logger.info(f"\n전략 비교 결과:\n{comparison.to_string()}")
    return comparison
