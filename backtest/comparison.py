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
    only_strategies: list[str] | None = None,
    llm_filter: str | None = None,
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

    # RL 모델이 있으면 추가
    if "rl" in model_paths:
        from src.strategy.factor_rl import FactorRLStrategy
        strategies["factor_rl"] = FactorRLStrategy(
            model_path=model_paths["rl"],
        )

    # Hybrid(XGBoost + RL) 모델이 있으면 추가
    if "xgboost" in model_paths and "rl" in model_paths:
        from src.strategy.factor_hybrid import FactorHybridStrategy
        strategies["factor_hybrid"] = FactorHybridStrategy(
            ml_model_path=model_paths["xgboost"],
            rl_model_path=model_paths["rl"],
        )

    # 특정 전략만 실행
    if only_strategies:
        strategies = {k: v for k, v in strategies.items() if k in only_strategies}

    # LLM 필터 설정: "mock"=규칙 기반, "real"=실제 Claude API, None=비활성
    validator = None
    if llm_filter == "mock":
        from src.timing.llm_validator import MockSignalValidator
        validator = MockSignalValidator()
        logger.info("LLM 필터(mock) 활성 — 라이브와 동일 규칙을 백테스트에 적용")
    elif llm_filter == "real":
        from src.timing.llm_validator import SignalValidator
        validator = SignalValidator()
        if not validator.is_enabled:
            logger.warning("ANTHROPIC_API_KEY 없음 — LLM 필터 비활성")
            validator = None
        else:
            logger.warning("LLM 필터(real) 활성 — 시간/비용 증가 주의")

    results = {}
    equity_curves = {}

    for name, strategy in strategies.items():
        logger.info(f"전략 실행: {name}")

        engine = PortfolioBacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
            max_positions=config.risk.max_total_positions,
            llm_validator=validator,
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
