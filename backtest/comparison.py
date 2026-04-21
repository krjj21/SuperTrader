"""
전략 비교 프레임워크
- 동일 데이터/종목풀에서 여러 전략 동시 실행
- 성과 지표 비교 테이블
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_config
from src.strategy.factor_only import FactorOnlyStrategy
from src.strategy.factor_macd import FactorMACDStrategy
from src.strategy.factor_kdj import FactorKDJStrategy
from backtest.portfolio_engine import PortfolioBacktestEngine
from backtest.llm_filter_report import (
    generate_report as generate_llm_filter_report,
    comparison_columns as llm_filter_comparison_cols,
    save_report as save_llm_filter_report,
)


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
    filter_cols: dict[str, dict] = {}
    filter_markdowns: list[str] = []
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")

    for name, strategy in strategies.items():
        logger.info(f"전략 실행: {name}")

        run_id = f"bt_{name}_{run_ts}" if validator is not None else ""
        engine = PortfolioBacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
            max_positions=config.risk.max_total_positions,
            llm_validator=validator,
            run_id=run_id,
            persist_signals=validator is not None,
        )

        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)

        if "error" not in result:
            results[name] = result["metrics"]
            equity_curves[name] = result["equity_curve"]

            # LLM 필터 활성 + 결정 존재 시 차단 효과 리포트 생성
            decisions = result.get("llm_decisions") or []
            if validator is not None and decisions:
                summary, md = generate_llm_filter_report(
                    decisions, ohlcv_dict,
                    strategy=name, filter_mode=str(llm_filter),
                )
                filter_cols[name] = llm_filter_comparison_cols(summary, primary_horizon=5)
                # 각 전략 개별 리포트 저장
                md_path = reports_dir / f"llm_filter_{name}_{run_ts}.md"
                save_llm_filter_report(md, md_path)
                logger.info(f"[{name}] LLM 필터 리포트 저장: {md_path}")
                filter_markdowns.append(md)
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

    # LLM 필터 활성 시 차단/알파 열 추가
    if filter_cols:
        filter_df = pd.DataFrame(filter_cols).T
        filter_df.index.name = "strategy"
        comparison = comparison.join(filter_df, how="left")

        # 통합 Markdown 리포트 저장
        if filter_markdowns:
            combined_path = reports_dir / f"llm_filter_ALL_{run_ts}.md"
            save_llm_filter_report("\n\n---\n\n".join(filter_markdowns), combined_path)
            logger.info(f"LLM 필터 통합 리포트 저장: {combined_path}")

    logger.info(f"\n전략 비교 결과:\n{comparison.to_string()}")
    return comparison
