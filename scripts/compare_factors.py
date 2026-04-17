"""
Alpha101 vs Alpha158 팩터 A/B 비교 백테스트

Usage:
    /mnt/e/SuperTrader/venv/Scripts/python.exe scripts/compare_factors.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.add("logs/factor_comparison.log", rotation="10 MB")


def run_factor_backtest(module_name: str, ohlcv_dict: dict, config) -> dict:
    """지정된 팩터 모듈로 factor_only 백테스트를 실행합니다."""
    from src.factors.stock_pool import build_stock_pool
    from backtest.portfolio_engine import PortfolioBacktestEngine
    from backtest.metrics import calculate_metrics
    from src.strategy.factor_only import FactorOnlyStrategy

    # 팩터 모듈 임시 변경
    original = config.factors.factor_module
    config.factors.factor_module = module_name
    logger.info(f"=== {module_name} 백테스트 시작 ===")

    try:
        start = config.backtest.start_date.replace("-", "")
        end = config.backtest.end_date.replace("-", "")

        # 리밸런싱 날짜 생성
        from main import _generate_rebalance_dates
        rebalance_dates = _generate_rebalance_dates(start, end, config.factors.rebalance_freq)

        # 종목풀 히스토리 구축
        from main import _build_pool_history_factor_based
        pool_history = _build_pool_history_factor_based(ohlcv_dict, rebalance_dates, build_stock_pool)

        # 백테스트 실행
        strategy = FactorOnlyStrategy()
        engine = PortfolioBacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
        )
        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)

        # 메트릭 계산
        metrics = calculate_metrics(result)
        return metrics

    finally:
        config.factors.factor_module = original


def main():
    from src.config import load_config, get_config
    from src.data.market_data import get_universe, get_ohlcv_batch

    load_config("config/settings.yaml")
    config = get_config()

    logger.info("Alpha101 vs Alpha158 팩터 비교 시작")

    # 데이터 1회 로드 (공유)
    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")
    universe = get_universe(end)
    codes = universe["code"].tolist()[:100]
    logger.info(f"유니버스: {len(codes)}종목")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    logger.info(f"OHLCV 로드: {len(ohlcv_dict)}종목")

    # A/B 실행
    t0 = time.time()
    metrics_101 = run_factor_backtest("alpha101", ohlcv_dict, config)
    t1 = time.time()
    metrics_158 = run_factor_backtest("alpha158", ohlcv_dict, config)
    t2 = time.time()

    # 결과 비교
    print("\n" + "=" * 60)
    print("  Alpha101 vs Alpha158 Factor Comparison")
    print("=" * 60)

    keys = [
        ("total_return", "Total Return", "%"),
        ("cagr", "CAGR", "%"),
        ("sharpe_ratio", "Sharpe", ""),
        ("sortino_ratio", "Sortino", ""),
        ("max_drawdown", "Max Drawdown", "%"),
        ("calmar_ratio", "Calmar", ""),
        ("win_rate", "Win Rate", "%"),
        ("profit_factor", "Profit Factor", ""),
        ("total_trades", "Total Trades", ""),
        ("avg_holding_days", "Avg Hold Days", ""),
    ]

    print(f"\n{'Metric':<18} {'Alpha101':>12} {'Alpha158':>12} {'Diff':>10}")
    print("-" * 55)
    for key, label, unit in keys:
        v101 = metrics_101.get(key, 0)
        v158 = metrics_158.get(key, 0)
        diff = v158 - v101 if isinstance(v101, (int, float)) else 0
        suffix = unit
        print(f"{label:<18} {v101:>11.2f}{suffix} {v158:>11.2f}{suffix} {diff:>+9.2f}")

    print(f"\nTime: alpha101={t1-t0:.0f}s, alpha158={t2-t1:.0f}s")
    logger.info(f"비교 완료: alpha101={t1-t0:.0f}s, alpha158={t2-t1:.0f}s")


if __name__ == "__main__":
    main()
