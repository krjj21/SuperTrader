"""Confidence-proportional sizing on/off A/B 백테스트.

`factor_hybrid` 단일 전략으로 `risk.confidence_sizing_enabled` 만 토글해
사이드바이사이드 비교한다. 풀 히스토리/OHLCV/모델 학습은 한 번만 수행하고
두 엔진 실행 사이에 재사용해 시간을 절약한다.

사용:
    /mnt/e/SuperTrader/venv/Scripts/python.exe scripts/confidence_sizing_compare.py
출력:
    reports/confidence_sizing_compare_<stamp>.csv
    reports/confidence_sizing_compare_<stamp>.md
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.data.market_data import (  # noqa: E402
    filter_by_listing_date,
    get_ohlcv_batch,
    get_universe,
)
from src.runtime.backtest import (  # noqa: E402
    _build_pool_history_factor_based,
    _generate_rebalance_dates,
    _train_models_if_needed,
)
from src.factors.stock_pool import build_stock_pool  # noqa: E402
from src.strategy.factor_hybrid import FactorHybridStrategy  # noqa: E402
from backtest.portfolio_engine import PortfolioBacktestEngine  # noqa: E402


def _run_one(
    label: str,
    enabled: bool,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.risk.confidence_sizing_enabled = enabled
    logger.info("=" * 60)
    logger.info(f"[{label}] confidence_sizing_enabled={enabled}")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        name=f"factor_hybrid_{label}",
    )

    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
    )
    result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)
    if "error" in result:
        logger.error(f"[{label}] 실패: {result['error']}")
        return {}
    return result["metrics"]


def main() -> None:
    cfg = get_config()
    logger.info("Confidence sizing A/B 비교 시작")

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")

    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
    logger.info(f"리밸런싱: {len(rebalance_dates)}회")

    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        legacy = get_universe()
        all_codes = set(legacy["code"].tolist())
    codes = sorted(all_codes)
    logger.info(f"유니버스: {len(codes)}종목")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)
    logger.info(f"OHLCV: {len(ohlcv_dict)}종목 로드")

    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )
    model_paths = _train_models_if_needed(ohlcv_dict)

    train_ratio = float(getattr(cfg.backtest, "train_ratio", 1.0))
    if train_ratio < 1.0 and len(rebalance_dates) > 4:
        cutoff_idx = int(len(rebalance_dates) * train_ratio)
        rebalance_dates = rebalance_dates[cutoff_idx:]
        pool_history = {d: p for d, p in pool_history.items() if d in rebalance_dates}
        logger.info(f"OOS 슬라이스: {len(rebalance_dates)}회 리밸런싱")

    metrics_off = _run_one(
        "OFF", False, ohlcv_dict, pool_history, rebalance_dates, model_paths,
    )
    metrics_on = _run_one(
        "ON", True, ohlcv_dict, pool_history, rebalance_dates, model_paths,
    )

    if not metrics_off or not metrics_on:
        logger.error("결과 누락 — 비교 생략")
        return

    df = pd.DataFrame(
        {"conf_sizing_OFF": metrics_off, "conf_sizing_ON": metrics_on},
    )
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key_metrics if m in df.index]]
    df["delta"] = df["conf_sizing_ON"] - df["conf_sizing_OFF"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    csv_path = reports_dir / f"confidence_sizing_compare_{stamp}.csv"
    md_path = reports_dir / f"confidence_sizing_compare_{stamp}.md"

    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# Confidence sizing A/B — {stamp}\n\n"
        f"strategy=factor_hybrid, period={cfg.backtest.start_date}~{cfg.backtest.end_date}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )

    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")
    logger.info(f"저장: {md_path}")


if __name__ == "__main__":
    main()
