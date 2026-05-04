"""B1 투자자 필터 모드 A/B — foreign vs foreign_organ.

투자자 필터 모드 변경 시 pool cache 가 자동 무효화되므로 풀 빌드가 한 번 더 일어난다.
pool 빌드 ~90분 + hybrid 백테스트 ~37분 × 2 ≈ ~165분 첫 실행.
ON_up scale + atr_filter 채택값을 그대로 두고 모드만 토글.
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
    mode: str,
    ohlcv_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.factors.investor_filter_mode = mode
    logger.info("=" * 60)
    logger.info(f"[{label}] investor_filter_mode={mode}")
    logger.info("=" * 60)

    # 모드 변경마다 pool 재구축 (pool_cache hash 가 mode 포함)
    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )

    train_ratio = float(getattr(cfg.backtest, "train_ratio", 1.0))
    rebal_used = list(rebalance_dates)
    pool_used = dict(pool_history)
    if train_ratio < 1.0 and len(rebal_used) > 4:
        cutoff_idx = int(len(rebal_used) * train_ratio)
        rebal_used = rebal_used[cutoff_idx:]
        pool_used = {d: p for d, p in pool_used.items() if d in rebal_used}

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        name=f"factor_hybrid_inv_{label}",
    )
    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
    )
    result = engine.run(strategy, ohlcv_dict, pool_used, rebal_used)
    return result.get("metrics", {})


def main() -> None:
    cfg = get_config()
    logger.info("B1 investor_filter_mode A/B 시작")

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")

    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        all_codes = set(get_universe()["code"].tolist())
    codes = sorted(all_codes)
    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)
    model_paths = _train_models_if_needed(ohlcv_dict)
    logger.info(f"OHLCV: {len(ohlcv_dict)}종목")

    metrics_foreign = _run_one("foreign", "foreign", ohlcv_dict, rebalance_dates, model_paths)
    metrics_combined = _run_one("foreign_organ", "foreign_organ", ohlcv_dict, rebalance_dates, model_paths)

    if not metrics_foreign or not metrics_combined:
        logger.error("결과 누락")
        return

    df = pd.DataFrame({"foreign": metrics_foreign, "foreign_organ": metrics_combined})
    df["delta"] = df["foreign_organ"] - df["foreign"]
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key_metrics if m in df.index]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/investor_filter_compare_{stamp}.csv"
    md_path = ROOT / f"reports/investor_filter_compare_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# B1 investor_filter_mode A/B — {stamp}\n\n"
        f"strategy=factor_hybrid + ON_up scale 1.0/2.0\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")

    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            msg = (
                "🌐 *B1 투자자 필터 A/B* (foreign vs foreign+organ 합산)\n"
                f"strategy=factor_hybrid + ON_up scale\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


if __name__ == "__main__":
    main()
