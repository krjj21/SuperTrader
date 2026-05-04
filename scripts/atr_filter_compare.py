"""E1 변동성 필터 A/B 백테스트 — atr_filter OFF vs ON.

ON_up scale 1.0/2.0 + buy_action_threshold=0.03 환경 위에서 atr_filter_enabled
하나만 토글해 비교한다. Slack 자동 전송.
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
    atr_enabled: bool,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.risk.atr_filter_enabled = atr_enabled
    cfg.risk.atr_filter_max_pct = 0.05
    cfg.risk.atr_filter_period = 14
    logger.info("=" * 60)
    logger.info(f"[{label}] atr_filter_enabled={atr_enabled} max_pct=0.05")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        name=f"factor_hybrid_atr_{label}",
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
    logger.info("E1 atr_filter A/B 시작")

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
    logger.info(f"유니버스: {len(codes)}종목")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)

    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )
    model_paths = _train_models_if_needed(ohlcv_dict)

    train_ratio = float(getattr(cfg.backtest, "train_ratio", 1.0))
    if train_ratio < 1.0 and len(rebalance_dates) > 4:
        cutoff_idx = int(len(rebalance_dates) * train_ratio)
        rebalance_dates = rebalance_dates[cutoff_idx:]
        pool_history = {d: p for d, p in pool_history.items() if d in rebalance_dates}
        logger.info(f"OOS 슬라이스: {len(rebalance_dates)}회")

    metrics_off = _run_one("OFF", False, ohlcv_dict, pool_history, rebalance_dates, model_paths)
    metrics_on = _run_one("ON", True, ohlcv_dict, pool_history, rebalance_dates, model_paths)
    if not metrics_off or not metrics_on:
        logger.error("결과 누락")
        return

    df = pd.DataFrame({"atr_OFF": metrics_off, "atr_ON": metrics_on})
    df["delta"] = df["atr_ON"] - df["atr_OFF"]
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key_metrics if m in df.index]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/atr_filter_compare_{stamp}.csv"
    md_path = ROOT / f"reports/atr_filter_compare_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# E1 atr_filter A/B — {stamp}\n\n"
        f"strategy=factor_hybrid + ON_up scale 1.0/2.0, "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )

    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            msg = (
                "🛡️ *E1 변동성 필터 A/B* (atr_filter_max_pct=0.05)\n"
                f"strategy=factor_hybrid + ON_up scale, OOS {len(rebalance_dates)}회\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"리포트: `{csv_path.name}`"
            )
            ok = n._send(msg)
            logger.info(f"Slack 전송: {'OK' if ok else 'FAIL'}")
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


if __name__ == "__main__":
    main()
