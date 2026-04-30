"""Confidence sizing 상향 모드(min=1.0, max=2.0, mode=scale) 단독 백테스트.

기존 `confidence_sizing_compare_20260430_224300.csv` 의 OFF 베이스라인을
재사용해 OFF / ON_down(0.5,1.0) / ON_up(1.0,2.0) 3-way 비교 표를 만든다.
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

PRIOR_CSV = ROOT / "reports/confidence_sizing_compare_20260430_224300.csv"


def _run_one_engine(
    label: str,
    enabled: bool,
    mode: str,
    min_mult: float,
    max_mult: float,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.risk.confidence_sizing_enabled = enabled
    cfg.risk.confidence_sizing_mode = mode
    cfg.risk.confidence_sizing_min_mult = float(min_mult)
    cfg.risk.confidence_sizing_max_mult = float(max_mult)
    logger.info("=" * 60)
    logger.info(
        f"[{label}] enabled={enabled} mode={mode} "
        f"min={min_mult} max={max_mult}"
    )
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
    logger.info("Confidence sizing 상향 모드 단독 실행")

    if not PRIOR_CSV.exists():
        logger.error(f"기존 OFF 결과 누락: {PRIOR_CSV}")
        return

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")

    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
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
    logger.info(f"OHLCV: {len(ohlcv_dict)}종목")

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

    metrics_up = _run_one_engine(
        "UP", True, "scale", 1.0, 2.0,
        ohlcv_dict, pool_history, rebalance_dates, model_paths,
    )
    if not metrics_up:
        logger.error("ON_up 결과 누락")
        return

    prior = pd.read_csv(PRIOR_CSV, index_col=0)
    df = pd.DataFrame(
        {
            "OFF": prior["conf_sizing_OFF"],
            "ON_down": prior["conf_sizing_ON"],
            "ON_up": pd.Series(metrics_up),
        }
    )
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key_metrics if m in df.index]]
    df["delta_up_vs_off"] = df["ON_up"] - df["OFF"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "reports"
    csv_path = reports_dir / f"confidence_sizing_up_{stamp}.csv"
    md_path = reports_dir / f"confidence_sizing_up_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# Confidence sizing 상향 A/B — {stamp}\n\n"
        f"strategy=factor_hybrid, period={cfg.backtest.start_date}~{cfg.backtest.end_date}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )

    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")
    logger.info(f"저장: {md_path}")

    # ── Slack 자동 전송 ──
    try:
        from src.notification.slack_bot import SlackNotifier
        notifier = SlackNotifier()
        if notifier.token:
            tbl = df.to_string()
            msg = (
                "📈 *Confidence Sizing — 상향 모드 A/B* (mode=scale, min=1.0, max=2.0)\n"
                f"strategy=factor_hybrid, OOS {len(rebalance_dates)}회 리밸런싱\n"
                "```\n"
                f"{tbl}\n"
                "```\n"
                f"리포트: `{csv_path.name}`"
            )
            ok = notifier._send(msg)
            logger.info(f"Slack 전송: {'OK' if ok else 'FAIL'}")
        else:
            logger.warning("Slack 토큰 없음 — 전송 생략")
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


if __name__ == "__main__":
    main()
