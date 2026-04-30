"""buy_action_threshold 스윕 — ON_up (mode=scale, min=1.0, max=2.0) 고정 위에서.

기존 ON_up @ threshold=0.03 결과(`confidence_sizing_up_20260430_233755.csv`)를
재사용하고 0.01 / 0.05 / 0.07 만 추가 실행해 4-way 비교한다.

Slack 자동 전송.
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

PRIOR_UP_CSV = ROOT / "reports/confidence_sizing_up_20260430_233755.csv"


def _run_one(
    th: float,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.risk.confidence_sizing_enabled = True
    cfg.risk.confidence_sizing_mode = "scale"
    cfg.risk.confidence_sizing_min_mult = 1.0
    cfg.risk.confidence_sizing_max_mult = 2.0
    cfg.timing.rl.buy_action_threshold = float(th)
    logger.info("=" * 60)
    logger.info(f"[buy_th={th}] ON_up scale 1.0/2.0 고정")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        name=f"factor_hybrid_th{int(th*100):02d}",
    )
    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
    )
    result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)
    if "error" in result:
        logger.error(f"[th={th}] 실패: {result['error']}")
        return {}
    return result["metrics"]


def main() -> None:
    cfg = get_config()
    logger.info("buy_action_threshold 스윕 시작 (ON_up scale 1.0/2.0 고정)")

    if not PRIOR_UP_CSV.exists():
        logger.error(f"기존 ON_up 결과 누락: {PRIOR_UP_CSV}")
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

    # 신규 실행 — 0.01 / 0.05 / 0.07
    new_thresholds = [0.01, 0.05, 0.07]
    new_results: dict[float, dict] = {}
    for th in new_thresholds:
        m = _run_one(th, ohlcv_dict, pool_history, rebalance_dates, model_paths)
        if not m:
            logger.warning(f"th={th} 실패 — 스윕 계속")
            continue
        new_results[th] = m
        logger.info(
            f"[th={th}] return={m.get('total_return',0):.2f}, "
            f"sharpe={m.get('sharpe_ratio',0):.2f}, "
            f"trades={m.get('total_trades',0):.0f}, "
            f"mdd={m.get('max_drawdown',0):.2f}"
        )

    # 기존 ON_up @ 0.03 + 신규 결과 합치기
    prior = pd.read_csv(PRIOR_UP_CSV, index_col=0)
    prior_up = prior["ON_up"]

    columns = {"th=0.03": prior_up}
    for th in new_thresholds:
        if th in new_results:
            columns[f"th={th:.2f}"] = pd.Series(new_results[th])

    df = pd.DataFrame(columns)
    key_metrics = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key_metrics if m in df.index]]
    df = df[sorted(df.columns, key=lambda c: float(c.split("=")[1]))]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "reports"
    csv_path = reports_dir / f"buy_threshold_sweep_{stamp}.csv"
    md_path = reports_dir / f"buy_threshold_sweep_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# buy_action_threshold 스윕 — {stamp}\n\n"
        f"strategy=factor_hybrid + ON_up (scale 1.0/2.0), "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )

    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    # Slack 전송
    try:
        from src.notification.slack_bot import SlackNotifier
        notifier = SlackNotifier()
        if notifier.token:
            best_col = df.loc["total_return"].idxmax()
            best_ret = df.loc["total_return", best_col]
            best_sharpe = df.loc["sharpe_ratio", best_col]
            best_mdd = df.loc["max_drawdown", best_col]
            best_trades = df.loc["total_trades", best_col]
            msg = (
                "📊 *buy_action_threshold 스윕* — ON_up scale 1.0/2.0 위에서\n"
                f"strategy=factor_hybrid, OOS {len(rebalance_dates)}회 리밸런싱\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"🏆 *peak total_return*: `{best_col}` "
                f"return={best_ret:+.2f}, sharpe={best_sharpe:.2f}, "
                f"mdd={best_mdd:.2f}, trades={best_trades:.0f}\n"
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
