"""XGB confidence threshold sweep (BUY/SELL pair).

목적: 현재 (0.55, 0.60) default 가 최적인지 검증.
가설 (오늘 진단): baseline +5%/3.5년 underperform 의 원인이 *신호 게이트 너무 보수적*.
threshold ↓ → 신호 빈도 ↑ → 시장 follow 정확도 ↑ 가정.

cells:
  cell 1: BUY=0.45, SELL=0.50  (가장 관대)
  cell 2: BUY=0.50, SELL=0.55
  cell 3: BUY=0.55, SELL=0.60  (현재 default = baseline)
  cell 4: BUY=0.60, SELL=0.65  (더 보수적)

같은 train + pool + 모델 사이클 안에서 4 cell 모두 실행.
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

CELLS = [
    ("buy045_sell050", 0.45, 0.50),
    ("buy050_sell055", 0.50, 0.55),
    ("buy055_sell060", 0.55, 0.60),  # default
    ("buy060_sell065", 0.60, 0.65),
]


def _run_one(label, buy_th, sell_th, ohlcv_dict, pool_history, rebalance_dates, model_paths):
    cfg = get_config()
    cfg.timing.rl.xgb_buy_confidence_threshold = buy_th
    cfg.timing.rl.xgb_sell_confidence_threshold = sell_th

    logger.info("=" * 60)
    logger.info(f"[{label}] BUY={buy_th}, SELL={sell_th}")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        ml_buy_threshold=buy_th,
        ml_sell_threshold=sell_th,
        name=f"factor_hybrid_{label}",
    )
    from src.timing.llm_validator import MockSignalValidator
    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
        llm_validator=MockSignalValidator(),
    )
    try:
        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)
    except Exception as e:
        logger.error(f"[{label}] engine.run 예외: {e}")
        return {}
    if "error" in result:
        return {}
    return result.get("metrics", {})


def main():
    cfg = get_config()
    logger.info("XGB threshold sweep 시작")
    orig = {
        "buy": cfg.timing.rl.xgb_buy_confidence_threshold,
        "sell": cfg.timing.rl.xgb_sell_confidence_threshold,
    }

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
    logger.info(f"유니버스: {len(codes)}, 리밸런싱: {len(rebalance_dates)}")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)
    pool_history = _build_pool_history_factor_based(ohlcv_dict, rebalance_dates, build_stock_pool)
    model_paths = _train_models_if_needed(ohlcv_dict)

    train_ratio = float(getattr(cfg.backtest, "train_ratio", 1.0))
    if train_ratio < 1.0 and len(rebalance_dates) > 4:
        cutoff_idx = int(len(rebalance_dates) * train_ratio)
        rebalance_dates = rebalance_dates[cutoff_idx:]
        pool_history = {d: p for d, p in pool_history.items() if d in rebalance_dates}
        logger.info(f"OOS: {len(rebalance_dates)}회")

    results = {}
    try:
        for label, buy_th, sell_th in CELLS:
            m = _run_one(label, buy_th, sell_th, ohlcv_dict, pool_history, rebalance_dates, model_paths)
            if not m:
                logger.warning(f"{label} 실패")
                continue
            results[label] = m
            logger.info(
                f"[{label}] return={m.get('total_return',0):+.2f}%, "
                f"sharpe={m.get('sharpe_ratio',0):.2f}, "
                f"trades={m.get('total_trades',0):.0f}, "
                f"mdd={m.get('max_drawdown',0):+.2f}%"
            )
    finally:
        cfg.timing.rl.xgb_buy_confidence_threshold = orig["buy"]
        cfg.timing.rl.xgb_sell_confidence_threshold = orig["sell"]

    if not results:
        logger.error("결과 없음")
        return

    df = pd.DataFrame(results)
    key = [
        "total_return", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key if m in df.index]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/xgb_threshold_sweep_{stamp}.csv"
    md_path = ROOT / f"reports/xgb_threshold_sweep_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# XGB threshold sweep — {stamp}\n\n"
        f"strategy=factor_hybrid + scale, MockLLM, "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"train_ratio={cfg.backtest.train_ratio}, OOS {len(rebalance_dates)}회\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            # 점수: return + 5×Sharpe
            scores = {}
            for col in df.columns:
                tr = float(df.loc["total_return", col]) if "total_return" in df.index else 0
                sh = float(df.loc["sharpe_ratio", col]) if "sharpe_ratio" in df.index else 0
                scores[col] = tr + sh * 5
            best = max(scores, key=scores.get)
            best_tr = float(df.loc["total_return", best])
            best_sh = float(df.loc["sharpe_ratio", best])
            best_mdd = float(df.loc["max_drawdown", best])
            best_tr_default = float(df.loc["total_return", "buy055_sell060"]) if "buy055_sell060" in df.columns else 0
            msg = (
                "🔬 *XGB confidence threshold sweep*\n"
                f"factor_hybrid + scale, OOS {len(rebalance_dates)}회\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"🏆 best: *{best}* — return {best_tr:+.2f}%, Sharpe {best_sh:.2f}, MDD {best_mdd:+.2f}%\n"
                f"vs default (0.55/0.60): {best_tr - best_tr_default:+.2f}%p\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 예외: {e}")


if __name__ == "__main__":
    main()
