"""SELL cooldown × RL prob floor 4-cell A/B (효과 분리 측정).

cell 1: cd=0, floor=0    (baseline)
cell 2: cd=3, floor=0    (cooldown only)
cell 3: cd=0, floor=0.45 (floor only)
cell 4: cd=3, floor=0.45 (both)

같은 train + pool + 모델 사이클 안에서 4-cell 모두 실행.
Slack 자동 전송 + verdict.
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
    ("baseline",     0, 0.0),
    ("cooldown",     3, 0.0),
    ("floor",        0, 0.45),
    ("both",         3, 0.45),
]


def _run_one(label, cooldown, sell_floor, ohlcv_dict, pool_history, rebalance_dates, model_paths):
    cfg = get_config()
    cfg.risk.reentry_cooldown_days = cooldown
    cfg.timing.rl.rl_sell_ml_prob_floor = sell_floor
    logger.info("=" * 60)
    logger.info(f"[{label}] cd={cooldown}, floor={sell_floor}")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
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
        logger.error(f"[{label}] 실패: {result['error']}")
        return {}
    metrics = result.get("metrics", {})
    metrics["cooldown_blocked"] = float(getattr(engine, "_cooldown_blocked_count", 0))
    return metrics


def main():
    cfg = get_config()
    logger.info("4-cell A/B 시작 (cd × floor)")

    orig = {
        "cd": cfg.risk.reentry_cooldown_days,
        "sf": cfg.timing.rl.rl_sell_ml_prob_floor,
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
    logger.info(f"OHLCV: {len(ohlcv_dict)}")

    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )
    model_paths = _train_models_if_needed(ohlcv_dict)

    train_ratio = float(getattr(cfg.backtest, "train_ratio", 1.0))
    if train_ratio < 1.0 and len(rebalance_dates) > 4:
        cutoff_idx = int(len(rebalance_dates) * train_ratio)
        rebalance_dates = rebalance_dates[cutoff_idx:]
        pool_history = {d: p for d, p in pool_history.items() if d in rebalance_dates}
        logger.info(f"OOS: {len(rebalance_dates)}회")

    results: dict[str, dict] = {}
    try:
        for label, cd, fl in CELLS:
            m = _run_one(label, cd, fl, ohlcv_dict, pool_history, rebalance_dates, model_paths)
            if not m:
                logger.warning(f"{label} 실패 — 계속")
                continue
            results[label] = m
    finally:
        cfg.risk.reentry_cooldown_days = orig["cd"]
        cfg.timing.rl.rl_sell_ml_prob_floor = orig["sf"]

    if not results:
        logger.error("결과 없음")
        return

    df = pd.DataFrame(results)
    key = [
        "total_return", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days", "cooldown_blocked",
    ]
    df = df.loc[[m for m in key if m in df.index]]

    # 효과 분리: cooldown only / floor only / interaction
    if all(k in results for k in ("baseline", "cooldown", "floor", "both")):
        delta_cd = df["cooldown"] - df["baseline"]
        delta_fl = df["floor"] - df["baseline"]
        delta_both = df["both"] - df["baseline"]
        # 상호작용: both - (cd_only + fl_only - baseline) — 양수면 시너지, 음수면 상쇄
        interaction = df["both"] - (df["cooldown"] + df["floor"] - df["baseline"])
        df["Δ_cd"] = delta_cd
        df["Δ_fl"] = delta_fl
        df["Δ_both"] = delta_both
        df["interact"] = interaction

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/cooldown_floor_4cell_{stamp}.csv"
    md_path = ROOT / f"reports/cooldown_floor_4cell_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# cooldown × floor 4-cell A/B — {stamp}\n\n"
        f"strategy=factor_hybrid + scale, MockLLM, "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"train_ratio={cfg.backtest.train_ratio}\n\n"
        f"```\n{df.to_string()}\n```\n\n"
        f"## 분리 효과 해석\n\n"
        f"- Δ_cd = cooldown only - baseline  (cooldown 단독 효과)\n"
        f"- Δ_fl = floor only - baseline      (floor 단독 효과)\n"
        f"- Δ_both = both - baseline           (둘 다 적용 효과)\n"
        f"- interact = both - (cd_only + fl_only - baseline)  "
        f"(양수=시너지, 음수=상쇄)\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            # 권고 결정
            if all(k in results for k in ("baseline", "cooldown", "floor", "both")):
                tr_b = float(df.loc["total_return", "baseline"])
                tr_cd = float(df.loc["total_return", "cooldown"])
                tr_fl = float(df.loc["total_return", "floor"])
                tr_bo = float(df.loc["total_return", "both"])
                sh_b = float(df.loc["sharpe_ratio", "baseline"])
                sh_cd = float(df.loc["sharpe_ratio", "cooldown"])
                sh_fl = float(df.loc["sharpe_ratio", "floor"])
                sh_bo = float(df.loc["sharpe_ratio", "both"])
                # 가장 좋은 cell (수익률+Sharpe 합산 점수)
                scores = {
                    "baseline": tr_b + sh_b * 5,
                    "cooldown": tr_cd + sh_cd * 5,
                    "floor": tr_fl + sh_fl * 5,
                    "both": tr_bo + sh_bo * 5,
                }
                best = max(scores, key=scores.get)
                rec = f"🏆 best cell: *{best}* (return+5×Sharpe={scores[best]:.2f})"
            else:
                rec = ""
            msg = (
                "🔬 *cooldown × floor 4-cell A/B*\n"
                f"strategy=factor_hybrid + scale, OOS {len(rebalance_dates)}회\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"{rec}\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 예외: {e}")


if __name__ == "__main__":
    main()
