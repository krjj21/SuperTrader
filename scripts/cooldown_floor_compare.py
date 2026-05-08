"""SELL cooldown + RL-only SELL prob floor A/B.

OFF: reentry_cooldown_days=0, rl_sell_ml_prob_floor=0.0
ON : reentry_cooldown_days=3, rl_sell_ml_prob_floor=0.45

같은 train + pool + 모델 사이클 안에서 OFF/ON 둘 다 실행 (fair compare).
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


def _run_one(
    label: str,
    cooldown: int,
    sell_floor: float,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    cfg.risk.reentry_cooldown_days = cooldown
    cfg.timing.rl.rl_sell_ml_prob_floor = sell_floor
    logger.info("=" * 60)
    logger.info(f"[{label}] cooldown={cooldown}d, sell_floor={sell_floor}")
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
    # cooldown 차단 카운트 첨부
    metrics["cooldown_blocked"] = float(getattr(engine, "_cooldown_blocked_count", 0))
    return metrics


def main() -> None:
    cfg = get_config()
    logger.info("SELL cooldown + RL prob floor A/B 시작")

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
    logger.info(f"유니버스: {len(codes)}종목, 리밸런싱 {len(rebalance_dates)}회")

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

    try:
        m_off = _run_one("OFF", 0, 0.0, ohlcv_dict, pool_history, rebalance_dates, model_paths)
        m_on = _run_one("ON", 3, 0.45, ohlcv_dict, pool_history, rebalance_dates, model_paths)
    finally:
        cfg.risk.reentry_cooldown_days = orig["cd"]
        cfg.timing.rl.rl_sell_ml_prob_floor = orig["sf"]

    if not m_off or not m_on:
        logger.error("결과 누락")
        return

    df = pd.DataFrame({"OFF": m_off, "ON": m_on})
    df["delta"] = df["ON"] - df["OFF"]
    key = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days", "cooldown_blocked",
    ]
    df = df.loc[[m for m in key if m in df.index]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/cooldown_floor_compare_{stamp}.csv"
    md_path = ROOT / f"reports/cooldown_floor_compare_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# SELL cooldown(3d) + RL prob floor(0.45) A/B — {stamp}\n\n"
        f"strategy=factor_hybrid + scale, MockLLM, "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"train_ratio={cfg.backtest.train_ratio}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            tr_off = float(df.loc["total_return", "OFF"])
            tr_on = float(df.loc["total_return", "ON"])
            sh_off = float(df.loc["sharpe_ratio", "OFF"])
            sh_on = float(df.loc["sharpe_ratio", "ON"])
            mdd_off = float(df.loc["max_drawdown", "OFF"])
            mdd_on = float(df.loc["max_drawdown", "ON"])
            verdict = "✅ ON 우월" if (tr_on > tr_off and sh_on > sh_off) else (
                "🟡 mixed (수익↑ Sharpe↓)" if tr_on > tr_off else (
                    "🟡 mixed (수익↓ Sharpe↑)" if sh_on > sh_off else "❌ OFF 우월"
                )
            )
            msg = (
                "🔬 *SELL cooldown(3d) + RL prob floor(0.45) A/B*\n"
                f"strategy=factor_hybrid + scale, OOS {len(rebalance_dates)}회\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"{verdict}: total_return {tr_off:+.2f}% → {tr_on:+.2f}% "
                f"(Δ {tr_on - tr_off:+.2f}%p), Sharpe {sh_off:.2f} → {sh_on:.2f}, "
                f"MDD {mdd_off:+.2f}% → {mdd_on:+.2f}%\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


if __name__ == "__main__":
    main()
