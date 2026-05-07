"""C+E A/B — 부분 익절 + 트레일링 스탑 + LLM RSI 보류 강화.

기준: factor_hybrid + confidence_sizing scale (현재 채택 baseline).
변수:
  OFF: partial_take_profit_enabled=false, llm_strong_momentum_hold_enabled=false
  ON : 둘 다 true (default partial_pct=0.50, trailing=0.03, rsi_threshold=80)

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
    enabled: bool,
    ohlcv_dict: dict[str, pd.DataFrame],
    pool_history: dict[str, list[str]],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
) -> dict:
    cfg = get_config()
    # C: 부분 익절 + 트레일링
    cfg.risk.partial_take_profit_enabled = enabled
    cfg.risk.partial_take_profit_pct = 0.50
    cfg.risk.trailing_stop_pct = 0.03
    # E: LLM RSI 보류
    cfg.risk.llm_strong_momentum_hold_enabled = enabled
    cfg.risk.llm_strong_momentum_rsi_threshold = 80.0

    logger.info("=" * 60)
    logger.info(f"[{label}] partial_tp + trailing + llm_rsi_hold = {enabled}")
    logger.info("=" * 60)

    strategy = FactorHybridStrategy(
        ml_model_path=model_paths["xgboost"],
        rl_model_path=model_paths["rl"],
        ml_model_type="xgboost",
        name=f"factor_hybrid_{label}",
    )

    # Mock LLM validator (라이브와 동일한 rule-based 필터, API 비용 0)
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
    return result.get("metrics", {})


def main() -> None:
    cfg = get_config()
    logger.info("C+E A/B 시작 (partial_tp + trailing + llm_rsi_hold)")

    # 원본 cfg 백업 (finally 에서 복원)
    orig = {
        "ptp_e": cfg.risk.partial_take_profit_enabled,
        "ptp_p": cfg.risk.partial_take_profit_pct,
        "ts_p": cfg.risk.trailing_stop_pct,
        "llm_e": cfg.risk.llm_strong_momentum_hold_enabled,
        "llm_t": cfg.risk.llm_strong_momentum_rsi_threshold,
    }

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")

    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
    logger.info(f"리밸런싱: {len(rebalance_dates)}회 ({cfg.factors.rebalance_freq})")

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
        logger.info(f"OOS 슬라이스: {len(rebalance_dates)}회")

    try:
        metrics_off = _run_one("OFF", False, ohlcv_dict, pool_history, rebalance_dates, model_paths)
        metrics_on = _run_one("ON",  True,  ohlcv_dict, pool_history, rebalance_dates, model_paths)
    finally:
        cfg.risk.partial_take_profit_enabled = orig["ptp_e"]
        cfg.risk.partial_take_profit_pct = orig["ptp_p"]
        cfg.risk.trailing_stop_pct = orig["ts_p"]
        cfg.risk.llm_strong_momentum_hold_enabled = orig["llm_e"]
        cfg.risk.llm_strong_momentum_rsi_threshold = orig["llm_t"]

    if not metrics_off or not metrics_on:
        logger.error("결과 누락 — 비교 불가")
        return

    df = pd.DataFrame({"OFF": metrics_off, "ON": metrics_on})
    df["delta"] = df["ON"] - df["OFF"]
    key = [
        "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
        "total_trades", "avg_holding_days",
    ]
    df = df.loc[[m for m in key if m in df.index]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/partial_tp_compare_{stamp}.csv"
    md_path = ROOT / f"reports/partial_tp_compare_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# C+E A/B — partial TP + trailing + LLM RSI hold ({stamp})\n\n"
        f"strategy=factor_hybrid + ON_up scale, MockLLM, "
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"train_ratio={cfg.backtest.train_ratio}\n\n"
        f"```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    # Slack
    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            tr_off = df.loc["total_return", "OFF"] if "total_return" in df.index else 0
            tr_on = df.loc["total_return", "ON"] if "total_return" in df.index else 0
            sh_off = df.loc["sharpe_ratio", "OFF"] if "sharpe_ratio" in df.index else 0
            sh_on = df.loc["sharpe_ratio", "ON"] if "sharpe_ratio" in df.index else 0
            mdd_off = df.loc["max_drawdown", "OFF"] if "max_drawdown" in df.index else 0
            mdd_on = df.loc["max_drawdown", "ON"] if "max_drawdown" in df.index else 0
            tr_off, tr_on = float(tr_off), float(tr_on)
            sh_off, sh_on = float(sh_off), float(sh_on)
            mdd_off, mdd_on = float(mdd_off), float(mdd_on)
            verdict = "✅ ON 우월" if (tr_on > tr_off and sh_on > sh_off) else (
                "🟡 mixed (수익↑ Sharpe↓)" if tr_on > tr_off else (
                    "🟡 mixed (수익↓ Sharpe↑)" if sh_on > sh_off else "❌ OFF 우월"
                )
            )
            msg = (
                "🔬 *C+E A/B — partial TP + trailing + LLM RSI hold*\n"
                f"strategy=factor_hybrid + scale, period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
                f"OOS {len(rebalance_dates)}회\n"
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
