"""Phase 1 — Layer Ablation: factor / +ML / +ML+RL / +ML+RL+LLM 4-cell.

목적: 각 layer (factor / XGB / RL / LLM) 의 *진짜 alpha 기여* 분리 측정.
- Cell 1: factor_only — 풀만, timing 없음 (engine 의 force-BUY 만)
- Cell 2: factor_xgb — XGB 단독 timing
- Cell 3: factor_xgb_rl — XGB + RL hybrid (현재 default minus LLM)
- Cell 4: factor_xgb_rl_llm — 전체 (현재 운영 + MockLLM)

같은 train + pool + 모델 사이클 안에서 4 cell 모두 실행 (fair compare).
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
from src.strategy.factor_only import FactorOnlyStrategy  # noqa: E402
from src.strategy.factor_ml import FactorMLStrategy  # noqa: E402
from src.strategy.factor_hybrid import FactorHybridStrategy  # noqa: E402
from src.timing.llm_validator import MockSignalValidator  # noqa: E402
from backtest.portfolio_engine import PortfolioBacktestEngine  # noqa: E402


def _run_one(label: str, strategy, llm_validator, ohlcv_dict, pool_history, rebalance_dates):
    cfg = get_config()
    logger.info("=" * 60)
    logger.info(f"[{label}] LLM={'ON' if llm_validator else 'OFF'}, strategy={strategy.name}")
    logger.info("=" * 60)

    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
        llm_validator=llm_validator,
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
    logger.info("Layer ablation 시작 (factor / +XGB / +RL / +LLM)")

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

    # 4 cells
    cells = [
        (
            "factor_only",
            FactorOnlyStrategy(),
            None,
        ),
        (
            "factor_xgb",
            FactorMLStrategy(model_type="xgboost", model_path=model_paths["xgboost"]),
            None,
        ),
        (
            "factor_xgb_rl",
            FactorHybridStrategy(
                ml_model_path=model_paths["xgboost"],
                rl_model_path=model_paths["rl"],
                ml_model_type="xgboost",
                name="factor_xgb_rl",
            ),
            None,
        ),
        (
            "factor_xgb_rl_llm",
            FactorHybridStrategy(
                ml_model_path=model_paths["xgboost"],
                rl_model_path=model_paths["rl"],
                ml_model_type="xgboost",
                name="factor_xgb_rl_llm",
            ),
            MockSignalValidator(),
        ),
    ]

    results: dict[str, dict] = {}
    for label, strategy, validator in cells:
        m = _run_one(label, strategy, validator, ohlcv_dict, pool_history, rebalance_dates)
        if not m:
            logger.warning(f"{label} 실패 — 계속")
            continue
        results[label] = m
        logger.info(
            f"[{label}] return={m.get('total_return',0):+.2f}%, "
            f"sharpe={m.get('sharpe_ratio',0):.2f}, "
            f"trades={m.get('total_trades',0):.0f}, "
            f"mdd={m.get('max_drawdown',0):+.2f}%"
        )

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

    # 레이어별 delta (factor_only baseline 대비)
    if "factor_only" in df.columns:
        for col in df.columns:
            df[f"Δ_{col}"] = df[col] - df["factor_only"]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/layer_ablation_{stamp}.csv"
    md_path = ROOT / f"reports/layer_ablation_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# Layer Ablation — {stamp}\n\n"
        f"strategy: factor_only / +XGB / +XGB+RL / +XGB+RL+LLM\n"
        f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"train_ratio={cfg.backtest.train_ratio}, OOS {len(rebalance_dates)}회\n\n"
        f"## 결과\n\n```\n{df.to_string()}\n```\n\n"
        f"## 진단\n\n"
        f"- Δ_factor_xgb: XGB layer 단독 추가 효과\n"
        f"- Δ_factor_xgb_rl: +RL 효과 (cumulative)\n"
        f"- Δ_factor_xgb_rl_llm: +LLM 효과 (cumulative, 현재 운영)\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    # Slack
    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token and "factor_only" in df.columns:
            tr_only = float(df.loc["total_return", "factor_only"])
            tr_xgb = float(df.loc["total_return", "factor_xgb"]) if "factor_xgb" in df.columns else None
            tr_rl = float(df.loc["total_return", "factor_xgb_rl"]) if "factor_xgb_rl" in df.columns else None
            tr_llm = float(df.loc["total_return", "factor_xgb_rl_llm"]) if "factor_xgb_rl_llm" in df.columns else None

            # 레이어별 incremental delta
            delta_xgb = (tr_xgb - tr_only) if tr_xgb is not None else 0
            delta_rl = (tr_rl - tr_xgb) if tr_rl is not None and tr_xgb is not None else 0
            delta_llm = (tr_llm - tr_rl) if tr_llm is not None and tr_rl is not None else 0

            # 진단
            verdict_lines = []
            if abs(tr_llm - tr_only) < 1.0 if tr_llm is not None else False:
                verdict_lines.append(f"🚨 *factor_only ≈ 전체 layer ({tr_only:+.2f}% ≈ {tr_llm:+.2f}%)* — layer 들이 alpha 기여 거의 없음. *factor 자체 한계*")
            if delta_xgb < 0:
                verdict_lines.append(f"❌ XGB 추가 시 수익 {delta_xgb:+.2f}%p 악화 — XGB timing 이 noise")
            if delta_rl < 0:
                verdict_lines.append(f"❌ RL 추가 시 수익 {delta_rl:+.2f}%p 악화")
            if delta_llm < 0:
                verdict_lines.append(f"❌ LLM 추가 시 수익 {delta_llm:+.2f}%p 악화")
            if not verdict_lines:
                verdict_lines.append(f"✓ 각 layer 가 점진적으로 alpha 추가")

            msg = (
                "🔬 *Phase 1 Layer Ablation* (factor / +XGB / +RL / +LLM)\n"
                f"OOS {len(rebalance_dates)}회, train_ratio={cfg.backtest.train_ratio}\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"*Incremental Δ (total_return)*:\n"
                f"  +XGB: {delta_xgb:+.2f}%p (factor_only → factor_xgb)\n"
                f"  +RL:  {delta_rl:+.2f}%p (factor_xgb → factor_xgb_rl)\n"
                f"  +LLM: {delta_llm:+.2f}%p (factor_xgb_rl → factor_xgb_rl_llm)\n\n"
                f"*진단*:\n" + "\n".join(verdict_lines) + "\n\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 예외: {e}")


if __name__ == "__main__":
    main()
