"""
Hybrid 전략 buy_threshold 그리드 서치 백테스트

종목풀을 1회만 구축하고, threshold만 바꿔가며 6회 백테스트 반복.
결과를 비교표로 출력하고 Slack으로 전송.

Usage:
    /mnt/e/SuperTrader/venv/Scripts/python.exe scripts/threshold_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

logger.add("logs/threshold_sweep.log", rotation="10 MB")


def main():
    from src.config import load_config, get_config
    from src.data.market_data import get_universe, get_ohlcv_batch
    from src.factors.stock_pool import build_stock_pool
    from backtest.portfolio_engine import PortfolioBacktestEngine
    from src.strategy.factor_hybrid import FactorHybridStrategy

    load_config("config/settings.yaml")
    config = get_config()

    # ── 1. 데이터 로드 (1회) ──
    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")

    logger.info("데이터 로드 시작")
    universe = get_universe(end)
    codes = universe["code"].tolist()[:100]
    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    logger.info(f"데이터 로드 완료: {len(ohlcv_dict)}종목")

    # ── 2. 종목풀 히스토리 구축 (1회) ──
    from main import _generate_rebalance_dates, _build_pool_history_factor_based
    rebalance_dates = _generate_rebalance_dates(start, end, config.factors.rebalance_freq)

    logger.info(f"종목풀 구축 시작 ({len(rebalance_dates)} 리밸런싱)")
    t0 = time.time()
    pool_history = _build_pool_history_factor_based(ohlcv_dict, rebalance_dates, build_stock_pool)
    pool_time = time.time() - t0
    logger.info(f"종목풀 구축 완료: {pool_time/60:.0f}분")

    # ── 3. Threshold 그리드 서치 ──
    thresholds = [0.15, 0.13, 0.11, 0.09, 0.07, 0.05]
    sell_threshold = config.timing.rl.sell_action_threshold
    results = {}

    for th in thresholds:
        logger.info(f"=== buy_threshold={th} 백테스트 시작 ===")
        config.timing.rl.buy_action_threshold = th
        t1 = time.time()

        strategy = FactorHybridStrategy(
            ml_model_path="models/xgboost_timing.pkl",
            rl_model_path="models/rl_timing.pt",
            params=config.strategy.params.model_dump(),
        )

        engine = PortfolioBacktestEngine(
            initial_capital=config.backtest.initial_capital,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
        )

        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)
        metrics = result.get("metrics", {})
        elapsed = time.time() - t1

        results[th] = metrics
        logger.info(
            f"  buy_th={th}: return={metrics.get('total_return',0):.1f}%, "
            f"sharpe={metrics.get('sharpe_ratio',0):.2f}, "
            f"mdd={metrics.get('max_drawdown',0):.1f}%, "
            f"trades={metrics.get('total_trades',0):.0f}, "
            f"time={elapsed/60:.0f}min"
        )

    # ── 4. 결과 출력 ──
    print("\n" + "=" * 70)
    print("  Hybrid buy_threshold Grid Search Results")
    print("=" * 70)

    header = f"{'Threshold':>10} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}"
    print(header)
    print("-" * 70)

    for th in thresholds:
        m = results[th]
        print(
            f"{th:>10.2f} "
            f"{m.get('total_return',0):>9.1f}% "
            f"{m.get('cagr',0):>7.1f}% "
            f"{m.get('sharpe_ratio',0):>8.2f} "
            f"{m.get('max_drawdown',0):>7.1f}% "
            f"{m.get('win_rate',0):>7.1f}% "
            f"{m.get('profit_factor',0):>6.2f} "
            f"{m.get('total_trades',0):>7.0f}"
        )

    # ── 5. Slack 전송 ──
    try:
        from slack_sdk import WebClient
        from src.config import get_secrets
        client = WebClient(token=get_secrets().slack_bot_token)

        lines = [
            "📊 *Hybrid buy_threshold 그리드 서치 결과*",
            f"sell_threshold={sell_threshold} 고정 | both 220팩터 | 시총5000억+ | 월간",
            "",
            "```",
            f"{'Threshold':>10} {'Return':>10} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'WinRate':>8} {'PF':>6} {'Trades':>7}",
            "-" * 70,
        ]
        for th in thresholds:
            m = results[th]
            lines.append(
                f"{th:>10.2f} "
                f"{m.get('total_return',0):>9.1f}% "
                f"{m.get('cagr',0):>7.1f}% "
                f"{m.get('sharpe_ratio',0):>8.2f} "
                f"{m.get('max_drawdown',0):>7.1f}% "
                f"{m.get('win_rate',0):>7.1f}% "
                f"{m.get('profit_factor',0):>6.2f} "
                f"{m.get('total_trades',0):>7.0f}"
            )
        lines.append("```")

        # 최적값 찾기 (Sharpe 기준)
        best_th = max(results, key=lambda t: results[t].get("sharpe_ratio", -999))
        best = results[best_th]
        lines.append("")
        lines.append(
            f"🏆 *최적 threshold={best_th}* "
            f"(Return {best.get('total_return',0):+.1f}%, "
            f"Sharpe {best.get('sharpe_ratio',0):.2f}, "
            f"MDD {best.get('max_drawdown',0):.1f}%)"
        )

        msg = "\n".join(lines)
        client.chat_postMessage(channel="C0AT0BM1AHF", text=msg)
        logger.info("Slack 전송 완료")
    except Exception as e:
        logger.error(f"Slack 전송 실패: {e}")


if __name__ == "__main__":
    main()
