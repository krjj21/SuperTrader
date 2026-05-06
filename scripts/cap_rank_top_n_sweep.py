"""cap_rank × top_n 그리드 sweep — 어느 풀 구성이 최적 total_return 인가.

배경: commit 0358742 의 가정 ("절대값 모드가 +515% baseline") 이 사실 오류였음.
CLAUDE.md L161 의 +515% baseline 은 mid-cap rank 30-150 모드에서 측정된 것.
어느 (cap_rank_min, cap_rank_max, top_n) 조합이 실제로 어느 수익률을 내는지
14개 풀 (7 cap_rank pair × 2 top_n) 직접 측정.

Pool cache 는 cap_rank/top_n 변경 시 자동 무효화 (pool_cache._payload 에 둘 다 포함).
factor_panel 디스크 캐시는 ohlcv_dict 가 동일하면 재사용 → 풀 N회 재빌드의 비용을
크게 절감 (factor 계산은 1회, top_n 선정만 N회).

OHLCV 로드: 가장 큰 universe (cap_rank=0/0 절대값 모드) union 으로 1회만 로드.
다른 (rank-based) 페어는 그 부분집합이므로 동일 ohlcv_dict 재사용 가능.

Slack 자동 전송 (best 페어 + delta vs baseline).
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


CAP_RANK_PAIRS: list[tuple[int, int]] = [
    (0, 0),       # 절대값 모드 (현재 운영, 메가캡 포함)
    (1, 30),      # 초대형주 30위
    (1, 100),     # 대형주 100위
    (30, 100),    # large-mid
    (30, 150),    # 검증된 mid-cap baseline (+515%/+638%)
    (50, 200),    # mid
    (100, 300),   # mid-small
]
TOP_N_VALUES: list[int] = [15, 30]

KEY_METRICS = [
    "total_return", "cagr", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio", "win_rate", "profit_factor",
    "total_trades", "avg_holding_days",
]


def _run_one(
    cap_min: int,
    cap_max: int,
    top_n: int,
    ohlcv_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[str],
    model_paths: dict[str, str],
    idx: int,
    total: int,
) -> dict:
    cfg = get_config()
    cfg.universe.cap_rank_min = cap_min
    cfg.universe.cap_rank_max = cap_max
    cfg.factors.top_n = top_n

    label = f"cap=({cap_min},{cap_max})_top{top_n}"
    logger.info("=" * 60)
    logger.info(f"[{idx}/{total}] {label}")
    logger.info("=" * 60)

    # cap_rank/top_n 변경으로 pool_cache 자동 miss → 신규 빌드 (factor_panel 은 재사용됨)
    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )

    # OOS slice (train_ratio 적용)
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
        name=f"factor_hybrid_{label}",
    )
    engine = PortfolioBacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        commission_rate=cfg.backtest.commission_rate,
        tax_rate=cfg.backtest.tax_rate,
        max_positions=cfg.risk.max_total_positions,
    )
    try:
        result = engine.run(strategy, ohlcv_dict, pool_used, rebal_used)
    except Exception as e:
        logger.error(f"[{label}] engine.run 예외: {e}")
        return {}

    if "error" in result:
        logger.error(f"[{label}] 실패: {result['error']}")
        return {}

    metrics = result.get("metrics", {})
    if metrics:
        logger.info(
            f"[{label}] return={metrics.get('total_return',0):+.2f}, "
            f"sharpe={metrics.get('sharpe_ratio',0):.2f}, "
            f"mdd={metrics.get('max_drawdown',0):.2f}, "
            f"trades={metrics.get('total_trades',0):.0f}"
        )
    return metrics


def main() -> None:
    cfg = get_config()
    logger.info("cap_rank × top_n 그리드 sweep 시작")
    logger.info(
        f"grid: {len(CAP_RANK_PAIRS)} cap_rank pairs × {len(TOP_N_VALUES)} top_n "
        f"= {len(CAP_RANK_PAIRS) * len(TOP_N_VALUES)} pools"
    )

    # 원본 cfg 백업 (finally 에서 복원)
    orig_cap_min = cfg.universe.cap_rank_min
    orig_cap_max = cfg.universe.cap_rank_max
    orig_top_n = cfg.factors.top_n

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")

    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
    logger.info(f"리밸런싱: {len(rebalance_dates)}회 ({cfg.factors.rebalance_freq})")

    # universe union — cap_rank=0/0 절대값 모드 (가장 큰 universe) 로 1회 로드
    cfg.universe.cap_rank_min = 0
    cfg.universe.cap_rank_max = 0
    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        legacy = get_universe()
        all_codes = set(legacy["code"].tolist())
    codes = sorted(all_codes)
    logger.info(f"유니버스 (cap_rank=0/0 union): {len(codes)}종목")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)
    logger.info(f"OHLCV: {len(ohlcv_dict)}종목 로드")

    # 첫 페어로 모델 로드 (재학습 안 함, 기존 .pkl/.pt 사용)
    model_paths = _train_models_if_needed(ohlcv_dict)
    logger.info(f"모델: {model_paths}")

    rows: list[dict] = []
    total = len(CAP_RANK_PAIRS) * len(TOP_N_VALUES)
    idx = 0

    try:
        for cap_min, cap_max in CAP_RANK_PAIRS:
            for top_n in TOP_N_VALUES:
                idx += 1
                metrics = _run_one(
                    cap_min, cap_max, top_n,
                    ohlcv_dict, rebalance_dates, model_paths,
                    idx, total,
                )
                if not metrics:
                    logger.warning(
                        f"[{idx}/{total}] cap=({cap_min},{cap_max}) top_n={top_n} "
                        f"실패 — sweep 계속"
                    )
                    continue
                row = {"cap_rank_min": cap_min, "cap_rank_max": cap_max, "top_n": top_n}
                row.update({k: metrics.get(k) for k in KEY_METRICS})
                rows.append(row)
    finally:
        # cfg 원복 (in-memory 만 변경했지만 안전을 위해)
        cfg.universe.cap_rank_min = orig_cap_min
        cfg.universe.cap_rank_max = orig_cap_max
        cfg.factors.top_n = orig_top_n
        logger.info(
            f"cfg 원복: cap_rank=({orig_cap_min},{orig_cap_max}), top_n={orig_top_n}"
        )

    if not rows:
        logger.error("결과 없음 — 모든 페어 실패")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["total_return", "sharpe_ratio"], ascending=[False, False])
    df = df.reset_index(drop=True)

    # Sanity check 메모
    sanity_lines = []
    abs_baseline = df[(df.cap_rank_min == 0) & (df.cap_rank_max == 0) & (df.top_n == 15)]
    midcap_baseline = df[(df.cap_rank_min == 30) & (df.cap_rank_max == 150) & (df.top_n == 15)]
    if not abs_baseline.empty:
        ret = abs_baseline["total_return"].iloc[0]
        ok = abs(ret - 0.064) < 0.05
        sanity_lines.append(
            f"- (0,0,15) total_return={ret:+.2%} "
            f"(현재 운영 기준 6.4% — {'OK' if ok else 'WARN: ±5%p 초과'})"
        )
    if not midcap_baseline.empty:
        ret = midcap_baseline["total_return"].iloc[0]
        ok = 4.0 < ret < 7.5  # 515%~640% 범위
        sanity_lines.append(
            f"- (30,150,15) total_return={ret:+.2%} "
            f"(CLAUDE.md baseline 515%~640% — {'OK' if ok else 'WARN: 범위 밖'})"
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = ROOT / "reports"
    csv_path = reports_dir / f"cap_rank_top_n_sweep_{stamp}.csv"
    md_path = reports_dir / f"cap_rank_top_n_sweep_{stamp}.md"

    df.to_csv(csv_path, index=False, encoding="utf-8")

    md_lines = [
        f"# cap_rank × top_n grid sweep — {stamp}",
        "",
        f"strategy=factor_hybrid (XGB+RL), period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
        f"rebalance={cfg.factors.rebalance_freq}, train_ratio={cfg.backtest.train_ratio}",
        "",
        f"총 {len(rows)}/{total} 페어 완료. total_return DESC, sharpe_ratio DESC 정렬.",
        "",
        "## 결과",
        "",
        "```",
        df.to_string(index=False),
        "```",
        "",
    ]
    if sanity_lines:
        md_lines += ["## Sanity check", ""] + sanity_lines + [""]

    top3 = df.head(3)
    md_lines += [
        "## 상위 3개",
        "",
        "```",
        top3.to_string(index=False),
        "```",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info(f"\n{'='*60}\n결과:\n{df.to_string(index=False)}\n{'='*60}")
    logger.info(f"저장: {csv_path}")
    logger.info(f"저장: {md_path}")
    for line in sanity_lines:
        logger.info(line)

    # Slack
    try:
        from src.notification.slack_bot import SlackNotifier
        notifier = SlackNotifier()
        if notifier.token:
            best = df.iloc[0]
            msg_parts = [
                "🔬 *cap_rank × top_n 그리드 sweep* (factor_hybrid)",
                f"period={cfg.backtest.start_date}~{cfg.backtest.end_date}, "
                f"rebalance={cfg.factors.rebalance_freq}, train_ratio={cfg.backtest.train_ratio}",
                f"총 {len(rows)}/{total} 페어 완료",
                "",
                "```",
                df.to_string(index=False),
                "```",
                "",
                f"🏆 *peak total_return*: cap=({int(best.cap_rank_min)},{int(best.cap_rank_max)}) "
                f"top_n={int(best.top_n)} "
                f"return={best.total_return:+.2%}, sharpe={best.sharpe_ratio:.2f}, "
                f"mdd={best.max_drawdown:+.2%}, trades={int(best.total_trades)}",
            ]
            if sanity_lines:
                msg_parts += ["", "*Sanity*:"] + sanity_lines
            msg_parts.append(f"\n리포트: `{csv_path.name}`")
            ok = notifier._send("\n".join(msg_parts))
            logger.info(f"Slack 전송: {'OK' if ok else 'FAIL'}")
        else:
            logger.warning("Slack 토큰 없음 — 전송 생략")
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


if __name__ == "__main__":
    main()
