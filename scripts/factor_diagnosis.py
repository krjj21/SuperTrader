"""B4 — 220 factor 진단.

7년 (backtest 기간) 의 factor_history × return_history 로 IC 측정.
- 양/음수 IC 분포
- IR 분포 + min_ir=0.3 통과율
- 카테고리별 mean IC / IR
- top 10 / bottom 10 factor (IR 절대값 기준)
- 한국 시장 적합성 진단
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
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
    _forward_returns_between,
)
from src.factors.calculator import compute_cross_sectional_factors  # noqa: E402
from src.factors import pool_cache  # noqa: E402
from src.factors.validity import validate_all_factors  # noqa: E402
from src.factors.neutralizer import neutralize_factor_matrix  # noqa: E402
from src.regime.weights import get_factor_category  # noqa: E402


def main():
    cfg = get_config()
    logger.info("Factor 진단 시작 (220 factor IC 분포)")

    start = cfg.backtest.start_date.replace("-", "")
    end = cfg.backtest.end_date.replace("-", "")
    rebalance_dates = _generate_rebalance_dates(start, end, cfg.factors.rebalance_freq)
    logger.info(f"리밸런싱: {len(rebalance_dates)}회 ({cfg.factors.rebalance_freq})")

    # universe + OHLCV
    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        legacy = get_universe()
        all_codes = set(legacy["code"].tolist())
    codes = sorted(all_codes)
    logger.info(f"유니버스: {len(codes)}")

    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)
    logger.info(f"OHLCV: {len(ohlcv_dict)}")

    # factor_panel 로드 (이미 cache 있으면) — 없으면 빌드
    factor_panel = pool_cache.load_factor_panel()
    if factor_panel is None:
        logger.info(f"factor_panel 캐시 부재 — 빌드 시작 ({len(ohlcv_dict)} 종목)")
        from src.factors.calculator import build_factor_panel
        factor_panel = build_factor_panel(ohlcv_dict)
        if factor_panel:
            pool_cache.save_factor_panel(factor_panel)
            logger.info(f"factor_panel 빌드/저장 완료: {len(factor_panel)} 종목")
    logger.info(f"factor_panel: {len(factor_panel)} 종목")

    # factor_history + return_history 누적 (전체 rebalance)
    factor_history: dict[str, pd.DataFrame] = {}
    return_history: dict[str, pd.Series] = {}

    for i, date in enumerate(rebalance_dates):
        date_fmt = date.replace("-", "")
        if i > 0:
            prev_fmt = rebalance_dates[i - 1].replace("-", "")
            ret = _forward_returns_between(ohlcv_dict, prev_fmt, date_fmt)
            if not ret.empty:
                return_history[prev_fmt] = ret

        try:
            factor_df = compute_cross_sectional_factors(
                codes=list(ohlcv_dict.keys()),
                date=date_fmt,
                ohlcv_dict=ohlcv_dict,
                factor_panel=factor_panel,
            )
            # 중립화 (실제 운영과 동일)
            factor_df = neutralize_factor_matrix(
                factor_df,
                date=date_fmt,
                ohlcv_dict=ohlcv_dict,
                neutralize_industry=cfg.factors.neutralize_industry,
                neutralize_market_cap=cfg.factors.neutralize_market_cap,
            )
            if factor_df is not None and not factor_df.empty:
                factor_history[date_fmt] = factor_df
        except Exception as e:
            logger.debug(f"{date} 팩터 계산 실패: {e}")
            continue

        if (i + 1) % 30 == 0:
            logger.info(f"  factor_history 누적 {i+1}/{len(rebalance_dates)}")

    logger.info(
        f"factor_history: {len(factor_history)} 시점, "
        f"return_history: {len(return_history)} 시점"
    )

    # IC 측정 — min_ir=0 으로 모든 factor 분석
    report = validate_all_factors(factor_history, return_history, min_ir=0.0)
    logger.info(f"factor_report: {len(report)} factor 분석")

    # 카테고리 매핑
    report["category"] = report.index.map(lambda f: get_factor_category(f) or "unknown")

    # 1. 양/음수 IC 분포
    pos = (report["mean_ic"] > 0).sum()
    neg = (report["mean_ic"] < 0).sum()
    zero = (report["mean_ic"] == 0).sum()
    print()
    print("=" * 78)
    print(f"220 Factor IC 분포 (전체 {len(report)} factor)")
    print("=" * 78)
    print(f"양수 IC (mean_ic > 0): {pos:>4d} ({pos/len(report)*100:.1f}%)")
    print(f"음수 IC (mean_ic < 0): {neg:>4d} ({neg/len(report)*100:.1f}%)")
    print(f"제로 IC (mean_ic = 0): {zero:>4d} ({zero/len(report)*100:.1f}%)")
    print()

    # 2. IC / IR 통계
    print(f"mean_ic       — mean: {report['mean_ic'].mean():+.4f}, "
          f"std: {report['mean_ic'].std():.4f}, "
          f"min: {report['mean_ic'].min():+.4f}, max: {report['mean_ic'].max():+.4f}")
    print(f"ir            — mean: {report['ir'].mean():+.4f}, "
          f"std: {report['ir'].std():.4f}, "
          f"|ir|>=0.3: {(report['ir'].abs() >= 0.3).sum()} ({(report['ir'].abs() >= 0.3).mean()*100:.1f}%)")
    if "t_stat" in report.columns:
        print(f"t_stat        — mean: {report['t_stat'].mean():+.4f}, "
              f"|t|>=2.0: {(report['t_stat'].abs() >= 2.0).sum()} ({(report['t_stat'].abs() >= 2.0).mean()*100:.1f}%)")
    print(f"ic_positive_ratio — mean: {report['ic_positive_ratio'].mean():.3f} "
          f"(0.5=무관, 1.0=항상양수)")

    # 3. 카테고리별 통계
    print()
    print("=" * 78)
    print("카테고리별 IC / IR")
    print("=" * 78)
    cat_stats = report.groupby("category").agg(
        n=("mean_ic", "size"),
        mean_ic=("mean_ic", "mean"),
        median_ic=("mean_ic", "median"),
        mean_ir=("ir", "mean"),
        valid_ir=("ir", lambda x: (x.abs() >= 0.3).sum()),
        pos_ratio=("ic_positive_ratio", "mean"),
    ).round(4)
    print(cat_stats.to_string())

    # 4. Top 10 (IR 절대값) + Bottom 10
    print()
    print("=" * 78)
    print("TOP 10 factor (|IR| 큰 순)")
    print("=" * 78)
    top = report.reindex(report["ir"].abs().sort_values(ascending=False).index).head(10)
    cols_show = ["category", "mean_ic", "ir", "ic_positive_ratio"]
    if "t_stat" in top.columns:
        cols_show.insert(3, "t_stat")
    print(top[cols_show].to_string())

    print()
    print("=" * 78)
    print("BOTTOM 10 factor (|IR| 가장 작은 = noise)")
    print("=" * 78)
    bot = report.reindex(report["ir"].abs().sort_values().index).head(10)
    print(bot[cols_show].to_string())

    # 5. 진단 결론
    print()
    print("=" * 78)
    print("진단 결론")
    print("=" * 78)
    valid_pct = (report["ir"].abs() >= 0.3).mean() * 100
    pos_neg_balance = pos / max(len(report), 1)
    avg_ic_pos_ratio = report["ic_positive_ratio"].mean()
    if valid_pct < 30:
        print(f"❌ |IR|>=0.3 통과율 {valid_pct:.1f}% — 절반 이상 factor 가 noise")
    elif valid_pct < 50:
        print(f"🟡 |IR|>=0.3 통과율 {valid_pct:.1f}% — factor 품질 보통")
    else:
        print(f"🟢 |IR|>=0.3 통과율 {valid_pct:.1f}% — factor 품질 양호")
    if pos_neg_balance < 0.4:
        print(f"❌ 양수 IC 비율 {pos_neg_balance*100:.1f}% < 40% — 음수 IC factor 다수, ic_signed 검토")
    elif pos_neg_balance > 0.6:
        print(f"🟢 양수 IC 비율 {pos_neg_balance*100:.1f}% — 정방향 factor 다수")
    else:
        print(f"🟡 양수 IC 비율 {pos_neg_balance*100:.1f}% — 균형, ic_signed 가치 있음")
    if avg_ic_pos_ratio < 0.45:
        print(f"❌ ic_positive_ratio 평균 {avg_ic_pos_ratio:.2f} — IC 부호 변동 심함, 시간 안정성 낮음")
    elif avg_ic_pos_ratio > 0.55:
        print(f"🟢 ic_positive_ratio 평균 {avg_ic_pos_ratio:.2f} — IC 안정")

    # 저장
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/factor_diagnosis_{stamp}.csv"
    report.to_csv(csv_path, encoding="utf-8")
    cat_csv = ROOT / f"reports/factor_diagnosis_category_{stamp}.csv"
    cat_stats.to_csv(cat_csv, encoding="utf-8")
    logger.info(f"저장: {csv_path}")
    logger.info(f"저장: {cat_csv}")

    # Slack
    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            top5 = top[cols_show].head(5).to_string()
            cat_summary = cat_stats.to_string()
            msg = (
                "🔬 *B4 — 220 Factor 진단*\n"
                f"기간 {cfg.backtest.start_date}~{cfg.backtest.end_date}, {len(rebalance_dates)} rebalance\n\n"
                f"*분포*: 양수 IC {pos} ({pos/len(report)*100:.0f}%), "
                f"음수 IC {neg} ({neg/len(report)*100:.0f}%)\n"
                f"|IR|>=0.3 통과: {(report['ir'].abs() >= 0.3).sum()} "
                f"({valid_pct:.1f}%)\n"
                f"mean IC: {report['mean_ic'].mean():+.4f}, "
                f"mean IR: {report['ir'].mean():+.4f}\n\n"
                f"*카테고리별*:\n```\n{cat_summary}\n```\n\n"
                f"*TOP 5 factor*:\n```\n{top5}\n```\n\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 예외: {e}")


if __name__ == "__main__":
    main()
