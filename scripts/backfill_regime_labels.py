"""백테스트 기간의 regime 라벨을 sappo_regime_labels 에 backfill.

forward_days_dynamic 토글 ON 시 학습 데이터의 모든 row 가 KOSPI regime 라벨을
조회하므로, 백테스트 기간 (예: 2018-01-02 ~ 2024-12-30) 의 모든 영업일에 라벨이
DB 에 있어야 한다. live 의 daily 08:45 cron 은 그 시점 이후만 채우므로 과거는
이 스크립트로 1회 backfill.

전략 (사용자 확정):
- LLM 점수: historical 뉴스 데이터 부재 → llm_score=0, overridden_by_llm=False
- _smooth_with_yesterday: 끔 (live 한정 휴리스틱; backfill 시 직전 라벨 의존성이
  복잡하므로 GMM raw label 만 저장)
- KOSPI 데이터: RegimeDetector._fetch_kospi(end=date) 로 매일 슬라이싱
- idempotent: 이미 DB 에 라벨이 있으면 skip (--force 로 덮어쓰기)

실행:
    ./venv/Scripts/python.exe scripts/backfill_regime_labels.py
    ./venv/Scripts/python.exe scripts/backfill_regime_labels.py --start 20180101 --end 20241230
    ./venv/Scripts/python.exe scripts/backfill_regime_labels.py --force
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.db.sappo_models import (  # noqa: E402
    init_sappo_db,
    upsert_regime_label,
    get_regime_for,
)
from src.regime.detector import (  # noqa: E402
    RegimeDetector,
    LABEL_REVERT,
)


def _parse_args() -> argparse.Namespace:
    cfg = get_config()
    p = argparse.ArgumentParser(description="sappo_regime_labels backfill")
    p.add_argument(
        "--start", type=str,
        default=cfg.backtest.start_date.replace("-", ""),
        help="YYYYMMDD (default: backtest.start_date)",
    )
    p.add_argument(
        "--end", type=str,
        default=cfg.backtest.end_date.replace("-", ""),
        help="YYYYMMDD (default: backtest.end_date)",
    )
    p.add_argument("--force", action="store_true", help="기존 라벨 덮어쓰기")
    return p.parse_args()


def _trading_days(start: str, end: str) -> list[str]:
    """KS11 일봉이 있는 영업일 YYYYMMDD 리스트."""
    import FinanceDataReader as fdr
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    df = fdr.DataReader("KS11", s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"))
    if df is None or df.empty:
        raise RuntimeError(f"KOSPI 데이터 없음: {start} ~ {end}")
    return [d.strftime("%Y%m%d") for d in df.index]


def main() -> None:
    args = _parse_args()
    init_sappo_db("data/trading.db")

    detector = RegimeDetector()
    days = _trading_days(args.start, args.end)
    logger.info(
        f"regime backfill: {args.start} ~ {args.end} ({len(days)} 영업일), "
        f"force={args.force}"
    )

    # GMM 적합에 lookback_days(=60) 만큼의 KOSPI 가 필요 → 시작 60일치는 fit 불가능.
    # 그래도 시도 후 RuntimeError 시 skip.
    n_done = 0
    n_skip = 0
    n_fail = 0
    for i, date_str in enumerate(days, 1):
        if not args.force:
            existing = get_regime_for(date_str)
            if existing is not None:
                n_skip += 1
                continue

        try:
            kospi = detector._fetch_kospi(end=date_str)
            feats = detector._build_features(kospi, end_date=date_str)
            min_len = max(20, detector.n_states * 5)
            if len(feats) < min_len:
                logger.debug(
                    f"  [{i}/{len(days)}] {date_str} skip (KOSPI feat {len(feats)} < {min_len})"
                )
                n_fail += 1
                continue
            cluster, post, cluster_to_label, ret_60d, vol_60d = detector._fit_and_assign(feats)
            label = cluster_to_label[cluster]
            from src.regime.detector import ALL_LABELS
            post_by_label = {lbl: 0.0 for lbl in ALL_LABELS}
            for k, lbl in cluster_to_label.items():
                post_by_label[lbl] = float(post[k])
            # LLM = 0 / overridden=False / smoothing 없음 (사용자 확정)
            upsert_regime_label(
                date=date_str,
                label=label,
                hmm_state=int(cluster),
                hmm_probs=(
                    post_by_label["risk_on_trend"],
                    post_by_label["high_vol_risk_off"],
                    post_by_label["mean_revert"],
                ),
                kospi_return_60d=ret_60d,
                kospi_vol_60d=vol_60d,
                llm_score=0.0,
                overridden_by_llm=False,
                notes="backfill_no_llm_no_smooth",
            )
            n_done += 1
            if i % 100 == 0 or i == len(days):
                logger.info(
                    f"  [{i}/{len(days)}] {date_str} label={label} "
                    f"(done={n_done}, skip={n_skip}, fail={n_fail})"
                )
        except Exception as e:
            logger.warning(f"  [{i}/{len(days)}] {date_str} 실패: {e}")
            n_fail += 1

    logger.info(
        f"backfill 완료: 신규={n_done}, 기존 skip={n_skip}, 실패={n_fail} / 전체 {len(days)}"
    )


if __name__ == "__main__":
    main()
