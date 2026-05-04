"""USD/KRW + VIX 일자별 매크로 feature 백필.

FDR 'USD/KRW' / 'VIX' 7년치 → sappo_macro_features.
1회 실행 (~5분). 라이브 detect_regime_daily 가 매일 incremental upsert.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.db.sappo_models import init_sappo_db, upsert_macro_feature  # noqa: E402


def fetch_macro_series(start: str, end: str) -> pd.DataFrame:
    """FDR 로 USD/KRW + VIX 시계열을 받아 (date, usdkrw_close, vix_close) DataFrame."""
    import FinanceDataReader as fdr

    usdkrw = fdr.DataReader("USD/KRW", start, end)
    vix = fdr.DataReader("VIX", start, end)

    if usdkrw.empty:
        raise RuntimeError(f"USD/KRW empty {start}~{end}")
    if vix.empty:
        raise RuntimeError(f"VIX empty {start}~{end}")

    # Close 컬럼 추출 + 날짜 정렬
    df = pd.DataFrame({
        "usdkrw_close": usdkrw["Close"],
        "vix_close": vix["Close"],
    })
    # 두 시계열 결측 보간 (다른 거래일정에 의한 결측은 forward-fill 한 후 drop)
    df = df.ffill().dropna()
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """log-return, 20일 rolling vol 계산."""
    df = df.copy()
    df["usdkrw_log_ret"] = np.log(df["usdkrw_close"] / df["usdkrw_close"].shift(1))
    df["usdkrw_vol_20d"] = df["usdkrw_log_ret"].rolling(20).std()
    df["vix_log_ret"] = np.log(df["vix_close"] / df["vix_close"].shift(1))
    df = df.fillna(0.0)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="종료일 (default: today)")
    args = parser.parse_args()

    cfg = get_config()
    init_sappo_db(cfg.database.path)

    end = args.end or pd.Timestamp.today().strftime("%Y-%m-%d")
    logger.info(f"매크로 feature 백필: {args.start} ~ {end}")

    df = fetch_macro_series(args.start, end)
    df = compute_features(df)
    logger.info(f"받은 시계열: {len(df)}행")

    n = 0
    for ts, row in df.iterrows():
        date_str = ts.strftime("%Y%m%d")
        upsert_macro_feature(
            date=date_str,
            usdkrw_close=float(row["usdkrw_close"]),
            usdkrw_log_ret=float(row["usdkrw_log_ret"]),
            usdkrw_vol_20d=float(row["usdkrw_vol_20d"]),
            vix_close=float(row["vix_close"]),
            vix_log_ret=float(row["vix_log_ret"]),
        )
        n += 1
        if n % 200 == 0:
            logger.info(f"진행: {n}/{len(df)}")

    logger.info(f"백필 완료: {n}일자")


if __name__ == "__main__":
    main()
