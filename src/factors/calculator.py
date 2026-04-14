"""
팩터 계산 모듈
- 전 종목 팩터 값 일괄 계산
- 펀더멘털 팩터 추가
- 크로스섹션 팩터 매트릭스 생성
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.factors.alpha101 import compute_all_factors, get_all_factor_names
from src.data.market_data import get_ohlcv_batch, get_close_prices
from src.data.factor_data import get_fundamentals, get_market_cap


def compute_cross_sectional_factors(
    codes: list[str],
    date: str,
    lookback_days: int = 300,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """주어진 날짜의 크로스섹션 팩터 매트릭스를 계산합니다.

    Args:
        codes: 종목코드 리스트
        date: 기준일 (YYYYMMDD)
        lookback_days: OHLCV lookback 기간
        ohlcv_dict: 사전 로드된 OHLCV 데이터 (None이면 자동 로드)

    Returns:
        DataFrame [code index, factor columns] - 각 종목의 최신 팩터 값
    """
    if ohlcv_dict is None:
        # 시작일 계산
        from datetime import datetime, timedelta
        end_dt = datetime.strptime(date, "%Y%m%d")
        start_dt = end_dt - timedelta(days=lookback_days)
        start = start_dt.strftime("%Y%m%d")

        # OHLCV 배치 로드
        ohlcv_dict = get_ohlcv_batch(codes, start, date)
    else:
        # 사전 로드된 데이터를 기준일까지 필터링
        dt = pd.Timestamp(date)
        filtered = {}
        for code in codes:
            if code not in ohlcv_dict:
                continue
            df = ohlcv_dict[code]
            mask = df["date"] <= dt
            df_until = df[mask]
            if len(df_until) >= 60:
                filtered[code] = df_until
        ohlcv_dict = filtered

    # 각 종목의 OHLCV 기반 팩터 계산
    factor_rows = {}
    for code, df in ohlcv_dict.items():
        if len(df) < 60:  # 최소 60일 데이터 필요
            continue
        try:
            factors = compute_all_factors(df)
            # 가장 최근 값 사용
            factor_rows[code] = factors.iloc[-1]
        except Exception as e:
            logger.debug(f"팩터 계산 실패: {code} - {e}")

    if not factor_rows:
        return pd.DataFrame()

    factor_df = pd.DataFrame(factor_rows).T
    factor_df.index.name = "code"

    # 펀더멘털 팩터 추가
    fund_df = _get_fundamental_factors(date)
    if not fund_df.empty:
        factor_df = factor_df.join(fund_df, how="left")

    logger.info(
        f"크로스섹션 팩터 계산 완료: {len(factor_df)}종목 × {len(factor_df.columns)}팩터 "
        f"(date={date})"
    )
    return factor_df


def _get_fundamental_factors(date: str) -> pd.DataFrame:
    """펀더멘털 기반 팩터를 계산합니다."""
    try:
        fund = get_fundamentals(date)
        cap = get_market_cap(date)
    except Exception as e:
        logger.warning(f"펀더멘털 데이터 로드 실패: {e}")
        return pd.DataFrame()

    if fund.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=fund["code"])

    # 가치 팩터
    per = fund.set_index("code")["per"]
    result["earnings_yield"] = 1.0 / per.replace(0, np.nan)  # 1/PER
    pbr = fund.set_index("code")["pbr"]
    result["book_yield"] = 1.0 / pbr.replace(0, np.nan)  # 1/PBR
    result["div_yield"] = fund.set_index("code")["div_yield"]

    # 퀄리티 팩터
    # ROE ≈ PBR / PER (근사)
    result["roe_approx"] = pbr / per.replace(0, np.nan)

    # EPS
    result["eps"] = fund.set_index("code")["eps"]

    # 사이즈 팩터
    if not cap.empty:
        mcap = cap.set_index("code")["market_cap"]
        result["log_market_cap"] = np.log(mcap.replace(0, np.nan))
        result["neg_log_market_cap"] = -result["log_market_cap"]  # 소형주 선호

        # 유동성 팩터
        result["avg_volume"] = cap.set_index("code")["volume"]

    result.index.name = "code"
    return result


def compute_factor_history(
    codes: list[str],
    dates: list[str],
    lookback_days: int = 300,
) -> dict[str, pd.DataFrame]:
    """여러 날짜에 대한 팩터 히스토리를 계산합니다.

    Returns:
        {date: factor_matrix DataFrame} 딕셔너리
    """
    history = {}
    for i, date in enumerate(dates):
        logger.info(f"팩터 히스토리 계산: {i+1}/{len(dates)} ({date})")
        factor_df = compute_cross_sectional_factors(codes, date, lookback_days)
        if not factor_df.empty:
            history[date] = factor_df
    return history
