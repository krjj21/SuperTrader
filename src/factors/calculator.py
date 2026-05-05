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

from src.data.market_data import get_ohlcv_batch, get_close_prices
from src.data.factor_data import get_fundamentals, get_market_cap


def _get_factor_functions() -> tuple:
    """config에 따라 팩터 계산 함수를 반환합니다.

    Returns:
        (compute_all_factors, get_all_factor_names)
    """
    from src.config import get_config
    module = getattr(get_config().factors, "factor_module", "alpha101")

    if module == "alpha158":
        from src.factors.alpha158 import compute_all_factors, get_all_factor_names
        return compute_all_factors, get_all_factor_names
    elif module == "both":
        from src.factors.alpha101 import (
            compute_all_factors as compute_101,
            get_all_factor_names as names_101,
        )
        from src.factors.alpha158 import (
            compute_all_factors as compute_158,
            get_all_factor_names as names_158,
        )

        def compute_all_factors(df):
            return pd.concat([compute_101(df), compute_158(df)], axis=1)

        def get_all_factor_names():
            return names_101() + names_158()

        return compute_all_factors, get_all_factor_names
    else:
        from src.factors.alpha101 import compute_all_factors, get_all_factor_names
        return compute_all_factors, get_all_factor_names


def build_factor_panel(
    ohlcv_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """B1: 종목별 팩터 시계열을 1회 계산해 반환.

    `compute_all_factors(df)` 가 종목당 1회만 호출됨 (기존 84 리밸런스 × 30 종목 = 2520회 → 30회).
    각 리밸런스 시점의 cross-sectional 값은 panel[code].asof(date) 로 슬라이스.

    Args:
        ohlcv_dict: {code: full OHLCV DataFrame} (date 컬럼 포함, 정렬됨)

    Returns:
        {code: factor_df indexed by 'date' (Timestamp)}.
        compute_all_factors 실패 종목은 dict 에 미포함.
    """
    compute_fn, _ = _get_factor_functions()
    panel: dict[str, pd.DataFrame] = {}
    for code, df in ohlcv_dict.items():
        if df is None or len(df) < 60:
            continue
        try:
            factors = compute_fn(df)
            # date 인덱스로 정렬 — compute_*_factors 는 df.index 를 그대로 쓰므로
            # df["date"] 가 있으면 이를 인덱스로 변환해 asof 조회 가능하게.
            if "date" in df.columns:
                idx = pd.to_datetime(df["date"].values)
                factors = factors.copy()
                factors.index = idx
            panel[code] = factors
        except Exception as e:
            logger.debug(f"팩터 panel 계산 실패: {code} - {e}")
    if panel:
        first = next(iter(panel.values()))
        logger.info(
            f"팩터 panel 빌드 완료: {len(panel)}종목 × {len(first.columns)}팩터 "
            f"(평균 {sum(len(p) for p in panel.values())/max(len(panel),1):.0f}일)"
        )
    return panel


def _slice_panel_at_date(
    factor_panel: dict[str, pd.DataFrame],
    codes: list[str],
    date: str,
) -> pd.DataFrame:
    """factor_panel 에서 주어진 date 의 cross-section row 슬라이스.

    각 code 별로 date 이전 가장 최근 행 (asof) 사용.
    """
    target = pd.Timestamp(date)
    rows = {}
    for code in codes:
        panel_df = factor_panel.get(code)
        if panel_df is None or panel_df.empty:
            continue
        try:
            # ≤ target 중 가장 최근. 없으면 skip.
            mask = panel_df.index <= target
            sub = panel_df[mask]
            if not sub.empty:
                rows[code] = sub.iloc[-1]
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).T
    out.index.name = "code"
    return out


def compute_cross_sectional_factors(
    codes: list[str],
    date: str,
    lookback_days: int = 300,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
    factor_panel: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """주어진 날짜의 크로스섹션 팩터 매트릭스를 계산합니다.

    Args:
        codes: 종목코드 리스트
        date: 기준일 (YYYYMMDD)
        lookback_days: OHLCV lookback 기간
        ohlcv_dict: 사전 로드된 OHLCV 데이터 (None이면 자동 로드)
        factor_panel: B1 사전 계산된 팩터 패널 ({code: factor_df}). 주어지면
            compute_all_factors 재계산 우회 — 84 리밸런스의 220 팩터 × 30 종목
            계산을 1회 빌드 + 슬라이스로 대체.

    Returns:
        DataFrame [code index, factor columns] - 각 종목의 최신 팩터 값
    """
    # B1 fast-path: 사전 빌드된 panel 이 있으면 슬라이스만으로 OHLCV 재계산 우회.
    if factor_panel is not None and factor_panel:
        factor_df = _slice_panel_at_date(factor_panel, codes, date)
        if not factor_df.empty:
            fund_df = _get_fundamental_factors(date)
            if not fund_df.empty:
                factor_df = factor_df.join(fund_df, how="left")
            logger.debug(
                f"크로스섹션 팩터(panel-sliced): {len(factor_df)}종목 × "
                f"{len(factor_df.columns)}팩터 (date={date})"
            )
            return factor_df
        # panel 에 매칭 없으면 폴스루로 기존 계산 경로

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
    compute_fn, _ = _get_factor_functions()
    factor_rows = {}
    for code, df in ohlcv_dict.items():
        if len(df) < 60:  # 최소 60일 데이터 필요
            continue
        try:
            factors = compute_fn(df)
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
