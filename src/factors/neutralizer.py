"""
팩터 바이어스 보정 모듈 (Du 2025)
- 산업 중립화: 산업 더미 회귀 잔차
- 시가총액 중립화: log(시총) + log(시총)² 회귀 잔차
- 이상치 처리: MAD 기반 윈저화
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def winsorize_mad(s: pd.Series, n_mad: float = 5.0) -> pd.Series:
    """MAD (Median Absolute Deviation) 기반 윈저화.

    Args:
        s: 팩터 값 Series
        n_mad: MAD 배수 (기본 5)
    """
    median = s.median()
    mad = (s - median).abs().median()
    if mad < 1e-10:
        return s
    upper = median + n_mad * mad
    lower = median - n_mad * mad
    return s.clip(lower, upper)


def standardize(s: pd.Series) -> pd.Series:
    """크로스섹션 Z-score 표준화."""
    mean = s.mean()
    std = s.std()
    if std < 1e-10:
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def neutralize_industry(
    factor_values: pd.Series,
    industry: pd.Series,
) -> pd.Series:
    """산업 중립화: 산업 더미 변수 회귀 잔차를 반환합니다.

    Args:
        factor_values: 종목별 팩터 값 (code index)
        industry: 종목별 산업 코드 (code index)

    Returns:
        산업 중립화된 팩터 값
    """
    # 공통 인덱스
    common = factor_values.dropna().index.intersection(industry.dropna().index)
    if len(common) < 10:
        return factor_values

    y = factor_values.loc[common].values
    dummies = pd.get_dummies(industry.loc[common], drop_first=True)

    if dummies.empty or dummies.shape[1] == 0:
        return factor_values

    X = dummies.values.astype(float)

    # OLS 회귀 잔차
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        return factor_values

    result = factor_values.copy()
    result.loc[common] = residuals
    return result


def neutralize_market_cap(
    factor_values: pd.Series,
    log_market_cap: pd.Series,
) -> pd.Series:
    """시가총액 중립화: log(시총) + log(시총)² 회귀 잔차를 반환합니다.

    Args:
        factor_values: 종목별 팩터 값 (code index)
        log_market_cap: 종목별 log(시가총액) (code index)

    Returns:
        시가총액 중립화된 팩터 값
    """
    common = factor_values.dropna().index.intersection(log_market_cap.dropna().index)
    if len(common) < 10:
        return factor_values

    y = factor_values.loc[common].values
    x1 = log_market_cap.loc[common].values
    x2 = x1 ** 2
    X = np.column_stack([np.ones(len(common)), x1, x2])

    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
    except np.linalg.LinAlgError:
        return factor_values

    result = factor_values.copy()
    result.loc[common] = residuals
    return result


def neutralize_factor_matrix(
    factor_df: pd.DataFrame,
    industry: pd.Series | None = None,
    log_market_cap: pd.Series | None = None,
    do_industry: bool = True,
    do_market_cap: bool = True,
) -> pd.DataFrame:
    """팩터 매트릭스 전체에 바이어스 보정을 적용합니다.

    Args:
        factor_df: 팩터 매트릭스 (code index, factor columns)
        industry: 종목별 산업 코드
        log_market_cap: 종목별 log(시가총액)

    Returns:
        보정된 팩터 매트릭스
    """
    result = factor_df.copy()

    for col in result.columns:
        # 1. 이상치 처리
        result[col] = winsorize_mad(result[col])

        # 2. 산업 중립화
        if do_industry and industry is not None:
            result[col] = neutralize_industry(result[col], industry)

        # 3. 시가총액 중립화
        if do_market_cap and log_market_cap is not None:
            result[col] = neutralize_market_cap(result[col], log_market_cap)

        # 4. 표준화
        result[col] = standardize(result[col])

    logger.info(
        f"팩터 중립화 완료: {len(result.columns)}팩터 "
        f"(산업={do_industry}, 시총={do_market_cap})"
    )
    return result
