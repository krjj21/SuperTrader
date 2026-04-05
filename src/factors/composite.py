"""
복합 점수 계산 모듈
- 동일 가중 또는 IC 가중 방식
- 팩터 결합 → 종합 점수
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_equal_weight_composite(
    factor_df: pd.DataFrame,
    valid_factors: list[str],
) -> pd.Series:
    """동일 가중 복합 점수를 계산합니다.

    Args:
        factor_df: 중립화/표준화된 팩터 매트릭스 (code index, factor columns)
        valid_factors: 유효한 팩터 이름 리스트

    Returns:
        종목별 복합 점수 (code index)
    """
    available = [f for f in valid_factors if f in factor_df.columns]
    if not available:
        return pd.Series(dtype=float)

    scores = factor_df[available].mean(axis=1)
    return scores.rename("composite_score")


def compute_ic_weighted_composite(
    factor_df: pd.DataFrame,
    valid_factors: list[str],
    factor_report: pd.DataFrame,
) -> pd.Series:
    """IC 가중 복합 점수를 계산합니다.

    각 팩터의 가중치는 |mean_IC|에 비례합니다.

    Args:
        factor_df: 중립화/표준화된 팩터 매트릭스
        valid_factors: 유효한 팩터 이름 리스트
        factor_report: 팩터 유효성 리포트 (validity.py 출력)

    Returns:
        종목별 복합 점수
    """
    available = [f for f in valid_factors if f in factor_df.columns and f in factor_report.index]
    if not available:
        return pd.Series(dtype=float)

    # IC 가중치 (절대값)
    ics = factor_report.loc[available, "mean_ic"].abs()
    weights = ics / ics.sum()

    # 가중 합산
    scores = (factor_df[available] * weights.values).sum(axis=1)
    return scores.rename("composite_score")


def compute_composite_score(
    factor_df: pd.DataFrame,
    valid_factors: list[str],
    factor_report: pd.DataFrame | None = None,
    method: str = "ic_weighted",
) -> pd.Series:
    """복합 점수를 계산합니다 (통합 인터페이스).

    Args:
        method: "equal" 또는 "ic_weighted"
    """
    if method == "ic_weighted" and factor_report is not None:
        score = compute_ic_weighted_composite(factor_df, valid_factors, factor_report)
    else:
        score = compute_equal_weight_composite(factor_df, valid_factors)

    logger.info(f"복합 점수 계산 완료: {len(score)}종목, method={method}")
    return score
