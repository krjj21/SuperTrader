"""
라벨 생성 모듈
- 미래 N일 수익률 기반 3클래스 라벨
- BUY (1), HOLD (0), SELL (-1)
"""
from __future__ import annotations

import pandas as pd


def generate_labels(
    close: pd.Series,
    forward_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pd.Series:
    """미래 수익률 기반 라벨을 생성합니다.

    Args:
        close: 종가 시리즈
        forward_days: 미래 수익률 계산 기간
        buy_threshold: 매수 임계값 (예: 0.02 = +2%)
        sell_threshold: 매도 임계값 (예: -0.02 = -2%)

    Returns:
        라벨 시리즈: 1 (BUY), 0 (HOLD), -1 (SELL)
    """
    forward_return = close.shift(-forward_days) / close - 1

    labels = pd.Series(0, index=close.index, dtype=int)
    labels[forward_return >= buy_threshold] = 1
    labels[forward_return <= sell_threshold] = -1

    # 미래 데이터가 없는 마지막 N일은 NaN
    labels.iloc[-forward_days:] = None

    return labels.rename("label")


def generate_forward_returns(
    close: pd.Series,
    forward_days: int = 5,
) -> pd.Series:
    """미래 수익률을 계산합니다 (IC 계산용)."""
    return (close.shift(-forward_days) / close - 1).rename("forward_return")
