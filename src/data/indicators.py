"""
기술적 지표 계산 모듈
- ta 라이브러리 기반 RSI, MACD, Bollinger Bands, ATR, ADX, MFI
- KDJ 지표 수동 구현
- add_all_indicators()로 전체 지표 일괄 추가
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import MFIIndicator


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    return RSIIndicator(close=close, window=period).rsi()


def calc_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd = MACD(close=close, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd(), macd.macd_signal(), macd.macd_diff()


def calc_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    return AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()


def calc_bollinger(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    bb = BollingerBands(close=close, window=period, window_dev=std)
    return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()


def calc_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    return ADXIndicator(high=high, low=low, close=close, window=period).adx()


def calc_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    return MFIIndicator(
        high=high, low=low, close=close, volume=volume, window=period,
    ).money_flow_index()


def calc_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ 지표를 계산합니다.

    RSV = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100
    K = SMA(RSV, m1)
    D = SMA(K, m2)
    J = 3*K - 2*D

    Returns:
        (K, D, J)
    """
    lowest_low = low.rolling(window=n, min_periods=1).min()
    highest_high = high.rolling(window=n, min_periods=1).max()

    rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100

    # SMA 방식 (중국/한국 시장 표준)
    k = rsv.ewm(span=m1, adjust=False).mean()
    d = k.ewm(span=m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


def add_all_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """DataFrame에 전략에 필요한 모든 기술적 지표를 추가합니다.

    필요 컬럼: open, high, low, close, volume
    """
    params = params or {}
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # RSI
    df["rsi"] = calc_rsi(close, period=params.get("rsi_period", 14))

    # MACD
    macd_line, macd_signal_line, macd_hist = calc_macd(
        close,
        fast=params.get("macd_fast", 12),
        slow=params.get("macd_slow", 26),
        signal=params.get("macd_signal", 9),
    )
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal_line
    df["macd_hist"] = macd_hist

    # Bollinger Bands
    bb = BollingerBands(
        close=close,
        window=params.get("bb_period", 20),
        window_dev=params.get("bb_std", 2.0),
    )
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_pctb"] = bb.bollinger_pband()
    df["bb_bandwidth"] = bb.bollinger_wband()

    # ATR
    df["atr"] = calc_atr(high, low, close, period=params.get("atr_period", 14))

    # ADX + DI+/DI-
    adx_ind = ADXIndicator(
        high=high, low=low, close=close,
        window=params.get("atr_period", 14),
    )
    df["adx"] = adx_ind.adx()
    df["di_plus"] = adx_ind.adx_pos()
    df["di_minus"] = adx_ind.adx_neg()

    # MFI
    df["mfi"] = calc_mfi(
        high, low, close, df["volume"],
        period=params.get("mfi_period", 14),
    )

    # KDJ
    kdj_k, kdj_d, kdj_j = calc_kdj(
        high, low, close,
        n=params.get("kdj_n", 9),
        m1=params.get("kdj_m1", 3),
        m2=params.get("kdj_m2", 3),
    )
    df["kdj_k"] = kdj_k
    df["kdj_d"] = kdj_d
    df["kdj_j"] = kdj_j

    return df
