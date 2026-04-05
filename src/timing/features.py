"""
피처 엔지니어링 모듈
- MACD/KDJ 파생 피처
- 기술적 지표 파생
- 팩터 기반 피처
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.indicators import add_all_indicators


def build_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """단일 종목 OHLCV에서 ML 타이밍 피처를 생성합니다.

    Args:
        df: OHLCV DataFrame (date, open, high, low, close, volume)

    Returns:
        피처 DataFrame (동일 인덱스)
    """
    params = params or {}
    df = df.copy()
    df = add_all_indicators(df, params)

    features = pd.DataFrame(index=df.index)

    # ── MACD 파생 피처 ──
    features["macd_value"] = df["macd"]
    features["macd_signal_value"] = df["macd_signal"]
    features["macd_hist_value"] = df["macd_hist"]
    features["macd_hist_direction"] = np.sign(df["macd_hist"])
    features["macd_hist_change"] = df["macd_hist"] - df["macd_hist"].shift(1)
    features["macd_hist_accel"] = features["macd_hist_change"] - features["macd_hist_change"].shift(1)
    features["macd_zero_cross"] = (
        (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    ).astype(float) - (
        (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
    ).astype(float)

    # ── KDJ 파생 피처 ──
    features["kdj_k"] = df["kdj_k"]
    features["kdj_d"] = df["kdj_d"]
    features["kdj_j"] = df["kdj_j"]
    features["kdj_kd_spread"] = df["kdj_k"] - df["kdj_d"]
    features["kdj_j_overbought"] = (df["kdj_j"] > 80).astype(float)
    features["kdj_j_oversold"] = (df["kdj_j"] < 20).astype(float)
    features["kdj_cross"] = (
        (df["kdj_k"] > df["kdj_d"]) & (df["kdj_k"].shift(1) <= df["kdj_d"].shift(1))
    ).astype(float) - (
        (df["kdj_k"] < df["kdj_d"]) & (df["kdj_k"].shift(1) >= df["kdj_d"].shift(1))
    ).astype(float)

    # ── RSI 피처 ──
    features["rsi"] = df["rsi"]
    features["rsi_change"] = df["rsi"] - df["rsi"].shift(1)
    features["rsi_overbought"] = (df["rsi"] > 70).astype(float)
    features["rsi_oversold"] = (df["rsi"] < 30).astype(float)

    # ── 볼린저밴드 피처 ──
    features["bb_pctb"] = df["bb_pctb"]
    features["bb_bandwidth"] = df["bb_bandwidth"]

    # ── ATR 피처 ──
    features["atr_ratio"] = df["atr"] / (df["close"] + 1e-10)

    # ── ADX 피처 ──
    features["adx"] = df["adx"]
    features["di_spread"] = df["di_plus"] - df["di_minus"]

    # ── MFI 피처 ──
    features["mfi"] = df["mfi"]

    # ── 거래량 비율 ──
    features["volume_ratio"] = df["volume"] / (df["volume"].rolling(20, min_periods=1).mean() + 1)

    # ── 수익률 피처 ──
    features["ret_1d"] = df["close"].pct_change(1)
    features["ret_5d"] = df["close"].pct_change(5)
    features["ret_20d"] = df["close"].pct_change(20)

    # ── 이동평균 위치 ──
    features["close_ma5_ratio"] = df["close"] / (df["close"].rolling(5).mean() + 1e-10)
    features["close_ma20_ratio"] = df["close"] / (df["close"].rolling(20).mean() + 1e-10)

    return features.replace([np.inf, -np.inf], np.nan)


def get_feature_names() -> list[str]:
    """피처 이름 리스트를 반환합니다."""
    # 더미 데이터로 피처 이름 추출
    dummy = pd.DataFrame({
        "open": [100]*50, "high": [105]*50, "low": [95]*50,
        "close": range(100, 150), "volume": [1000]*50,
    })
    features = build_features(dummy)
    return features.columns.tolist()
