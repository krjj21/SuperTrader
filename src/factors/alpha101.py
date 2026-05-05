"""
Alpha101 스타일 팩터 구현
- 가격/수익률 기반 (~30개)
- 거래량 기반 (~20개)
- 변동성 기반 (~15개)
- 펀더멘털 기반 (~15개)
- 기술적 지표 파생 (~20개)

각 팩터는 cross-sectional 값을 반환하는 함수.
입력: OHLCV DataFrame (date, open, high, low, close, volume)
출력: pd.Series (날짜별 팩터 값)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════

def _rank(s: pd.Series) -> pd.Series:
    """Causal expanding percentile rank (0~1).

    이전: `s.rank(pct=True)` — full-series 랭킹이라 panel 1회 빌드 시 미래 데이터 leakage.
    변경: `s.expanding().rank(pct=True)` — 각 행 D에서 [0..D] prefix 내 랭킹 (causal).
    consumed 값은 동일: 기존 per-rebalance 경로도 잘린 df_until 의 .iloc[-1] = 같은 prefix rank.
    """
    return s.expanding().rank(pct=True)


def _ts_rank(s: pd.Series, window: int) -> pd.Series:
    """Time-series rank over rolling window."""
    return s.rolling(window, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


def _ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window, min_periods=max(window // 2, 2)).corr(y)


def _ts_std(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).std()


def _ts_mean(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).mean()


def _ts_sum(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).sum()


def _ts_max(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).max()


def _ts_min(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=1).min()


def _delta(s: pd.Series, period: int) -> pd.Series:
    return s - s.shift(period)


def _ret(close: pd.Series, period: int) -> pd.Series:
    return close.pct_change(period)


# ═══════════════════════════════════════════════
# 가격/수익률 기반 팩터 (~30개)
# ═══════════════════════════════════════════════

def ret_1d(df: pd.DataFrame) -> pd.Series:
    """1일 수익률"""
    return _ret(df["close"], 1)

def ret_5d(df: pd.DataFrame) -> pd.Series:
    """5일 수익률"""
    return _ret(df["close"], 5)

def ret_10d(df: pd.DataFrame) -> pd.Series:
    """10일 수익률"""
    return _ret(df["close"], 10)

def ret_20d(df: pd.DataFrame) -> pd.Series:
    """20일(1개월) 수익률"""
    return _ret(df["close"], 20)

def ret_60d(df: pd.DataFrame) -> pd.Series:
    """60일(3개월) 수익률"""
    return _ret(df["close"], 60)

def ret_120d(df: pd.DataFrame) -> pd.Series:
    """120일(6개월) 수익률"""
    return _ret(df["close"], 120)

def ret_250d(df: pd.DataFrame) -> pd.Series:
    """250일(12개월) 수익률"""
    return _ret(df["close"], 250)

def momentum_12m_1m(df: pd.DataFrame) -> pd.Series:
    """12개월 모멘텀 - 1개월 반전 (가장 유명한 모멘텀 팩터)"""
    return _ret(df["close"], 250) - _ret(df["close"], 20)

def reversal_5d(df: pd.DataFrame) -> pd.Series:
    """5일 단기 반전"""
    return -_ret(df["close"], 5)

def reversal_20d(df: pd.DataFrame) -> pd.Series:
    """20일 단기 반전"""
    return -_ret(df["close"], 20)

def close_to_ma5(df: pd.DataFrame) -> pd.Series:
    """종가 / 5일 이동평균"""
    return df["close"] / _ts_mean(df["close"], 5)

def close_to_ma20(df: pd.DataFrame) -> pd.Series:
    """종가 / 20일 이동평균"""
    return df["close"] / _ts_mean(df["close"], 20)

def close_to_ma60(df: pd.DataFrame) -> pd.Series:
    """종가 / 60일 이동평균"""
    return df["close"] / _ts_mean(df["close"], 60)

def close_to_ma120(df: pd.DataFrame) -> pd.Series:
    """종가 / 120일 이동평균"""
    return df["close"] / _ts_mean(df["close"], 120)

def ma5_to_ma20(df: pd.DataFrame) -> pd.Series:
    """5일MA / 20일MA (골든크로스 근접도)"""
    return _ts_mean(df["close"], 5) / _ts_mean(df["close"], 20)

def ma20_to_ma60(df: pd.DataFrame) -> pd.Series:
    """20일MA / 60일MA"""
    return _ts_mean(df["close"], 20) / _ts_mean(df["close"], 60)

def high_to_close(df: pd.DataFrame) -> pd.Series:
    """당일 고가 / 종가 (상방 그림자)"""
    return df["high"] / (df["close"] + 1e-10)

def low_to_close(df: pd.DataFrame) -> pd.Series:
    """당일 저가 / 종가 (하방 그림자)"""
    return df["low"] / (df["close"] + 1e-10)

def close_to_high_20d(df: pd.DataFrame) -> pd.Series:
    """종가 / 20일 최고가"""
    return df["close"] / (_ts_max(df["high"], 20) + 1e-10)

def close_to_low_20d(df: pd.DataFrame) -> pd.Series:
    """종가 / 20일 최저가"""
    return df["close"] / (_ts_min(df["low"], 20) + 1e-10)

def ret_rank_5d(df: pd.DataFrame) -> pd.Series:
    """5일 수익률의 시계열 rank"""
    return _ts_rank(_ret(df["close"], 5), 60)

def ret_rank_20d(df: pd.DataFrame) -> pd.Series:
    """20일 수익률의 시계열 rank"""
    return _ts_rank(_ret(df["close"], 20), 120)

def price_acceleration(df: pd.DataFrame) -> pd.Series:
    """가격 가속도: 5일 수익률 변화"""
    r5 = _ret(df["close"], 5)
    return r5 - r5.shift(5)

def gap_factor(df: pd.DataFrame) -> pd.Series:
    """갭 팩터: (시가 - 전일종가) / 전일종가"""
    return (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)

def intraday_return(df: pd.DataFrame) -> pd.Series:
    """장중 수익률: (종가 - 시가) / 시가"""
    return (df["close"] - df["open"]) / (df["open"] + 1e-10)

def overnight_return(df: pd.DataFrame) -> pd.Series:
    """야간 수익률: (시가 - 전일종가) / 전일종가"""
    return (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-10)

def body_ratio(df: pd.DataFrame) -> pd.Series:
    """캔들 몸통 비율: |종가-시가| / (고가-저가)"""
    body = (df["close"] - df["open"]).abs()
    wick = df["high"] - df["low"] + 1e-10
    return body / wick

def upper_shadow(df: pd.DataFrame) -> pd.Series:
    """윗꼬리 비율"""
    return (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-10)

def lower_shadow(df: pd.DataFrame) -> pd.Series:
    """아래꼬리 비율"""
    return (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-10)


# ═══════════════════════════════════════════════
# 거래량 기반 팩터 (~20개)
# ═══════════════════════════════════════════════

def volume_ma5_ratio(df: pd.DataFrame) -> pd.Series:
    """거래량 / 5일 평균 거래량"""
    return df["volume"] / (_ts_mean(df["volume"], 5) + 1)

def volume_ma20_ratio(df: pd.DataFrame) -> pd.Series:
    """거래량 / 20일 평균 거래량"""
    return df["volume"] / (_ts_mean(df["volume"], 20) + 1)

def volume_ma60_ratio(df: pd.DataFrame) -> pd.Series:
    """거래량 / 60일 평균 거래량"""
    return df["volume"] / (_ts_mean(df["volume"], 60) + 1)

def volume_change_5d(df: pd.DataFrame) -> pd.Series:
    """5일 거래량 변화율"""
    return _ret(df["volume"], 5)

def volume_change_20d(df: pd.DataFrame) -> pd.Series:
    """20일 거래량 변화율"""
    return _ret(df["volume"], 20)

def volume_price_corr_10d(df: pd.DataFrame) -> pd.Series:
    """10일 거래량-가격 상관"""
    return _ts_corr(df["volume"].astype(float), df["close"].astype(float), 10)

def volume_price_corr_20d(df: pd.DataFrame) -> pd.Series:
    """20일 거래량-가격 상관"""
    return _ts_corr(df["volume"].astype(float), df["close"].astype(float), 20)

def volume_price_corr_60d(df: pd.DataFrame) -> pd.Series:
    """60일 거래량-가격 상관"""
    return _ts_corr(df["volume"].astype(float), df["close"].astype(float), 60)

def volume_ret_corr_20d(df: pd.DataFrame) -> pd.Series:
    """20일 거래량-수익률 상관"""
    return _ts_corr(df["volume"].astype(float), _ret(df["close"], 1), 20)

def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """VWAP 대비 괴리율 (당일)"""
    vwap = (df["close"] * df["volume"]).rolling(20).sum() / (df["volume"].rolling(20).sum() + 1)
    return df["close"] / (vwap + 1e-10) - 1

def alpha_vol_price(df: pd.DataFrame) -> pd.Series:
    """Alpha101 스타일: rank(close_change) * rank(volume)"""
    return _rank(_ret(df["close"], 1)) * _rank(df["volume"].astype(float))

def volume_concentration(df: pd.DataFrame) -> pd.Series:
    """거래량 집중도: 최근 5일 거래량 / 최근 20일 거래량"""
    return _ts_sum(df["volume"], 5) / (_ts_sum(df["volume"], 20) + 1)

def volume_std_20d(df: pd.DataFrame) -> pd.Series:
    """20일 거래량 변동성"""
    return _ts_std(df["volume"].astype(float), 20)

def turnover_rate(df: pd.DataFrame) -> pd.Series:
    """거래회전율 (20일 평균) - listed_shares 없으면 volume 자체 사용"""
    return _ts_mean(df["volume"].astype(float), 20)

def obv_slope(df: pd.DataFrame) -> pd.Series:
    """OBV 기울기 (20일)"""
    signed_vol = df["volume"] * np.where(df["close"] > df["close"].shift(1), 1, -1)
    obv = signed_vol.cumsum()
    return _ret(obv, 20)

def volume_rank_20d(df: pd.DataFrame) -> pd.Series:
    """20일 거래량 시계열 rank"""
    return _ts_rank(df["volume"].astype(float), 20)

def amihud_illiquidity(df: pd.DataFrame) -> pd.Series:
    """Amihud 비유동성: |수익률| / 거래대금"""
    traded_value = df["close"] * df["volume"] + 1
    return (_ret(df["close"], 1).abs() / traded_value).rolling(20, min_periods=1).mean()

def volume_asymmetry(df: pd.DataFrame) -> pd.Series:
    """상승일 거래량 / 하락일 거래량 (20일)"""
    ret = _ret(df["close"], 1)
    up_vol = (df["volume"] * (ret > 0)).rolling(20, min_periods=1).sum()
    dn_vol = (df["volume"] * (ret < 0)).rolling(20, min_periods=1).sum() + 1
    return up_vol / dn_vol

def price_volume_divergence(df: pd.DataFrame) -> pd.Series:
    """가격-거래량 다이버전스: 가격 상승 + 거래량 감소"""
    p_trend = _ret(df["close"], 20)
    v_trend = _ret(df["volume"], 20)
    return p_trend - v_trend


# ═══════════════════════════════════════════════
# 변동성 기반 팩터 (~15개)
# ═══════════════════════════════════════════════

def volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20일 변동성 (일간 수익률 표준편차)"""
    return _ts_std(_ret(df["close"], 1), 20)

def volatility_60d(df: pd.DataFrame) -> pd.Series:
    """60일 변동성"""
    return _ts_std(_ret(df["close"], 1), 60)

def volatility_120d(df: pd.DataFrame) -> pd.Series:
    """120일 변동성"""
    return _ts_std(_ret(df["close"], 1), 120)

def volatility_ratio(df: pd.DataFrame) -> pd.Series:
    """변동성 비율: 20일 / 60일"""
    v20 = _ts_std(_ret(df["close"], 1), 20)
    v60 = _ts_std(_ret(df["close"], 1), 60) + 1e-10
    return v20 / v60

def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range (당일)"""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def atr_ratio(df: pd.DataFrame) -> pd.Series:
    """ATR / 종가 (정규화된 변동성)"""
    tr = true_range(df)
    atr = tr.rolling(14, min_periods=1).mean()
    return atr / (df["close"] + 1e-10)

def high_low_range_20d(df: pd.DataFrame) -> pd.Series:
    """20일 고저가 범위: (max_high - min_low) / close"""
    hh = _ts_max(df["high"], 20)
    ll = _ts_min(df["low"], 20)
    return (hh - ll) / (df["close"] + 1e-10)

def downside_volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20일 하방 변동성 (음수 수익률만)"""
    ret = _ret(df["close"], 1)
    down_ret = ret.where(ret < 0, 0)
    return _ts_std(down_ret, 20)

def upside_volatility_20d(df: pd.DataFrame) -> pd.Series:
    """20일 상방 변동성"""
    ret = _ret(df["close"], 1)
    up_ret = ret.where(ret > 0, 0)
    return _ts_std(up_ret, 20)

def volatility_skew(df: pd.DataFrame) -> pd.Series:
    """변동성 비대칭: 상방 - 하방"""
    ret = _ret(df["close"], 1)
    up = _ts_std(ret.where(ret > 0, 0), 20)
    dn = _ts_std(ret.where(ret < 0, 0), 20) + 1e-10
    return up / dn

def vol_of_vol(df: pd.DataFrame) -> pd.Series:
    """변동성의 변동성"""
    vol = _ts_std(_ret(df["close"], 1), 20)
    return _ts_std(vol, 60)

def max_drawdown_20d(df: pd.DataFrame) -> pd.Series:
    """20일 최대 낙폭"""
    def _mdd(x):
        cum = (1 + x).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return dd.min()
    return _ret(df["close"], 1).rolling(20, min_periods=5).apply(_mdd, raw=False)

def realized_skewness(df: pd.DataFrame) -> pd.Series:
    """실현 왜도 (20일)"""
    return _ret(df["close"], 1).rolling(20, min_periods=10).skew()

def realized_kurtosis(df: pd.DataFrame) -> pd.Series:
    """실현 첨도 (20일)"""
    return _ret(df["close"], 1).rolling(20, min_periods=10).kurt()

def tail_risk(df: pd.DataFrame) -> pd.Series:
    """꼬리 위험: 60일 중 -3% 이하 발생 비율"""
    ret = _ret(df["close"], 1)
    return (ret < -0.03).rolling(60, min_periods=1).mean()


# ═══════════════════════════════════════════════
# 팩터 레지스트리
# ═══════════════════════════════════════════════

# 모든 팩터를 카테고리별로 등록
FACTOR_REGISTRY: dict[str, dict[str, callable]] = {
    "price_return": {
        "ret_1d": ret_1d, "ret_5d": ret_5d, "ret_10d": ret_10d,
        "ret_20d": ret_20d, "ret_60d": ret_60d, "ret_120d": ret_120d,
        "ret_250d": ret_250d, "momentum_12m_1m": momentum_12m_1m,
        "reversal_5d": reversal_5d, "reversal_20d": reversal_20d,
        "close_to_ma5": close_to_ma5, "close_to_ma20": close_to_ma20,
        "close_to_ma60": close_to_ma60, "close_to_ma120": close_to_ma120,
        "ma5_to_ma20": ma5_to_ma20, "ma20_to_ma60": ma20_to_ma60,
        "high_to_close": high_to_close, "low_to_close": low_to_close,
        "close_to_high_20d": close_to_high_20d, "close_to_low_20d": close_to_low_20d,
        "ret_rank_5d": ret_rank_5d, "ret_rank_20d": ret_rank_20d,
        "price_acceleration": price_acceleration, "gap_factor": gap_factor,
        "intraday_return": intraday_return, "overnight_return": overnight_return,
        "body_ratio": body_ratio, "upper_shadow": upper_shadow,
        "lower_shadow": lower_shadow,
    },
    "volume": {
        "volume_ma5_ratio": volume_ma5_ratio, "volume_ma20_ratio": volume_ma20_ratio,
        "volume_ma60_ratio": volume_ma60_ratio,
        "volume_change_5d": volume_change_5d, "volume_change_20d": volume_change_20d,
        "volume_price_corr_10d": volume_price_corr_10d,
        "volume_price_corr_20d": volume_price_corr_20d,
        "volume_price_corr_60d": volume_price_corr_60d,
        "volume_ret_corr_20d": volume_ret_corr_20d,
        "vwap_deviation": vwap_deviation, "alpha_vol_price": alpha_vol_price,
        "volume_concentration": volume_concentration, "volume_std_20d": volume_std_20d,
        "turnover_rate": turnover_rate, "obv_slope": obv_slope,
        "volume_rank_20d": volume_rank_20d, "amihud_illiquidity": amihud_illiquidity,
        "volume_asymmetry": volume_asymmetry,
        "price_volume_divergence": price_volume_divergence,
    },
    "volatility": {
        "volatility_20d": volatility_20d, "volatility_60d": volatility_60d,
        "volatility_120d": volatility_120d, "volatility_ratio": volatility_ratio,
        "atr_ratio": atr_ratio, "high_low_range_20d": high_low_range_20d,
        "downside_volatility_20d": downside_volatility_20d,
        "upside_volatility_20d": upside_volatility_20d,
        "volatility_skew": volatility_skew, "vol_of_vol": vol_of_vol,
        "max_drawdown_20d": max_drawdown_20d,
        "realized_skewness": realized_skewness, "realized_kurtosis": realized_kurtosis,
        "tail_risk": tail_risk,
    },
}


def get_all_factor_names() -> list[str]:
    """등록된 모든 팩터 이름을 반환합니다."""
    names = []
    for category in FACTOR_REGISTRY.values():
        names.extend(category.keys())
    return names


def compute_single_factor(
    name: str,
    df: pd.DataFrame,
) -> pd.Series:
    """단일 팩터를 계산합니다."""
    for category in FACTOR_REGISTRY.values():
        if name in category:
            return category[name](df)
    raise ValueError(f"알 수 없는 팩터: {name}")


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """모든 OHLCV 기반 팩터를 계산합니다.

    Args:
        df: 단일 종목 OHLCV DataFrame

    Returns:
        팩터 값 DataFrame (date index, factor columns)
    """
    result = {}
    for category_name, factors in FACTOR_REGISTRY.items():
        for factor_name, factor_fn in factors.items():
            try:
                result[factor_name] = factor_fn(df)
            except Exception:
                result[factor_name] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(result, index=df.index)
