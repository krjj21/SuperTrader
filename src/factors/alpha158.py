"""
Qlib Alpha158 스타일 팩터 구현
- 캔들스틱 패턴 (9개)
- 원시 가격 (4개)
- 롤링 시계열 (115개): 23 유형 x 5 윈도우(5,10,20,30,60일)
- 거래량 기반 (30개): 6 유형 x 5 윈도우

입력: OHLCV DataFrame (date, open, high, low, close, volume)
출력: pd.Series (날짜별 팩터 값)

Reference: Microsoft Qlib Alpha158
https://github.com/microsoft/qlib
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════
# 헬퍼 함수
# ═══════════════════════════════════════════════

WINDOWS = [5, 10, 20, 30, 60]


def _ts_mean(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _ts_std(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).std()


def _ts_sum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).sum()


def _ts_max(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).max()


def _ts_min(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=1).min()


def _ts_corr(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=max(w // 2, 2)).corr(y)


def _ts_argmax(s: pd.Series, w: int) -> pd.Series:
    """롤링 윈도우 내 최댓값 위치 (0~1 정규화)."""
    return s.rolling(w, min_periods=1).apply(
        lambda x: np.argmax(x) / max(len(x) - 1, 1), raw=True,
    )


def _ts_argmin(s: pd.Series, w: int) -> pd.Series:
    """롤링 윈도우 내 최솟값 위치 (0~1 정규화)."""
    return s.rolling(w, min_periods=1).apply(
        lambda x: np.argmin(x) / max(len(x) - 1, 1), raw=True,
    )


def _ts_quantile(s: pd.Series, w: int, q: float) -> pd.Series:
    return s.rolling(w, min_periods=1).quantile(q)


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


# ═══════════════════════════════════════════════
# 1. 캔들스틱 패턴 팩터 (9개)
# ═══════════════════════════════════════════════

def a158_kmid(df: pd.DataFrame) -> pd.Series:
    """K-line mid: (close - open) / open"""
    return (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

def a158_klen(df: pd.DataFrame) -> pd.Series:
    """K-line length: (high - low) / open"""
    return (df["high"] - df["low"]) / df["open"].replace(0, np.nan)

def a158_kmid2(df: pd.DataFrame) -> pd.Series:
    return a158_kmid(df) ** 2

def a158_kup(df: pd.DataFrame) -> pd.Series:
    """Upper shadow: (high - max(open,close)) / open"""
    return (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"].replace(0, np.nan)

def a158_kup2(df: pd.DataFrame) -> pd.Series:
    return a158_kup(df) ** 2

def a158_klow(df: pd.DataFrame) -> pd.Series:
    """Lower shadow: (min(open,close) - low) / open"""
    return (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"].replace(0, np.nan)

def a158_klow2(df: pd.DataFrame) -> pd.Series:
    return a158_klow(df) ** 2

def a158_ksft(df: pd.DataFrame) -> pd.Series:
    """Shift: (2*close - high - low) / open"""
    return (2 * df["close"] - df["high"] - df["low"]) / df["open"].replace(0, np.nan)

def a158_ksft2(df: pd.DataFrame) -> pd.Series:
    return a158_ksft(df) ** 2


# ═══════════════════════════════════════════════
# 2. 원시 가격 팩터 (4개)
# ═══════════════════════════════════════════════

def a158_open0(df: pd.DataFrame) -> pd.Series:
    return df["open"] / df["close"].replace(0, np.nan)

def a158_high0(df: pd.DataFrame) -> pd.Series:
    return df["high"] / df["close"].replace(0, np.nan)

def a158_low0(df: pd.DataFrame) -> pd.Series:
    return df["low"] / df["close"].replace(0, np.nan)

def a158_vwap0(df: pd.DataFrame) -> pd.Series:
    """VWAP proxy / close"""
    vwap = (df["high"] + df["low"] + df["close"]) / 3
    return vwap / df["close"].replace(0, np.nan)


# ═══════════════════════════════════════════════
# 3. 롤링 시계열 팩터 (23 유형 x 5 윈도우 = 115개)
# ═══════════════════════════════════════════════

def _make_roc(w: int):
    """Rate of change"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(w)
    fn.__name__ = f"a158_roc_{w}d"
    return fn

def _make_ma(w: int):
    """Moving average ratio"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_mean(df["close"], w) / df["close"].replace(0, np.nan)
    fn.__name__ = f"a158_ma_{w}d"
    return fn

def _make_beta(w: int):
    """Linear regression slope (trend)"""
    def fn(df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        return close.rolling(w, min_periods=max(w // 2, 3)).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 3 else np.nan,
            raw=True,
        )
    fn.__name__ = f"a158_beta_{w}d"
    return fn

def _make_rsqr(w: int):
    """R-squared from linear regression"""
    def fn(df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        def _r2(x):
            if len(x) < 3:
                return np.nan
            t = np.arange(len(x))
            p = np.polyfit(t, x, 1)
            fitted = np.polyval(p, t)
            ss_res = np.sum((x - fitted) ** 2)
            ss_tot = np.sum((x - np.mean(x)) ** 2)
            return 1 - ss_res / max(ss_tot, 1e-12)
        return close.rolling(w, min_periods=max(w // 2, 3)).apply(_r2, raw=True)
    fn.__name__ = f"a158_rsqr_{w}d"
    return fn

def _make_resi(w: int):
    """Residual from linear regression (last point)"""
    def fn(df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        def _res(x):
            if len(x) < 3:
                return np.nan
            t = np.arange(len(x))
            p = np.polyfit(t, x, 1)
            return (x[-1] - np.polyval(p, t[-1])) / max(abs(x[-1]), 1e-12)
        return close.rolling(w, min_periods=max(w // 2, 3)).apply(_res, raw=True)
    fn.__name__ = f"a158_resi_{w}d"
    return fn

def _make_std(w: int):
    """Standard deviation of returns"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _returns(df["close"]).rolling(w, min_periods=1).std()
    fn.__name__ = f"a158_std_{w}d"
    return fn

def _make_max(w: int):
    """Max high / close"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_max(df["high"], w) / df["close"].replace(0, np.nan)
    fn.__name__ = f"a158_max_{w}d"
    return fn

def _make_min(w: int):
    """Min low / close"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_min(df["low"], w) / df["close"].replace(0, np.nan)
    fn.__name__ = f"a158_min_{w}d"
    return fn

def _make_qtlu(w: int):
    """80th percentile of close / close"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_quantile(df["close"], w, 0.8) / df["close"].replace(0, np.nan)
    fn.__name__ = f"a158_qtlu_{w}d"
    return fn

def _make_qtld(w: int):
    """20th percentile of close / close"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_quantile(df["close"], w, 0.2) / df["close"].replace(0, np.nan)
    fn.__name__ = f"a158_qtld_{w}d"
    return fn

def _make_rsv(w: int):
    """Stochastic RSV: (close - min) / (max - min)"""
    def fn(df: pd.DataFrame) -> pd.Series:
        hi = _ts_max(df["close"], w)
        lo = _ts_min(df["close"], w)
        denom = (hi - lo).replace(0, np.nan)
        return (df["close"] - lo) / denom
    fn.__name__ = f"a158_rsv_{w}d"
    return fn

def _make_imax(w: int):
    """Argmax position in window (0~1)"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_argmax(df["high"], w)
    fn.__name__ = f"a158_imax_{w}d"
    return fn

def _make_imin(w: int):
    """Argmin position in window (0~1)"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_argmin(df["low"], w)
    fn.__name__ = f"a158_imin_{w}d"
    return fn

def _make_imxd(w: int):
    """Distance between argmax and argmin"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_argmax(df["high"], w) - _ts_argmin(df["low"], w)
    fn.__name__ = f"a158_imxd_{w}d"
    return fn

def _make_corr(w: int):
    """Price-volume correlation"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_corr(df["close"], df["volume"].astype(float), w)
    fn.__name__ = f"a158_corr_{w}d"
    return fn

def _make_cord(w: int):
    """Return-volume correlation"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        return _ts_corr(ret, df["volume"].astype(float), w)
    fn.__name__ = f"a158_cord_{w}d"
    return fn

def _make_cntp(w: int):
    """Count of positive return days / window"""
    def fn(df: pd.DataFrame) -> pd.Series:
        pos = (_returns(df["close"]) > 0).astype(float)
        return _ts_mean(pos, w)
    fn.__name__ = f"a158_cntp_{w}d"
    return fn

def _make_cntn(w: int):
    """Count of negative return days / window"""
    def fn(df: pd.DataFrame) -> pd.Series:
        neg = (_returns(df["close"]) < 0).astype(float)
        return _ts_mean(neg, w)
    fn.__name__ = f"a158_cntn_{w}d"
    return fn

def _make_cntd(w: int):
    """Net direction: cntp - cntn"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        pos = (ret > 0).astype(float)
        neg = (ret < 0).astype(float)
        return _ts_mean(pos, w) - _ts_mean(neg, w)
    fn.__name__ = f"a158_cntd_{w}d"
    return fn

def _make_sump(w: int):
    """Sum of positive returns"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        return _ts_sum(ret.clip(lower=0), w)
    fn.__name__ = f"a158_sump_{w}d"
    return fn

def _make_sumn(w: int):
    """Sum of absolute negative returns"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        return _ts_sum((-ret).clip(lower=0), w)
    fn.__name__ = f"a158_sumn_{w}d"
    return fn

def _make_sumd(w: int):
    """Net return: sump - sumn"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        return _ts_sum(ret.clip(lower=0), w) - _ts_sum((-ret).clip(lower=0), w)
    fn.__name__ = f"a158_sumd_{w}d"
    return fn

def _make_vstd(w: int):
    """Volume standard deviation"""
    def fn(df: pd.DataFrame) -> pd.Series:
        return _ts_std(df["volume"].astype(float), w)
    fn.__name__ = f"a158_vstd_{w}d"
    return fn


# ═══════════════════════════════════════════════
# 4. 거래량 기반 팩터 (6 유형 x 5 윈도우 = 30개)
# ═══════════════════════════════════════════════

def _make_vma(w: int):
    """Volume moving average ratio"""
    def fn(df: pd.DataFrame) -> pd.Series:
        vol = df["volume"].astype(float)
        return _ts_mean(vol, w) / vol.replace(0, np.nan)
    fn.__name__ = f"a158_vma_{w}d"
    return fn

def _make_vol_vstd(w: int):
    """Volume std / volume mean"""
    def fn(df: pd.DataFrame) -> pd.Series:
        vol = df["volume"].astype(float)
        return _ts_std(vol, w) / _ts_mean(vol, w).replace(0, np.nan)
    fn.__name__ = f"a158_vol_vstd_{w}d"
    return fn

def _make_wvma(w: int):
    """Weighted volume MA: volume * |return| rolling mean"""
    def fn(df: pd.DataFrame) -> pd.Series:
        weighted = df["volume"].astype(float) * _returns(df["close"]).abs()
        return _ts_mean(weighted, w)
    fn.__name__ = f"a158_wvma_{w}d"
    return fn

def _make_vsump(w: int):
    """Volume on up days"""
    def fn(df: pd.DataFrame) -> pd.Series:
        up = (_returns(df["close"]) > 0).astype(float)
        return _ts_sum(df["volume"].astype(float) * up, w)
    fn.__name__ = f"a158_vsump_{w}d"
    return fn

def _make_vsumn(w: int):
    """Volume on down days"""
    def fn(df: pd.DataFrame) -> pd.Series:
        down = (_returns(df["close"]) < 0).astype(float)
        return _ts_sum(df["volume"].astype(float) * down, w)
    fn.__name__ = f"a158_vsumn_{w}d"
    return fn

def _make_vsumd(w: int):
    """Net volume direction: vsump - vsumn"""
    def fn(df: pd.DataFrame) -> pd.Series:
        ret = _returns(df["close"])
        vol = df["volume"].astype(float)
        up_vol = _ts_sum(vol * (ret > 0).astype(float), w)
        dn_vol = _ts_sum(vol * (ret < 0).astype(float), w)
        return up_vol - dn_vol
    fn.__name__ = f"a158_vsumd_{w}d"
    return fn


# ═══════════════════════════════════════════════
# 팩터 레지스트리 자동 생성
# ═══════════════════════════════════════════════

# 캔들스틱 + 원시 가격 (고정)
_STATIC_FACTORS = {
    "a158_kmid": a158_kmid,
    "a158_klen": a158_klen,
    "a158_kmid2": a158_kmid2,
    "a158_kup": a158_kup,
    "a158_kup2": a158_kup2,
    "a158_klow": a158_klow,
    "a158_klow2": a158_klow2,
    "a158_ksft": a158_ksft,
    "a158_ksft2": a158_ksft2,
    "a158_open0": a158_open0,
    "a158_high0": a158_high0,
    "a158_low0": a158_low0,
    "a158_vwap0": a158_vwap0,
}

# 롤링 팩터 팩토리 (23 유형)
_ROLLING_FACTORIES = [
    _make_roc, _make_ma, _make_beta, _make_rsqr, _make_resi,
    _make_std, _make_max, _make_min, _make_qtlu, _make_qtld, _make_rsv,
    _make_imax, _make_imin, _make_imxd,
    _make_corr, _make_cord, _make_cntp, _make_cntn, _make_cntd,
    _make_sump, _make_sumn, _make_sumd, _make_vstd,
]

# 거래량 팩터 팩토리 (6 유형)
_VOLUME_FACTORIES = [
    _make_vma, _make_vol_vstd, _make_wvma,
    _make_vsump, _make_vsumn, _make_vsumd,
]

# 전체 팩터 딕셔너리 생성
_ALL_FACTORS: dict[str, callable] = {}
_ALL_FACTORS.update(_STATIC_FACTORS)

for factory in _ROLLING_FACTORIES:
    for w in WINDOWS:
        fn = factory(w)
        _ALL_FACTORS[fn.__name__] = fn

for factory in _VOLUME_FACTORIES:
    for w in WINDOWS:
        fn = factory(w)
        _ALL_FACTORS[fn.__name__] = fn

# 레지스트리 (alpha101 호환 구조)
FACTOR_REGISTRY: dict[str, dict[str, callable]] = {
    "candlestick": {k: v for k, v in _ALL_FACTORS.items() if k in _STATIC_FACTORS and "open0" not in k and "high0" not in k and "low0" not in k and "vwap0" not in k},
    "raw_price": {k: v for k, v in _ALL_FACTORS.items() if k.endswith("0") and k.startswith("a158_")},
    "rolling_trend": {k: v for k, v in _ALL_FACTORS.items() if any(x in k for x in ["_roc_", "_ma_", "_beta_", "_rsqr_", "_resi_"])},
    "rolling_volatility": {k: v for k, v in _ALL_FACTORS.items() if any(x in k for x in ["_std_", "_max_", "_min_", "_qtlu_", "_qtld_", "_rsv_"])},
    "rolling_cycle": {k: v for k, v in _ALL_FACTORS.items() if any(x in k for x in ["_imax_", "_imin_", "_imxd_"])},
    "rolling_pv_corr": {k: v for k, v in _ALL_FACTORS.items() if any(x in k for x in ["_corr_", "_cord_", "_cntp_", "_cntn_", "_cntd_", "_sump_", "_sumn_", "_sumd_", "_vstd_"])},
    "volume_rolling": {k: v for k, v in _ALL_FACTORS.items() if any(x in k for x in ["_vma_", "_vol_vstd_", "_wvma_", "_vsump_", "_vsumn_", "_vsumd_"])},
}


# ═══════════════════════════════════════════════
# 공개 API (alpha101 호환)
# ═══════════════════════════════════════════════

def get_all_factor_names() -> list[str]:
    """등록된 모든 Alpha158 팩터 이름을 반환합니다."""
    return list(_ALL_FACTORS.keys())


def compute_single_factor(name: str, df: pd.DataFrame) -> pd.Series:
    """단일 팩터를 계산합니다."""
    if name not in _ALL_FACTORS:
        raise ValueError(f"Unknown alpha158 factor: {name}")
    result = _ALL_FACTORS[name](df)
    return result.replace([np.inf, -np.inf], np.nan)


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """모든 Alpha158 팩터를 계산합니다.

    Args:
        df: OHLCV DataFrame (columns: open, high, low, close, volume)

    Returns:
        DataFrame with factor columns
    """
    results = {}
    for name, fn in _ALL_FACTORS.items():
        try:
            values = fn(df)
            results[name] = values.replace([np.inf, -np.inf], np.nan)
        except Exception:
            results[name] = pd.Series(np.nan, index=df.index)

    return pd.DataFrame(results, index=df.index)
