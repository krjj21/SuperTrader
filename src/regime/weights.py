"""Regime 라벨 → 팩터 카테고리 가중치 / 포지션 사이즈 multiplier.

라벨 정의:
  risk_on_trend      : 상승 추세 / 낮은 변동성 — 모멘텀에 비중
  high_vol_risk_off  : 높은 변동성 / 약세 — 저변동성·퀄리티에 비중, 사이즈 축소
  mean_revert        : 횡보 / 단기 리버설 — 단기 반전 팩터에 비중
"""
from __future__ import annotations

from src.factors import alpha101, alpha158


# ──────────────────────────────────────────────
# 카테고리 가중치 multiplier
# (1.0 = 변경 없음, IC-가중 후 카테고리별로 곱한 뒤 재정규화)
# ──────────────────────────────────────────────
_CATEGORY_WEIGHTS: dict[str, dict[str, float]] = {
    "risk_on_trend": {
        # alpha101
        "price_return": 1.30,
        "volume": 1.00,
        "volatility": 0.70,
        # alpha158
        "rolling_trend": 1.30,
        "rolling_volatility": 0.70,
        "rolling_pv_corr": 1.10,
        "rolling_cycle": 1.00,
        "candlestick": 1.00,
        "raw_price": 1.00,
        "volume_rolling": 1.00,
    },
    "high_vol_risk_off": {
        "price_return": 0.60,
        "volume": 0.90,
        "volatility": 1.30,
        "rolling_trend": 0.60,
        "rolling_volatility": 1.30,
        "rolling_pv_corr": 0.90,
        "rolling_cycle": 1.00,
        "candlestick": 1.00,
        "raw_price": 1.00,
        "volume_rolling": 1.00,
    },
    "mean_revert": {
        "price_return": 0.80,
        "volume": 1.00,
        "volatility": 1.00,
        "rolling_trend": 0.80,
        "rolling_volatility": 1.00,
        "rolling_pv_corr": 1.00,
        "rolling_cycle": 1.30,    # cycle/argmax-min은 단기 반전에 민감
        "candlestick": 1.20,      # intraday 패턴
        "raw_price": 1.00,
        "volume_rolling": 1.00,
    },
}

# 라벨별 포지션 사이즈 multiplier (calculate_position_size에 곱)
_POSITION_MULTIPLIER: dict[str, float] = {
    "risk_on_trend": 1.10,
    "high_vol_risk_off": 0.60,
    "mean_revert": 0.90,
}


def get_category_weights(label: str | None) -> dict[str, float]:
    """라벨에 해당하는 카테고리 multiplier dict. 라벨이 없으면 빈 dict (=영향 없음)."""
    if not label:
        return {}
    return dict(_CATEGORY_WEIGHTS.get(label, {}))


def get_position_multiplier(label: str | None) -> float:
    """라벨에 해당하는 포지션 사이즈 multiplier. 라벨이 없으면 1.0."""
    if not label:
        return 1.0
    return _POSITION_MULTIPLIER.get(label, 1.0)


# ──────────────────────────────────────────────
# 팩터 이름 → 카테고리 매핑
# (alpha101 + alpha158 FACTOR_REGISTRY 누적)
# ──────────────────────────────────────────────
_FACTOR_CATEGORY_CACHE: dict[str, str] | None = None


def _build_factor_category_map() -> dict[str, str]:
    cache: dict[str, str] = {}
    for category, factors in alpha101.FACTOR_REGISTRY.items():
        for name in factors:
            cache[name] = category
    for category, factors in alpha158.FACTOR_REGISTRY.items():
        for name in factors:
            cache[name] = category
    return cache


def get_factor_category(factor_name: str) -> str | None:
    """팩터 이름의 카테고리. 미등록이면 None (=multiplier 적용 안 함)."""
    global _FACTOR_CATEGORY_CACHE
    if _FACTOR_CATEGORY_CACHE is None:
        _FACTOR_CATEGORY_CACHE = _build_factor_category_map()
    return _FACTOR_CATEGORY_CACHE.get(factor_name)
