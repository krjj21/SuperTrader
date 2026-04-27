"""시장 국면(regime) 감지 모듈.

- detector.RegimeDetector: KOSPI 60일 (return, vol) → GMM 3-state → 사후 라벨링 → LLM 결합
- weights: 라벨별 팩터 카테고리 가중치 / 포지션 사이즈 multiplier

Phase 1+2 (현 라운드): 라벨 산출 + 팩터/포지션 모듈레이션. RL state는 미통합.
"""
from src.regime.detector import RegimeDetector, RegimeResult
from src.regime.weights import (
    get_category_weights,
    get_position_multiplier,
    get_factor_category,
)

__all__ = [
    "RegimeDetector", "RegimeResult",
    "get_category_weights", "get_position_multiplier", "get_factor_category",
]
