"""
통합 예측 인터페이스
- 모든 타이밍 모델의 예측을 통일된 인터페이스로 제공
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.timing.features import build_features


class TimingPredictor:
    """타이밍 예측 통합 인터페이스"""

    def __init__(self, model_type: str, model_path: str):
        self.model_type = model_type
        from src.timing.trainer import create_model
        self.model = create_model(model_type)
        self.model.load(model_path)
        logger.info(f"타이밍 모델 로드: {model_type} ({model_path})")

    def _resolve_features(
        self,
        df: pd.DataFrame | None,
        features: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """features 가 주어지면 그대로, 아니면 df 에서 build_features.

        백테스트 precompute 경로(features 전달) ↔ 라이브 경로(df 전달) 통합.
        """
        if features is not None:
            return features
        if df is None:
            raise ValueError("predictor: df 또는 features 둘 중 하나는 필요합니다")
        return build_features(df)

    def predict(
        self,
        df: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
    ) -> int:
        """단일 종목의 최신 타이밍 시그널을 예측합니다.

        Args:
            df: OHLCV DataFrame (최소 60일 이상). features 미전달 시 사용.
            features: 사전 계산된 build_features 결과. 백테스트 precompute 경로.

        Returns:
            1 (BUY), 0 (HOLD), -1 (SELL)
        """
        feats = self._resolve_features(df, features)
        seq_len = getattr(self.model, "sequence_length", None)
        if seq_len and len(feats) > seq_len:
            feats = feats.iloc[-seq_len:]
        predictions = self.model.predict(feats)
        return int(predictions.iloc[-1])

    def predict_with_position(
        self,
        df: pd.DataFrame | None = None,
        holding: bool = False,
        unrealized_pnl: float = 0.0,
        holding_days: int = 0,
        buy_threshold: float = 0.08,
        sell_threshold: float = 0.05,
        features: pd.DataFrame | None = None,
    ) -> int:
        """포지션 상태를 반영한 예측 (RL 모델용)."""
        if hasattr(self.model, "predict_with_position"):
            feats = self._resolve_features(df, features)
            return self.model.predict_with_position(
                feats, holding, unrealized_pnl, holding_days,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
        return self.predict(df=df, features=features)

    def predict_with_position_with_probs(
        self,
        df: pd.DataFrame | None = None,
        holding: bool = False,
        unrealized_pnl: float = 0.0,
        holding_days: int = 0,
        buy_threshold: float = 0.08,
        sell_threshold: float = 0.05,
        features: pd.DataFrame | None = None,
    ):
        """predict_with_position 과 동일 + raw probs ([HOLD,BUY,SELL]) 함께 반환."""
        if hasattr(self.model, "predict_with_position_with_probs"):
            feats = self._resolve_features(df, features)
            return self.model.predict_with_position_with_probs(
                feats, holding, unrealized_pnl, holding_days,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
        return (self.predict(df=df, features=features), None)

    def predict_proba_last(
        self,
        df: pd.DataFrame | None = None,
        label: int = -1,
        features: pd.DataFrame | None = None,
    ) -> float | None:
        """최근 bar 의 특정 라벨 예측 확률을 반환합니다."""
        if not hasattr(self.model, "predict_proba"):
            return None
        try:
            feats = self._resolve_features(df, features)
            seq_len = getattr(self.model, "sequence_length", None)
            if seq_len and len(feats) > seq_len:
                feats = feats.iloc[-seq_len:]
            proba = self.model.predict_proba(feats)
            if label not in proba.columns:
                return None
            val = float(proba[label].iloc[-1])
            if val != val:  # NaN 방어
                return None
            return val
        except Exception as e:
            logger.debug(f"predict_proba_last 실패: {e}")
            return None

    def predict_batch(self, ohlcv_dict: dict[str, pd.DataFrame]) -> dict[str, int]:
        """여러 종목의 타이밍 시그널을 배치 예측합니다.

        Returns:
            {종목코드: 시그널} 딕셔너리
        """
        results = {}
        for code, df in ohlcv_dict.items():
            try:
                results[code] = self.predict(df)
            except Exception as e:
                logger.debug(f"예측 실패: {code} - {e}")
                results[code] = 0
        return results
