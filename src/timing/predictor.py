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

    def predict(self, df: pd.DataFrame) -> int:
        """단일 종목의 최신 타이밍 시그널을 예측합니다.

        Args:
            df: OHLCV DataFrame (최소 60일 이상)

        Returns:
            1 (BUY), 0 (HOLD), -1 (SELL)
        """
        features = build_features(df)
        predictions = self.model.predict(features)
        return int(predictions.iloc[-1])

    def predict_with_position(
        self,
        df: pd.DataFrame,
        holding: bool = False,
        unrealized_pnl: float = 0.0,
        holding_days: int = 0,
    ) -> int:
        """포지션 상태를 반영한 예측 (RL 모델용).

        비-RL 모델은 포지션 무시하고 일반 predict() 사용.
        """
        if hasattr(self.model, "predict_with_position"):
            features = build_features(df)
            return self.model.predict_with_position(
                features, holding, unrealized_pnl, holding_days,
            )
        return self.predict(df)

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
