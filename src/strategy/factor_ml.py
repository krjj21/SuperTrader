"""
팩터 + ML 타이밍 전략 (DT, XGBoost, LightGBM, LSTM, Transformer 통합)
- 종목풀 내 종목에 대해 ML 모델 예측 기반 시그널
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal


class FactorMLStrategy(BaseStrategy):
    """ML 타이밍 기반 전략 (모든 ML 모델 통합)"""

    def __init__(
        self,
        model_type: str,
        model_path: str,
        params: dict | None = None,
    ):
        super().__init__(name=f"factor_{model_type}", params=params)
        self.model_type = model_type
        from src.timing.predictor import TimingPredictor
        self.predictor = TimingPredictor(model_type, model_path)
        self._pool: set[str] = set()

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        if stock_code not in self._pool:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6, reason="종목풀 퇴출",
            )

        if len(df) < 60:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        try:
            prediction = self.predictor.predict(df)
        except Exception as e:
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=f"예측 실패: {e}",
            )

        if prediction == 1:
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.7,
                reason=f"{self.model_type} BUY 예측",
            )
        elif prediction == -1:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6,
                reason=f"{self.model_type} SELL 예측",
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
