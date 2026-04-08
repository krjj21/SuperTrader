"""
팩터 + GRPO RL 타이밍 전략
- 종목풀 내 종목에 대해 RL 에이전트 예측 기반 시그널
- 포지션 상태를 추적하여 RL 에이전트에 전달
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal


class FactorRLStrategy(BaseStrategy):
    """GRPO RL 타이밍 기반 전략 (포지션 인식)"""

    def __init__(self, model_path: str, params: dict | None = None):
        super().__init__(name="factor_rl", params=params)
        from src.timing.predictor import TimingPredictor
        self.predictor = TimingPredictor("rl", model_path)
        self._pool: set[str] = set()
        # 포지션 추적: {종목코드: {"entry_price": float, "holding_days": int}}
        self._positions: dict[str, dict] = {}

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        if stock_code not in self._pool:
            if stock_code in self._positions:
                del self._positions[stock_code]
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6, reason="종목풀 퇴출",
            )

        if len(df) < 60:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # 포지션 상태 조회
        pos = self._positions.get(stock_code)
        holding = pos is not None
        unrealized_pnl = 0.0
        holding_days = 0

        if holding:
            unrealized_pnl = (price / pos["entry_price"] - 1.0) if pos["entry_price"] > 0 else 0.0
            holding_days = pos["holding_days"]
            pos["holding_days"] += 1  # 매일 증가

        try:
            prediction = self.predictor.predict_with_position(
                df, holding, unrealized_pnl, holding_days,
            )
        except Exception as e:
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=f"예측 실패: {e}",
            )

        if prediction == 1:
            # BUY 시그널 → 포지션 기록
            if not holding:
                self._positions[stock_code] = {
                    "entry_price": float(price),
                    "holding_days": 0,
                }
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.75,
                reason="GRPO RL BUY 예측",
            )
        elif prediction == -1:
            # SELL 시그널 → 포지션 제거
            if holding:
                del self._positions[stock_code]
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.65,
                reason="GRPO RL SELL 예측",
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
