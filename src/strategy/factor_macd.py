"""
팩터 + MACD 타이밍 전략
- 종목풀 내 종목에 대해 MACD 골든/데드크로스 시그널
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.data.indicators import add_all_indicators


class FactorMACDStrategy(BaseStrategy):

    def __init__(self, params: dict | None = None):
        super().__init__(name="factor_macd", params=params)
        self._pool: set[str] = set()

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        # 종목풀 퇴출 → SELL
        if stock_code not in self._pool:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6, reason="종목풀 퇴출",
            )

        if len(df) < 30:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        if "macd_hist" not in df.columns:
            df = add_all_indicators(df, self.params)

        hist = df["macd_hist"]
        if len(hist) < 2 or pd.isna(hist.iloc[-1]) or pd.isna(hist.iloc[-2]):
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # MACD 골든크로스 (히스토그램 음→양)
        if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.7,
                reason=f"MACD 골든크로스 (hist={hist.iloc[-1]:.4f})",
                metadata={"macd_hist": float(hist.iloc[-1])},
            )

        # MACD 데드크로스 (히스토그램 양→음)
        if hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6,
                reason=f"MACD 데드크로스 (hist={hist.iloc[-1]:.4f})",
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
