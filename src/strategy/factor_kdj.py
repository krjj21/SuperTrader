"""
팩터 + KDJ 타이밍 전략
- 종목풀 내 종목에 대해 KDJ 과매수/과매도 + 크로스 시그널
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.data.indicators import add_all_indicators


class FactorKDJStrategy(BaseStrategy):

    def __init__(self, params: dict | None = None):
        super().__init__(name="factor_kdj", params=params)
        self._pool: set[str] = set()

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        if len(df) < 30:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        if "kdj_k" not in df.columns:
            df = add_all_indicators(df, self.params)

        k = df["kdj_k"]
        d = df["kdj_d"]
        j = df["kdj_j"]

        if len(k) < 2 or pd.isna(k.iloc[-1]):
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # BUY: J < 20 (과매도) + K가 D를 상향 돌파
        if (j.iloc[-1] < 20 and k.iloc[-1] > d.iloc[-1] and k.iloc[-2] <= d.iloc[-2]):
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.7,
                reason=f"KDJ 과매도 골든크로스 (J={j.iloc[-1]:.1f}, K={k.iloc[-1]:.1f})",
            )

        # SELL: J > 80 (과매수) + K가 D를 하향 돌파
        if (j.iloc[-1] > 80 and k.iloc[-1] < d.iloc[-1] and k.iloc[-2] >= d.iloc[-2]):
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.6,
                reason=f"KDJ 과매수 데드크로스 (J={j.iloc[-1]:.1f})",
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
