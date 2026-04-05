"""
순수 팩터 종목 선정 전략 (베이스라인)
- 리밸런싱일: 종목풀 변경 시그널
- 그 외: HOLD
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal


class FactorOnlyStrategy(BaseStrategy):

    def __init__(self, params: dict | None = None):
        super().__init__(name="factor_only", params=params)
        self._current_pool: set[str] = set()
        self._new_pool: set[str] = set()

    def update_pool(self, new_codes: list[str]) -> None:
        self._new_pool = set(new_codes)

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        # 신규 편입 → BUY
        if stock_code in self._new_pool and stock_code not in self._current_pool:
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.7, reason="팩터 종목풀 신규 편입",
            )

        # 퇴출 → SELL
        if stock_code in self._current_pool and stock_code not in self._new_pool:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.7, reason="팩터 종목풀 퇴출",
            )

        return TradeSignal(
            signal=Signal.HOLD, stock_code=stock_code, stock_name=stock_name, price=price,
        )

    def commit_pool(self) -> None:
        self._current_pool = self._new_pool.copy()
