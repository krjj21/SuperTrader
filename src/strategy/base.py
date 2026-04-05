"""
전략 추상 클래스
- 모든 전략이 구현해야 할 인터페이스 정의
- 시그널 타입 정의
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    """매매 시그널"""
    signal: Signal
    stock_code: str
    stock_name: str = ""
    price: int = 0
    strength: float = 0.0       # 시그널 강도 (0.0 ~ 1.0)
    reason: str = ""
    stop_loss: int = 0          # 손절가
    take_profit: int = 0        # 익절가
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_actionable(self) -> bool:
        return self.signal in (Signal.BUY, Signal.SELL)


class BaseStrategy(ABC):
    """전략 추상 클래스"""

    def __init__(self, name: str, params: dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signal(
        self,
        stock_code: str,
        df: pd.DataFrame,
        stock_name: str = "",
    ) -> TradeSignal:
        """OHLCV + 지표 DataFrame을 분석하여 매매 시그널을 생성합니다.

        Args:
            stock_code: 종목코드
            df: 지표가 포함된 OHLCV DataFrame
            stock_name: 종목명

        Returns:
            TradeSignal 객체
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
