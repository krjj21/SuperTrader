"""
팩터 + PPO RL 타이밍 전략
- 종목풀 내 종목에 대해 RL 에이전트 예측 기반 시그널
- 포지션 상태를 추적하여 RL 에이전트에 전달
- 백테스트 엔진과 포지션 동기화 지원
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal


class FactorRLStrategy(BaseStrategy):
    """PPO RL 타이밍 기반 전략 (포지션 인식)

    RL 모델은 일봉 기준으로 학습되었으므로, 라이브에서 holding_days를
    사이클 단위가 아닌 영업일 단위로 계산하여 모델에 전달합니다.
    """

    def __init__(self, model_path: str, params: dict | None = None):
        super().__init__(name="factor_rl", params=params)
        from src.timing.predictor import TimingPredictor
        from src.config import get_config
        self.predictor = TimingPredictor("rl", model_path)
        rl_cfg = get_config().timing.rl
        self._buy_threshold = rl_cfg.buy_action_threshold
        self._sell_threshold = rl_cfg.sell_action_threshold
        self._pool: set[str] = set()
        # 포지션 추적: {종목코드: {"entry_price": float, "entry_date": str}}
        self._positions: dict[str, dict] = {}

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    @staticmethod
    def _business_days_held(entry_date: str) -> int:
        """매수일로부터 영업일 기준 보유 일수를 계산합니다."""
        from datetime import datetime
        try:
            entry = datetime.strptime(entry_date, "%Y%m%d")
            today = datetime.now()
            days = 0
            current = entry
            from datetime import timedelta
            while current < today:
                current += timedelta(days=1)
                if current.weekday() < 5:  # 월~금
                    days += 1
            return days
        except Exception:
            return 0

    def sync_positions(self, held_codes: set[str], prices: dict[str, float] | None = None) -> None:
        """백테스트 엔진의 실제 보유 종목과 동기화합니다."""
        from datetime import datetime
        # 엔진에서 매도된 종목 제거
        for code in list(self._positions):
            if code not in held_codes:
                del self._positions[code]
        # 엔진에서 보유 중이지만 전략이 모르는 종목 추가
        for code in held_codes:
            if code not in self._positions:
                price = prices.get(code, 0) if prices else 0
                self._positions[code] = {
                    "entry_price": float(price) if price > 0 else 1.0,
                    "entry_date": datetime.now().strftime("%Y%m%d"),
                }

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

        # 포지션 상태 조회 (영업일 기준 보유 일수)
        pos = self._positions.get(stock_code)
        holding = pos is not None
        unrealized_pnl = 0.0
        holding_days = 0

        if holding:
            unrealized_pnl = (price / pos["entry_price"] - 1.0) if pos["entry_price"] > 0 else 0.0
            holding_days = self._business_days_held(pos["entry_date"])

        try:
            prediction = self.predictor.predict_with_position(
                df, holding, unrealized_pnl, holding_days,
                buy_threshold=self._buy_threshold,
                sell_threshold=self._sell_threshold,
            )
        except Exception as e:
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=f"예측 실패: {e}",
            )

        if prediction == 1 and not holding:
            from datetime import datetime
            self._positions[stock_code] = {
                "entry_price": float(price),
                "entry_date": datetime.now().strftime("%Y%m%d"),
            }
            return TradeSignal(
                signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.75,
                reason="PPO RL BUY",
            )
        elif prediction == -1 and holding:
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=0.65,
                reason=f"PPO RL SELL (보유 {holding_days}일)",
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
