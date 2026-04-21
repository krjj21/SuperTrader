"""
팩터 + PPO RL 타이밍 전략
- 종목풀 내 종목에 대해 RL 에이전트 예측 기반 시그널
- 포지션 상태를 추적하여 RL 에이전트에 전달
- 백테스트 엔진과 포지션 동기화 지원
"""
from __future__ import annotations

import pandas as pd

from src.strategy.base import BaseStrategy, TradeSignal, Signal
from src.strategy._position_utils import business_days_held, resolve_buy_date


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
        # 백테스트용 기준일 (라이브에서는 None → datetime.now() 사용)
        self._current_date: str | None = None

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    # 공통 헬퍼 (_position_utils.py) 의 래퍼 — 기존 호출 코드 호환성 유지
    _business_days_held = staticmethod(business_days_held)
    _resolve_buy_date = staticmethod(resolve_buy_date)

    def sync_positions(self, held_codes: set[str], prices: dict[str, float] | None = None,
                       avg_prices: dict[str, float] | None = None,
                       entry_dates: dict[str, str] | None = None,
                       current_date: str | None = None) -> None:
        """보유 종목과 동기화합니다.

        Args:
            held_codes: 보유 종목 코드 집합
            prices: {종목코드: 현재가}
            avg_prices: {종목코드: 매수평균가} — 매수일 조회/추정용
            entry_dates: {종목코드: 매수일(YYYYMMDD)} — 백테스트 엔진이 직접 전달
            current_date: 백테스트 기준일 (YYYYMMDD or YYYY-MM-DD)
        """
        if current_date:
            self._current_date = current_date.replace("-", "")
        # 매도된 종목 제거
        for code in list(self._positions):
            if code not in held_codes:
                del self._positions[code]
        # 보유 중이지만 전략이 모르는 종목 추가
        for code in held_codes:
            if code not in self._positions:
                cur_price = prices.get(code, 0) if prices else 0
                avg_price = avg_prices.get(code, 0) if avg_prices else 0
                entry_price = float(avg_price) if avg_price > 0 else (float(cur_price) if cur_price > 0 else 1.0)
                # 우선순위: 엔진 제공 entry_date > DB/일봉 추정 > 오늘
                if entry_dates and code in entry_dates and entry_dates[code]:
                    entry_date = entry_dates[code].replace("-", "")
                else:
                    entry_date = self._resolve_buy_date(code, entry_price)
                self._positions[code] = {
                    "entry_price": entry_price,
                    "entry_date": entry_date,
                }

    def generate_signal(
        self, stock_code: str, df: pd.DataFrame, stock_name: str = "",
        current_date: str | None = None,
    ) -> TradeSignal:
        price = int(df["close"].iloc[-1]) if len(df) > 0 else 0

        if current_date:
            self._current_date = current_date.replace("-", "")

        if len(df) < 60:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # 포지션 상태 조회 (영업일 기준 보유 일수)
        pos = self._positions.get(stock_code)
        holding = pos is not None
        unrealized_pnl = 0.0
        holding_days = 0

        if holding:
            unrealized_pnl = (price / pos["entry_price"] - 1.0) if pos["entry_price"] > 0 else 0.0
            holding_days = self._business_days_held(pos["entry_date"], self._current_date)

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
            entry_date = self._current_date or datetime.now().strftime("%Y%m%d")
            self._positions[stock_code] = {
                "entry_price": float(price),
                "entry_date": entry_date,
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
