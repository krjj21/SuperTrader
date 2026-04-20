"""
Hybrid 전략: XGBoost(알파 생성) + RL(리스크/타이밍 필터)

구조:
  [Alpha Layer] XGBoost → BUY/SELL 시그널 생성 (raw alpha)
  [RL Layer]    PPO RL  → 실행 여부 판단 (entry timing + risk filter)
  [Output]      두 모델이 동의할 때만 실행

XGBoost: 순수 예측력 (방향성)
RL:      포지션 인식 + 리스크 관리 (언제/얼마나)
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

from src.strategy.base import BaseStrategy, TradeSignal, Signal


class FactorHybridStrategy(BaseStrategy):
    """XGBoost(알파) + RL(리스크 필터) Hybrid 전략"""

    def __init__(self, ml_model_path: str, rl_model_path: str, params: dict | None = None):
        super().__init__(name="factor_hybrid", params=params)
        from src.timing.predictor import TimingPredictor
        from src.config import get_config

        # Alpha Layer: XGBoost
        self.ml_predictor = TimingPredictor("xgboost", ml_model_path)

        # RL Layer: PPO (리스크 필터)
        self.rl_predictor = TimingPredictor("rl", rl_model_path)
        rl_cfg = get_config().timing.rl
        self._buy_threshold = rl_cfg.buy_action_threshold
        self._sell_threshold = rl_cfg.sell_action_threshold

        self._pool: set[str] = set()
        # 포지션 추적 (RL 레이어용)
        self._positions: dict[str, dict] = {}
        # 백테스트용 기준일 (라이브에서는 None)
        self._current_date: str | None = None

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    @staticmethod
    def _business_days_held(entry_date: str, reference_date: str | None = None) -> int:
        from datetime import datetime, timedelta
        try:
            entry = datetime.strptime(entry_date, "%Y%m%d")
            if reference_date:
                today = datetime.strptime(reference_date.replace("-", ""), "%Y%m%d")
            else:
                today = datetime.now()
            days = 0
            current = entry
            while current < today:
                current += timedelta(days=1)
                if current.weekday() < 5:
                    days += 1
            return days
        except Exception:
            return 0

    @staticmethod
    def _resolve_buy_date(code: str, avg_price: float) -> str:
        from datetime import datetime, timedelta
        try:
            from src.db.models import get_holding_buy_date
            db_date = get_holding_buy_date(code)
            if db_date:
                return db_date
        except Exception:
            pass
        try:
            from src.data.market_data import get_ohlcv
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")
            df = get_ohlcv(code, start, end)
            if df is not None and not df.empty and avg_price > 0:
                df["diff"] = (df["close"] - avg_price).abs()
                best_idx = df["diff"].idxmin()
                buy_date = pd.Timestamp(df.loc[best_idx, "date"]).strftime("%Y%m%d")
                try:
                    from src.db.models import save_holding
                    save_holding(code, "", int(avg_price), 0, buy_date)
                except Exception:
                    pass
                return buy_date
        except Exception:
            pass
        return datetime.now().strftime("%Y%m%d")

    def sync_positions(self, held_codes: set[str], prices: dict[str, float] | None = None,
                       avg_prices: dict[str, float] | None = None,
                       entry_dates: dict[str, str] | None = None,
                       current_date: str | None = None) -> None:
        if current_date:
            self._current_date = current_date.replace("-", "")
        for code in list(self._positions):
            if code not in held_codes:
                del self._positions[code]
        for code in held_codes:
            if code not in self._positions:
                cur_price = prices.get(code, 0) if prices else 0
                avg_price = avg_prices.get(code, 0) if avg_prices else 0
                entry_price = float(avg_price) if avg_price > 0 else (float(cur_price) if cur_price > 0 else 1.0)
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

        # ── Alpha Layer: XGBoost 시그널 ──
        try:
            ml_signal = self.ml_predictor.predict(df)  # 1=BUY, -1=SELL, 0=HOLD
        except Exception:
            ml_signal = 0

        # XGBoost가 HOLD이면 → 바로 HOLD (알파 없음)
        if ml_signal == 0:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # ── RL Layer: 리스크/타이밍 필터 ──
        pos = self._positions.get(stock_code)
        holding = pos is not None
        unrealized_pnl = 0.0
        holding_days = 0

        if holding:
            unrealized_pnl = (price / pos["entry_price"] - 1.0) if pos["entry_price"] > 0 else 0.0
            holding_days = self._business_days_held(pos["entry_date"], self._current_date)

        try:
            rl_signal = self.rl_predictor.predict_with_position(
                df, holding, unrealized_pnl, holding_days,
                buy_threshold=self._buy_threshold,
                sell_threshold=self._sell_threshold,
            )
        except Exception:
            # RL 실패 시 XGBoost 시그널 그대로 사용
            rl_signal = ml_signal

        # ── Hybrid 결합 로직 ──

        # SELL: XGBoost SELL → 무조건 실행 (리스크 관리 우선)
        if ml_signal == -1 and holding:
            # RL도 SELL이면 강한 SELL, 아니어도 실행
            strength = 0.9 if rl_signal == -1 else 0.65
            reason = "Hybrid SELL (XGB+RL 동의)" if rl_signal == -1 else "Hybrid SELL (XGB, RL 보류 무시)"
            return TradeSignal(
                signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                price=price, strength=strength,
                reason=f"{reason} (보유 {holding_days}일, PnL {unrealized_pnl:+.1%})",
            )

        # BUY: XGBoost BUY + RL BUY → 실행 (두 모델 동의)
        if ml_signal == 1 and not holding:
            if rl_signal == 1:
                from datetime import datetime
                entry_date = self._current_date or datetime.now().strftime("%Y%m%d")
                self._positions[stock_code] = {
                    "entry_price": float(price),
                    "entry_date": entry_date,
                }
                return TradeSignal(
                    signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=0.85,
                    reason="Hybrid BUY (XGB alpha + RL timing 동의)",
                )
            else:
                # XGBoost BUY지만 RL이 거부 → HOLD (RL이 리스크 차단)
                return TradeSignal(
                    signal=Signal.HOLD, stock_code=stock_code, price=price,
                    reason=f"Hybrid HOLD (XGB BUY, RL 거부 — 타이밍 부적절)",
                )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
