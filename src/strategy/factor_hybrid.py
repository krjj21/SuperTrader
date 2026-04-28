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
from src.strategy._position_utils import business_days_held, resolve_buy_date


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
        self._xgb_sell_threshold = float(getattr(rl_cfg, "xgb_sell_confidence_threshold", 0.60))
        self._xgb_buy_threshold = float(getattr(rl_cfg, "xgb_buy_confidence_threshold", 0.55))

        self._pool: set[str] = set()
        # 포지션 추적 (RL 레이어용)
        self._positions: dict[str, dict] = {}
        # 백테스트용 기준일 (라이브에서는 None)
        self._current_date: str | None = None

    def update_pool(self, codes: list[str]) -> None:
        self._pool = set(codes)

    _business_days_held = staticmethod(business_days_held)
    _resolve_buy_date = staticmethod(resolve_buy_date)

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

        # SELL: XGBoost SELL 시 신뢰도 + RL 합의로 3-way 분기
        if ml_signal == -1 and holding:
            xgb_sell_prob = self.ml_predictor.predict_proba_last(df, label=-1)
            # predict_proba 미지원 또는 실패 시 기존 동작(강제 실행) 유지
            high_conf = xgb_sell_prob is None or xgb_sell_prob >= self._xgb_sell_threshold
            conf_str = f"{xgb_sell_prob:.2f}" if xgb_sell_prob is not None else "n/a"

            if rl_signal == -1:
                # 1) RL 동의 — 항상 실행 (신뢰도 무관)
                return TradeSignal(
                    signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=0.90,
                    reason=(
                        f"Hybrid SELL (XGB+RL 동의, conf={conf_str}) "
                        f"보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                    ),
                )
            if high_conf:
                # 2) RL 보류이지만 XGB 고신뢰 → 실행
                return TradeSignal(
                    signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=0.70,
                    reason=(
                        f"Hybrid SELL (XGB 고신뢰 conf={conf_str} ≥ {self._xgb_sell_threshold:.2f}, "
                        f"RL 보류 무시) 보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                    ),
                )
            # 3) RL 보류 AND XGB 저신뢰 → HOLD (신규 보호)
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=(
                    f"Hybrid HOLD (XGB 저신뢰 conf={conf_str} < {self._xgb_sell_threshold:.2f} "
                    f"AND RL 반대) 보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                ),
            )

        # BUY: XGBoost BUY + XGB 고신뢰 + RL BUY → 실행 (3중 합의)
        # 2026-04-28: BUY 에 xgb_buy_confidence_threshold 추가. 잘못된 매수는 즉시 손실,
        # 잘못된 보류는 기회 비용에 그치므로 BUY 가 SELL 보다 더 엄격해야 한다.
        if ml_signal == 1 and not holding:
            xgb_buy_prob = self.ml_predictor.predict_proba_last(df, label=1)
            # predict_proba 미지원 시 None → 기존 동작(통과) 유지 (호환성)
            buy_conf_pass = xgb_buy_prob is None or xgb_buy_prob >= self._xgb_buy_threshold
            buy_conf_str = f"{xgb_buy_prob:.2f}" if xgb_buy_prob is not None else "n/a"

            if not buy_conf_pass:
                # XGB BUY 시그널이지만 신뢰도 미달 → HOLD (false positive 차단)
                return TradeSignal(
                    signal=Signal.HOLD, stock_code=stock_code, price=price,
                    reason=(
                        f"Hybrid HOLD (XGB 저신뢰 conf={buy_conf_str} "
                        f"< {self._xgb_buy_threshold:.2f} — 진입 보류)"
                    ),
                )

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
                    reason=f"Hybrid BUY (XGB conf={buy_conf_str} + RL 동의)",
                )
            else:
                # XGB 고신뢰 BUY지만 RL이 거부 → HOLD (RL이 타이밍 리스크 차단)
                return TradeSignal(
                    signal=Signal.HOLD, stock_code=stock_code, price=price,
                    reason=(
                        f"Hybrid HOLD (XGB BUY conf={buy_conf_str}, RL 거부 — "
                        f"타이밍 부적절)"
                    ),
                )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
