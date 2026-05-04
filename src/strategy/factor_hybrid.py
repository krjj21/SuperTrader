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

    def __init__(
        self,
        ml_model_path: str,
        rl_model_path: str,
        params: dict | None = None,
        ml_model_type: str = "xgboost",
        name: str = "factor_hybrid",
        ml_buy_threshold: float | None = None,
        ml_sell_threshold: float | None = None,
    ):
        super().__init__(name=name, params=params)
        from src.timing.predictor import TimingPredictor
        from src.config import get_config

        # Alpha Layer: ml_model_type (xgboost 또는 transformer)
        self._ml_model_type = ml_model_type
        self._ml_label = "XGB" if ml_model_type == "xgboost" else ml_model_type.capitalize()
        self.ml_predictor = TimingPredictor(ml_model_type, ml_model_path)

        # RL Layer: PPO (리스크 필터)
        self.rl_predictor = TimingPredictor("rl", rl_model_path)
        rl_cfg = get_config().timing.rl
        self._buy_threshold = rl_cfg.buy_action_threshold
        self._sell_threshold = rl_cfg.sell_action_threshold
        # 임계값 우선순위: 명시적 인자 > ml_model_type 기반 config > XGB 기본값
        # transformer 는 softmax 분포가 평탄해 XGB 와 같은 0.55/0.60 사용 시 게이팅 무력화.
        if ml_buy_threshold is not None and ml_sell_threshold is not None:
            self._ml_buy_threshold = float(ml_buy_threshold)
            self._ml_sell_threshold = float(ml_sell_threshold)
        elif ml_model_type == "transformer":
            self._ml_buy_threshold = float(getattr(rl_cfg, "transformer_buy_confidence_threshold", 0.70))
            self._ml_sell_threshold = float(getattr(rl_cfg, "transformer_sell_confidence_threshold", 0.75))
        else:
            self._ml_buy_threshold = float(getattr(rl_cfg, "xgb_buy_confidence_threshold", 0.55))
            self._ml_sell_threshold = float(getattr(rl_cfg, "xgb_sell_confidence_threshold", 0.60))

        # Profit-aware adaptive SELL threshold (#3, 2026-05-04)
        self._profit_aware_enabled = bool(getattr(rl_cfg, "profit_aware_sell_enabled", True))
        self._profit_aware_floor = float(getattr(rl_cfg, "profit_aware_sell_pnl_floor", 0.10))
        self._profit_aware_ceiling = float(getattr(rl_cfg, "profit_aware_sell_pnl_ceiling", 0.30))
        self._profit_aware_max_disc = float(getattr(rl_cfg, "profit_aware_sell_max_discount", 0.20))

        self._pool: set[str] = set()
        # 포지션 추적 (RL 레이어용)
        self._positions: dict[str, dict] = {}
        # 백테스트용 기준일 (라이브에서는 None)
        self._current_date: str | None = None
        # 진단 정보 (#4, 2026-05-04) — 매 generate_signal 직후 inference_logger 가 읽음
        self._last_diag: dict = {}

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

        # 진단 dict — generate_signal 호출자가 self._last_diag 로 읽음 (#4)
        diag: dict = {
            "stock_code": stock_code,
            "price": price,
            "df_len": len(df),
            "holding": False,
            "unrealized_pnl": 0.0,
            "holding_days": 0,
            "ml_signal": 0,
            "ml_buy_prob": None,
            "ml_sell_prob": None,
            "rl_signal": 0,
            "rl_p_hold": None,
            "rl_p_buy": None,
            "rl_p_sell": None,
            "ml_sell_threshold": self._ml_sell_threshold,
            "effective_sell_threshold": self._ml_sell_threshold,
            "final_signal": "HOLD",
        }
        self._last_diag = diag  # by-reference, 아래에서 mutate

        if len(df) < 60:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # ── Position state ──
        pos = self._positions.get(stock_code)
        holding = pos is not None
        unrealized_pnl = 0.0
        holding_days = 0
        if holding:
            unrealized_pnl = (price / pos["entry_price"] - 1.0) if pos["entry_price"] > 0 else 0.0
            holding_days = self._business_days_held(pos["entry_date"], self._current_date)
        diag["holding"] = holding
        diag["unrealized_pnl"] = unrealized_pnl
        diag["holding_days"] = holding_days

        # ── Alpha Layer: XGBoost 시그널 + 양방향 confidence (진단용) ──
        try:
            ml_signal = self.ml_predictor.predict(df)  # 1=BUY, -1=SELL, 0=HOLD
        except Exception:
            ml_signal = 0
        diag["ml_signal"] = int(ml_signal)
        # 양방향 confidence — 모든 사이클 진단 적재 위해 무조건 계산 (~ms 비용)
        try:
            diag["ml_buy_prob"] = self.ml_predictor.predict_proba_last(df, label=1)
            diag["ml_sell_prob"] = self.ml_predictor.predict_proba_last(df, label=-1)
        except Exception:
            pass

        # XGBoost 가 HOLD 면 RL 호출 없이 즉시 종료 (속도 보존)
        if ml_signal == 0:
            return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)

        # ── RL Layer: predict_with_position_with_probs 로 raw probs 도 캡처 ──
        try:
            rl_signal, rl_probs = self.rl_predictor.predict_with_position_with_probs(
                df, holding, unrealized_pnl, holding_days,
                buy_threshold=self._buy_threshold,
                sell_threshold=self._sell_threshold,
            )
        except Exception:
            rl_signal, rl_probs = ml_signal, None
        diag["rl_signal"] = int(rl_signal)
        if rl_probs is not None and len(rl_probs) >= 3:
            diag["rl_p_hold"] = float(rl_probs[0])
            diag["rl_p_buy"] = float(rl_probs[1])
            diag["rl_p_sell"] = float(rl_probs[2])

        # ── Hybrid 결합 로직 ──

        # SELL: Alpha SELL + (RL 동의 OR Alpha 고신뢰) → 실행
        if ml_signal == -1 and holding:
            ml_sell_prob = diag["ml_sell_prob"]
            conf_str = f"{ml_sell_prob:.2f}" if ml_sell_prob is not None else "n/a"

            # Profit-aware adaptive threshold (#3): pnl floor~ceiling 사이에서 선형 인하.
            # 예: floor=0.10, ceiling=0.30, max_disc=0.20 → pnl=+30% 시 0.60 × 0.80 = 0.48.
            # __new__ 로 우회 생성된 인스턴스(테스트) 호환 위해 getattr 기본값 사용.
            pa_enabled = getattr(self, "_profit_aware_enabled", False)
            pa_floor = getattr(self, "_profit_aware_floor", 0.10)
            pa_ceiling = getattr(self, "_profit_aware_ceiling", 0.30)
            pa_max_disc = getattr(self, "_profit_aware_max_disc", 0.20)
            effective_threshold = self._ml_sell_threshold
            if pa_enabled and unrealized_pnl >= pa_floor:
                span = pa_ceiling - pa_floor
                if span > 0:
                    ratio = min(1.0, max(0.0, (unrealized_pnl - pa_floor) / span))
                else:
                    ratio = 1.0
                effective_threshold = self._ml_sell_threshold * (1.0 - ratio * pa_max_disc)
            diag["effective_sell_threshold"] = float(effective_threshold)
            high_conf = ml_sell_prob is None or ml_sell_prob >= effective_threshold

            if rl_signal == -1:
                diag["final_signal"] = "SELL"
                return TradeSignal(
                    signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=0.90,
                    reason=(
                        f"Hybrid SELL ({self._ml_label}+RL 동의, conf={conf_str}) "
                        f"보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                    ),
                )
            if high_conf:
                eff_note = ""
                if effective_threshold < self._ml_sell_threshold - 1e-6:
                    eff_note = f" (profit-aware: {self._ml_sell_threshold:.2f}→{effective_threshold:.2f})"
                diag["final_signal"] = "SELL"
                return TradeSignal(
                    signal=Signal.SELL, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=0.70,
                    reason=(
                        f"Hybrid SELL ({self._ml_label} 고신뢰 conf={conf_str} ≥ {effective_threshold:.2f}{eff_note}, "
                        f"RL 보류 무시) 보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                    ),
                )
            diag["final_signal"] = "HOLD"
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=(
                    f"Hybrid HOLD ({self._ml_label} 저신뢰 conf={conf_str} < {effective_threshold:.2f} "
                    f"AND RL 반대) 보유 {holding_days}일, PnL {unrealized_pnl:+.1%}"
                ),
            )

        # BUY: Alpha BUY + Alpha 고신뢰 + RL BUY → 3중 합의
        if ml_signal == 1 and not holding:
            ml_buy_prob = diag["ml_buy_prob"]
            buy_conf_pass = ml_buy_prob is None or ml_buy_prob >= self._ml_buy_threshold
            buy_conf_str = f"{ml_buy_prob:.2f}" if ml_buy_prob is not None else "n/a"

            if not buy_conf_pass:
                diag["final_signal"] = "HOLD"
                return TradeSignal(
                    signal=Signal.HOLD, stock_code=stock_code, price=price,
                    reason=(
                        f"Hybrid HOLD ({self._ml_label} 저신뢰 conf={buy_conf_str} "
                        f"< {self._ml_buy_threshold:.2f} — 진입 보류)"
                    ),
                )

            if rl_signal == 1:
                from datetime import datetime
                entry_date = self._current_date or datetime.now().strftime("%Y%m%d")
                self._positions[stock_code] = {
                    "entry_price": float(price),
                    "entry_date": entry_date,
                }
                if ml_buy_prob is None:
                    strength = 0.85
                else:
                    span = max(1.0 - self._ml_buy_threshold, 1e-6)
                    norm = (ml_buy_prob - self._ml_buy_threshold) / span
                    strength = float(min(max(0.5 + 0.5 * norm, 0.5), 1.0))
                diag["final_signal"] = "BUY"
                return TradeSignal(
                    signal=Signal.BUY, stock_code=stock_code, stock_name=stock_name,
                    price=price, strength=strength,
                    reason=f"Hybrid BUY ({self._ml_label} conf={buy_conf_str} + RL 동의, size={strength:.2f})",
                )
            diag["final_signal"] = "HOLD"
            return TradeSignal(
                signal=Signal.HOLD, stock_code=stock_code, price=price,
                reason=(
                    f"Hybrid HOLD ({self._ml_label} BUY conf={buy_conf_str}, RL 거부 — "
                    f"타이밍 부적절)"
                ),
            )

        return TradeSignal(signal=Signal.HOLD, stock_code=stock_code, price=price)
