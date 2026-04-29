"""
강화학습 트레이딩 환경 — Risk-Adjusted Reward
- Differential Sharpe Ratio: 스텝마다 Sharpe를 직접 최적화
- Position-aware reward: 보유 상태에 따른 log return
- Transaction cost, drawdown penalty, holding cost 반영
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.timing.features import build_features


# Action 상수
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2


class TradingEnv:
    """Risk-Adjusted Reward 트레이딩 환경"""

    def __init__(
        self,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
        dsr_eta: float = 0.01,
        drawdown_lambda: float = 1.0,
        drawdown_threshold: float = 0.05,
        holding_cost: float = 0.0002,       # 일별 보유 비용 (v2: 0.0005 → 과매매)
        holding_cost_ramp: float = 0.00005,  # 보유일 증가에 따른 추가 비용
        opportunity_weight: float = 0.35,    # 미보유 시 기회비용 가중치 (v2: 0.5 → 과매매)
        invalid_penalty: float = 0.002,
        sentiment_lambda: float = 0.0,       # SAPPO: reward += lambda * sentiment(date)
        # Trading-frequency penalty (2026-04-28 추가 — 과매매 정책 학습 차단)
        # v2 (2026-04-29): 0.01 → 0.003 약화. v6 ensemble 결과 trades 0.9~1.0 으로 너무 신중 →
        # portfolio 환경에서 정책 경직. portfolio context 는 이미 30종목 분산이 있어
        # 단일종목 cautious 강제는 역효과.
        min_holding_days: int = 5,
        short_hold_penalty: float = 0.003,
    ):
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate

        # Differential Sharpe Ratio 파라미터
        self.dsr_eta = dsr_eta  # EMA 감쇠율

        # 리스크 패널티 파라미터
        self.drawdown_lambda = drawdown_lambda
        self.drawdown_threshold = drawdown_threshold
        self.holding_cost = holding_cost
        self.holding_cost_ramp = holding_cost_ramp
        self.opportunity_weight = opportunity_weight
        self.invalid_penalty = invalid_penalty
        self.sentiment_lambda = sentiment_lambda

        # Trading-frequency penalty
        self.min_holding_days = min_holding_days
        self.short_hold_penalty = short_hold_penalty

        # 에피소드 상태
        self._features: np.ndarray | None = None
        self._prices: np.ndarray | None = None
        self._log_prices: np.ndarray | None = None
        self._dates: np.ndarray | None = None      # YYYYMMDD 문자열 배열
        self._sentiment_series: dict[str, float] | None = None
        self._step: int = 0
        self._max_steps: int = 0

        # 포지션 상태
        self._holding: bool = False
        self._entry_price: float = 0.0
        self._holding_days: int = 0

        # 포트폴리오 추적
        self._portfolio_value: float = 1.0
        self._peak_value: float = 1.0

        # Differential Sharpe Ratio 상태
        self._dsr_A: float = 0.0  # EMA of returns
        self._dsr_B: float = 0.0  # EMA of squared returns

        # 에피소드 통계
        self._n_trades: int = 0
        self._total_cost: float = 0.0

    @property
    def state_dim(self) -> int:
        if self._features is not None:
            return self._features.shape[1] + 3
        return 43

    @property
    def n_actions(self) -> int:
        return 3

    def reset(
        self,
        df: pd.DataFrame,
        sentiment_series: dict[str, float] | pd.Series | None = None,
    ) -> np.ndarray:
        """에피소드 초기화.

        Args:
            df: OHLCV DataFrame (date 컬럼 필요)
            sentiment_series: {YYYYMMDD: score} dict 또는 pd.Series. 없는 날은 0 으로 처리.
        """
        features = build_features(df)
        valid_mask = features.notna().all(axis=1)
        valid_idx = valid_mask[valid_mask].index

        if len(valid_idx) < 10:
            raise ValueError("유효 데이터가 부족합니다 (최소 10일)")

        self._features = features.loc[valid_idx].values.astype(np.float32)
        self._prices = df.loc[valid_idx, "close"].values.astype(np.float64)
        self._log_prices = np.log(self._prices)

        # sentiment 용 날짜 배열 저장 (YYYYMMDD)
        if "date" in df.columns:
            dates_raw = df.loc[valid_idx, "date"]
            if pd.api.types.is_datetime64_any_dtype(dates_raw):
                self._dates = dates_raw.dt.strftime("%Y%m%d").values
            else:
                self._dates = dates_raw.astype(str).str.replace("-", "").values
        else:
            self._dates = None

        # sentiment_series 를 dict 로 정규화
        if sentiment_series is None or self.sentiment_lambda == 0.0:
            self._sentiment_series = None
        elif isinstance(sentiment_series, pd.Series):
            self._sentiment_series = {
                str(k).replace("-", ""): float(v) for k, v in sentiment_series.items()
            }
        elif isinstance(sentiment_series, dict):
            self._sentiment_series = {
                str(k).replace("-", ""): float(v) for k, v in sentiment_series.items()
            }
        else:
            self._sentiment_series = None

        self._step = 0
        self._max_steps = len(self._features) - 1

        # 포지션 초기화
        self._holding = False
        self._entry_price = 0.0
        self._holding_days = 0

        # 포트폴리오 초기화
        self._portfolio_value = 1.0
        self._peak_value = 1.0

        # DSR 초기화
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        # 통계 초기화
        self._n_trades = 0
        self._total_cost = 0.0

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """한 스텝 진행. Risk-adjusted reward를 반환합니다."""
        if self._step >= self._max_steps:
            return self._get_state(), 0.0, True, {}

        current_price = self._prices[self._step]
        next_price = self._prices[self._step + 1]
        log_return = self._log_prices[self._step + 1] - self._log_prices[self._step]

        # ── 1. Base return: 포지션에 따른 수익 ──
        base_return = 0.0
        cost = 0.0
        traded = False

        if action == ACTION_BUY and not self._holding:
            # 매수 진입
            self._holding = True
            self._entry_price = current_price
            self._holding_days = 0
            cost = self.commission_rate
            base_return = log_return - cost  # 매수 후 즉시 노출
            traded = True
            self._n_trades += 1

        elif action == ACTION_SELL and self._holding:
            # 매도 청산
            cost = self.commission_rate + self.tax_rate
            realized_pnl = (current_price - self._entry_price) / self._entry_price

            # SELL 보너스/페널티 v3 (2026-04-29 재조정):
            # v2 가 너무 강한 페널티(+10% ⇒ 0.6×pnl) 로 정책이 *매수 자체를 회피* 하게 학습됨
            # (v6 ensemble: trades 0.9~1.0, portfolio_sharpe -0.07). v3 는 추세 추종 인센티브
            # 는 유지하되 강도를 절반으로 완화 — portfolio 환경 cautiousness 역효과 회피.
            # - 손절(-3% 이하): 보너스 유지 (손실 확대 방지 학습)
            # - 미세 익절(+0~3%): 작은 페널티 0.10 (잡 trades 억제, but v2 의 0.15 보다 약화)
            # - 적정 익절(+3~10%): 중립적 작은 보상 0.10 유지
            # - 조기 익절(+10% 이상): 페널티 0.30 (v2 의 0.6 → 절반)
            if realized_pnl < -0.03:
                sell_bonus = 0.5 * abs(realized_pnl)         # 손절 보상 유지
            elif 0.0 <= realized_pnl < 0.03:
                sell_bonus = -0.10 * realized_pnl            # v2 0.15 → 0.10 약화
            elif 0.03 <= realized_pnl <= 0.10:
                sell_bonus = 0.10 * realized_pnl             # 유지
            elif realized_pnl > 0.10:
                sell_bonus = -0.30 * realized_pnl            # v2 0.6 → 0.30 절반
            else:
                sell_bonus = 0.0
            # 정량 비교 (예시):
            #   +5% 익절: bonus = 0.005 (v2 동일)
            #   +15% 추세 청산: penalty = -0.045 (v2 의 -0.090 → 절반, 여전히 5% 익절보다 큰 억제)
            #   → 추세 추종 인센티브 유지 + cautious 함정 회피

            # Trading-frequency penalty: min_holding_days 미만 보유 후 SELL 시 추가 페널티
            # 손절(-3% 이하)은 면제 (손실 확대 방지가 우선)
            short_hold_pen = 0.0
            if self._holding_days < self.min_holding_days and realized_pnl >= -0.03:
                short_hold_pen = self.short_hold_penalty * (
                    1.0 - self._holding_days / self.min_holding_days
                )

            base_return = -cost + sell_bonus - short_hold_pen
            self._holding = False
            self._entry_price = 0.0
            self._holding_days = 0
            traded = True
            self._n_trades += 1

        elif action == ACTION_BUY and self._holding:
            # 이미 보유 중 매수 시도 → 무효 행동
            base_return = log_return - self.invalid_penalty
            self._holding_days += 1

        elif action == ACTION_SELL and not self._holding:
            # 미보유 매도 시도 → 무효 행동
            base_return = -self.invalid_penalty

        elif action == ACTION_HOLD and self._holding:
            # 보유 유지 → log return - 체증하는 holding cost
            # 단기(~5일): 낮은 비용 → 장기(20일+): 높은 비용
            ramp_cost = self.holding_cost + self.holding_cost_ramp * min(self._holding_days, 20)
            base_return = log_return - ramp_cost
            self._holding_days += 1

        else:
            # 미보유 HOLD → 기회비용: 상승장에서 안 사면 벌점
            # 상승(log_return > 0): 놓친 수익의 opportunity_weight만큼 페널티
            # 하락(log_return < 0): 약한 보상 (올바른 관망)
            if log_return > 0:
                base_return = -log_return * self.opportunity_weight
            else:
                base_return = -log_return * 0.1  # 하락장 관망 보상

        self._total_cost += cost

        # ── 2. 포트폴리오 가치 업데이트 ──
        self._portfolio_value *= np.exp(base_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # ── 3. Differential Sharpe Ratio ──
        dsr_reward = self._compute_dsr(base_return)

        # ── 4. Drawdown penalty ──
        drawdown = 1.0 - self._portfolio_value / self._peak_value
        dd_penalty = 0.0
        if drawdown > self.drawdown_threshold:
            dd_penalty = self.drawdown_lambda * (drawdown - self.drawdown_threshold)

        # ── 5. 최종 보상 조합 ──
        reward = dsr_reward - dd_penalty

        # ── 5-1. SAPPO: sentiment-weighted reward term ──
        sentiment_term = 0.0
        if self.sentiment_lambda > 0.0 and self._sentiment_series is not None and self._dates is not None:
            try:
                date_key = str(self._dates[self._step])
                sentiment_value = self._sentiment_series.get(date_key, 0.0)
                sentiment_term = self.sentiment_lambda * sentiment_value
                reward += sentiment_term
            except (IndexError, KeyError, ValueError):
                pass

        # ── 6. 에피소드 종료 처리 ──
        self._step += 1
        done = self._step >= self._max_steps

        if done and self._holding:
            # 강제 청산 비용
            close_cost = self.commission_rate + self.tax_rate
            self._portfolio_value *= (1.0 - close_cost)

        info = {
            "portfolio_value": self._portfolio_value,
            "drawdown": drawdown,
            "traded": traded,
            "holding": self._holding,
            "base_return": base_return,
            "dsr_reward": dsr_reward,
            "dd_penalty": dd_penalty,
            "sentiment_term": sentiment_term,
            "n_trades": self._n_trades,
            "total_cost": self._total_cost,
        }

        return self._get_state(), reward, done, info

    def _compute_dsr(self, r: float) -> float:
        """Differential Sharpe Ratio를 계산합니다.

        Moody & Saffell (1998): 스텝마다 Sharpe ratio의 증분을 보상으로 사용.
        DSR_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}
        """
        eta = self.dsr_eta
        delta_A = r - self._dsr_A
        delta_B = r * r - self._dsr_B

        denom = self._dsr_B - self._dsr_A ** 2
        if denom > 1e-12:
            dsr = (self._dsr_B * delta_A - 0.5 * self._dsr_A * delta_B) / (denom ** 1.5)
        else:
            # 초기 스텝: 분산이 없을 때 return 자체를 사용
            dsr = r

        # EMA 업데이트
        self._dsr_A += eta * delta_A
        self._dsr_B += eta * delta_B

        return float(np.clip(dsr, -2.0, 2.0))

    def _get_state(self) -> np.ndarray:
        """현재 상태 벡터를 반환합니다."""
        step = min(self._step, len(self._features) - 1)
        features = self._features[step]

        position_state = np.array([
            1.0 if self._holding else 0.0,
            (self._prices[step] / self._entry_price - 1.0)
            if self._holding and self._entry_price > 0 else 0.0,
            min(self._holding_days / 20.0, 1.0),
        ], dtype=np.float32)

        state = np.concatenate([features, position_state])
        return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
