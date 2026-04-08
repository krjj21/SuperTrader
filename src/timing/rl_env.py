"""
강화학습 트레이딩 환경
- GRPO 학습용 단일 종목 시뮬레이션 환경
- State: 기술 피처(40) + 포지션 상태(3) = 43차원
- Action: HOLD(0), BUY(1), SELL(2)
- Reward: 리스크 조정 수익률 (비용/드로다운 반영)
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
    """단일 종목 트레이딩 환경"""

    def __init__(
        self,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
        drawdown_penalty: float = 0.5,
        hold_bonus: float = 0.001,
    ):
        self.commission_rate = commission_rate
        self.tax_rate = tax_rate
        self.drawdown_penalty = drawdown_penalty
        self.hold_bonus = hold_bonus

        # 에피소드 상태
        self._features: np.ndarray | None = None
        self._prices: np.ndarray | None = None
        self._step: int = 0
        self._max_steps: int = 0

        # 포지션 상태
        self._holding: bool = False
        self._entry_price: float = 0.0
        self._holding_days: int = 0

        # 포트폴리오 추적
        self._portfolio_value: float = 1.0
        self._peak_value: float = 1.0

    @property
    def state_dim(self) -> int:
        """상태 차원 (피처 + 포지션 정보 3개)"""
        if self._features is not None:
            return self._features.shape[1] + 3
        return 43  # 기본값

    @property
    def n_actions(self) -> int:
        return 3

    def reset(self, df: pd.DataFrame) -> np.ndarray:
        """에피소드를 초기화합니다.

        Args:
            df: OHLCV DataFrame (전체 기간)

        Returns:
            초기 상태 벡터
        """
        features = build_features(df)
        # NaN 행 제거하기 위한 유효 인덱스
        valid_mask = features.notna().all(axis=1)
        valid_idx = valid_mask[valid_mask].index

        if len(valid_idx) < 10:
            raise ValueError("유효 데이터가 부족합니다 (최소 10일)")

        self._features = features.loc[valid_idx].values.astype(np.float32)
        self._prices = df.loc[valid_idx, "close"].values.astype(np.float64)
        self._step = 0
        self._max_steps = len(self._features) - 1

        # 포지션 초기화
        self._holding = False
        self._entry_price = 0.0
        self._holding_days = 0

        # 포트폴리오 초기화
        self._portfolio_value = 1.0
        self._peak_value = 1.0

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """한 스텝 진행합니다.

        Args:
            action: 0=HOLD, 1=BUY, 2=SELL

        Returns:
            (next_state, reward, done, info)
        """
        if self._step >= self._max_steps:
            return self._get_state(), 0.0, True, {}

        current_price = self._prices[self._step]
        next_price = self._prices[self._step + 1]
        price_return = (next_price - current_price) / current_price

        reward = 0.0
        trade_cost = 0.0
        traded = False

        # 행동 실행
        if action == ACTION_BUY and not self._holding:
            # 매수
            self._holding = True
            self._entry_price = current_price
            self._holding_days = 0
            trade_cost = current_price * self.commission_rate
            traded = True

        elif action == ACTION_SELL and self._holding:
            # 매도: 실현 수익/손실을 보상에 반영
            realized_return = (current_price - self._entry_price) / self._entry_price
            trade_cost = current_price * (self.commission_rate + self.tax_rate)
            self._holding = False
            self._entry_price = 0.0
            self._holding_days = 0
            traded = True

        # 보상 계산
        if action == ACTION_SELL and traded:
            # 매도 시: 실현 수익이 핵심 보상
            reward = realized_return - trade_cost / current_price
        elif self._holding:
            # 보유 중: 가격 변동 반영
            reward = price_return
            self._holding_days += 1
        elif action == ACTION_BUY and not traded:
            # 이미 보유 중인데 BUY 시도 → 페널티
            reward = -0.001
        elif action == ACTION_SELL and not traded:
            # 미보유인데 SELL 시도 → 페널티
            reward = -0.001
        else:
            # 미보유 HOLD: 하락장이면 보상, 상승장이면 기회비용
            reward = -price_return * 0.3  # 안 샀을 때 올랐으면 패널티, 떨어졌으면 보상

        # 거래비용 차감 (매수 시)
        if traded and action == ACTION_BUY:
            reward -= trade_cost / current_price

        # 포트폴리오 가치 업데이트
        self._portfolio_value *= (1.0 + reward)
        self._peak_value = max(self._peak_value, self._portfolio_value)

        # 드로다운 패널티
        drawdown = (self._peak_value - self._portfolio_value) / self._peak_value
        if drawdown > 0.05:  # 5% 초과 시
            reward -= self.drawdown_penalty * (drawdown - 0.05)

        self._step += 1
        done = self._step >= self._max_steps

        # 에피소드 종료 시 보유 중이면 강제 청산 반영
        if done and self._holding:
            sell_cost = next_price * (self.commission_rate + self.tax_rate)
            reward -= sell_cost / next_price

        info = {
            "portfolio_value": self._portfolio_value,
            "drawdown": drawdown,
            "traded": traded,
            "holding": self._holding,
        }

        return self._get_state(), reward, done, info

    def simulate_action(self, action: int) -> float:
        """현재 상태에서 특정 행동의 즉시 보상을 시뮬레이션합니다 (상태 변경 없음).

        GRPO 그룹 샘플링에서 사용: 여러 행동의 보상을 비교.
        """
        if self._step >= self._max_steps:
            return 0.0

        current_price = self._prices[self._step]
        next_price = self._prices[self._step + 1]
        price_return = (next_price - current_price) / current_price

        reward = 0.0

        if action == ACTION_BUY and not self._holding:
            reward = price_return - self.commission_rate
        elif action == ACTION_SELL and self._holding:
            # 매도: 실현 수익 반영
            realized = (current_price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0
            reward = realized - (self.commission_rate + self.tax_rate)
        elif action == ACTION_BUY and self._holding:
            reward = -0.001  # 이미 보유 중 BUY 시도
        elif action == ACTION_SELL and not self._holding:
            reward = -0.001  # 미보유 SELL 시도
        elif action == ACTION_HOLD and self._holding:
            reward = price_return  # 보유 중 HOLD
        else:
            # 미보유 HOLD: 기회비용 반영
            reward = -price_return * 0.3

        # 드로다운 패널티 (시뮬레이션)
        sim_value = self._portfolio_value * (1.0 + reward)
        drawdown = (self._peak_value - sim_value) / self._peak_value
        if drawdown > 0.05:
            reward -= self.drawdown_penalty * (drawdown - 0.05)

        return reward

    def _get_state(self) -> np.ndarray:
        """현재 상태 벡터를 반환합니다."""
        if self._step >= len(self._features):
            step = len(self._features) - 1
        else:
            step = self._step

        features = self._features[step]

        # 포지션 상태 추가
        position_state = np.array([
            1.0 if self._holding else 0.0,
            # 미실현 수익률
            (self._prices[step] / self._entry_price - 1.0) if self._holding and self._entry_price > 0 else 0.0,
            # 보유일수 (정규화)
            min(self._holding_days / 20.0, 1.0),
        ], dtype=np.float32)

        state = np.concatenate([features, position_state])
        # NaN/Inf 처리
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state
