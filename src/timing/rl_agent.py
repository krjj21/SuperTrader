"""
GRPO (Group Relative Policy Optimization) 트레이딩 에이전트
- Critic-free: 그룹 내 상대 비교로 advantage 계산
- 기존 타이밍 모델 인터페이스 호환 (train/predict/save/load)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from src.timing.features import build_features


class ActorNetwork(nn.Module):
    """GRPO Actor 네트워크 (정책)"""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """행동 확률 분포를 반환합니다."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

    def get_log_prob(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """특정 행동의 로그 확률을 반환합니다."""
        probs = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

    def sample_action(self, state: torch.Tensor) -> tuple[int, float]:
        """행동을 샘플링하고 로그 확률을 반환합니다."""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()


class RLTimingModel:
    """GRPO 기반 타이밍 모델 — 기존 모델 인터페이스 호환"""

    def __init__(
        self,
        state_dim: int = 43,
        hidden_dim: int = 256,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        group_size: int = 8,
        clip_epsilon_low: float = 0.2,
        clip_epsilon_high: float = 0.28,
        entropy_coeff: float = 0.01,
        epochs_per_update: int = 4,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.group_size = group_size
        self.clip_epsilon_low = clip_epsilon_low
        self.clip_epsilon_high = clip_epsilon_high
        self.entropy_coeff = entropy_coeff
        self.epochs_per_update = epochs_per_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorNetwork(state_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.feature_names: list[str] = []

    def compute_group_advantage(self, rewards: np.ndarray) -> np.ndarray:
        """GRPO 그룹 advantage를 계산합니다.

        A_G(s, a_i) = (r_i - μ) / (σ + ε)
        """
        mu = rewards.mean()
        sigma = rewards.std() + 1e-8
        return (rewards - mu) / sigma

    def grpo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> dict:
        """GRPO 정책 업데이트 (Decoupled Clipping).

        Args:
            states: (batch, state_dim)
            actions: (batch,)
            advantages: (batch,) — 그룹 정규화된 advantage
            old_log_probs: (batch,) — 이전 정책의 로그 확률

        Returns:
            학습 통계 딕셔너리
        """
        total_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs_per_update):
            new_log_probs = self.actor.get_log_prob(states, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Decoupled Clipping: 비대칭 클리핑
            clip_low = 1.0 - self.clip_epsilon_low
            clip_high = 1.0 + self.clip_epsilon_high
            clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 엔트로피 보너스 (탐험 유도)
            probs = self.actor(states)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            entropy_loss = -self.entropy_coeff * entropy

            loss = policy_loss + entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item()

        n = self.epochs_per_update
        return {
            "policy_loss": total_loss / n,
            "entropy": total_entropy / n,
        }

    def collect_episode(self, env, df: pd.DataFrame) -> dict:
        """한 에피소드를 수집하고 GRPO advantage를 계산합니다.

        각 timestep에서 group_size개 행동을 시뮬레이션하여 그룹 비교.
        """
        from src.timing.rl_env import ACTION_HOLD, ACTION_BUY, ACTION_SELL

        state = env.reset(df)
        states, actions, rewards, log_probs, advantages = [], [], [], [], []

        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 그룹 샘플링: 모든 행동의 보상을 시뮬레이션
            group_rewards = np.array([
                env.simulate_action(a) for a in range(env.n_actions)
            ])

            # 그룹 advantage 계산
            group_adv = self.compute_group_advantage(group_rewards)

            # 정책에서 행동 샘플링
            action, log_prob = self.actor.sample_action(state_tensor)

            # advantage는 선택된 행동의 그룹 advantage
            adv = group_adv[action]

            # 환경 스텝
            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            advantages.append(adv)

            total_reward += reward
            state = next_state

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": np.array(rewards, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "advantages": np.array(advantages, dtype=np.float32),
            "total_reward": total_reward,
            "portfolio_value": info.get("portfolio_value", 1.0),
        }

    # ────────────────────────────────────────────
    # 기존 모델 인터페이스 호환
    # ────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict:
        """기존 trainer.py 호환 인터페이스.

        RL 모델은 실제로 rl_trainer.train_rl_model()에서 학습하므로
        여기서는 최소 호환만 제공합니다.
        """
        self.feature_names = X.columns.tolist()
        logger.info("RL 모델: train() 호출됨 — 실제 학습은 rl_trainer를 사용하세요")
        return {"accuracy": 0.0, "n_samples": len(X), "note": "use rl_trainer"}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """피처 DataFrame에서 시그널을 예측합니다.

        Returns:
            Series of {-1: SELL, 0: HOLD, 1: BUY}
        """
        self.actor.eval()
        predictions = pd.Series(0, index=X.index, dtype=int)

        # NaN 제거
        mask = X.notna().all(axis=1)
        if not mask.any():
            return predictions

        X_clean = X[mask].values.astype(np.float32)
        # 포지션 상태 추가 (예측 시에는 미보유 가정)
        position_state = np.zeros((len(X_clean), 3), dtype=np.float32)
        X_full = np.concatenate([X_clean, position_state], axis=1)

        with torch.no_grad():
            states = torch.FloatTensor(X_full).to(self.device)
            probs = self.actor(states)
            actions = probs.argmax(dim=-1).cpu().numpy()

        # action → signal 변환: HOLD(0)→0, BUY(1)→1, SELL(2)→-1
        signal_map = {0: 0, 1: 1, 2: -1}
        signals = np.array([signal_map[a] for a in actions])
        predictions[mask] = signals

        self.actor.train()
        return predictions

    def predict_with_position(
        self,
        X: pd.DataFrame,
        holding: bool = False,
        unrealized_pnl: float = 0.0,
        holding_days: int = 0,
    ) -> int:
        """포지션 상태를 반영하여 단일 종목의 최신 시그널을 예측합니다.

        Returns:
            1 (BUY), 0 (HOLD), -1 (SELL)
        """
        self.actor.eval()
        mask = X.notna().all(axis=1)
        if not mask.any():
            return 0

        # 마지막 유효 row만 사용
        last_valid = X[mask].iloc[[-1]].values.astype(np.float32)
        position_state = np.array([[
            1.0 if holding else 0.0,
            unrealized_pnl,
            min(holding_days / 20.0, 1.0),
        ]], dtype=np.float32)
        X_full = np.concatenate([last_valid, position_state], axis=1)

        with torch.no_grad():
            state = torch.FloatTensor(X_full).to(self.device)
            probs = self.actor(state)
            action = probs.argmax(dim=-1).item()

        self.actor.train()
        # action → signal: HOLD(0)→0, BUY(1)→1, SELL(2)→-1
        return {0: 0, 1: 1, 2: -1}[action]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """행동 확률을 반환합니다."""
        self.actor.eval()
        mask = X.notna().all(axis=1)
        X_clean = X[mask].values.astype(np.float32)
        position_state = np.zeros((len(X_clean), 3), dtype=np.float32)
        X_full = np.concatenate([X_clean, position_state], axis=1)

        with torch.no_grad():
            states = torch.FloatTensor(X_full).to(self.device)
            probs = self.actor(states).cpu().numpy()

        self.actor.train()
        return probs

    def save(self, path: str) -> None:
        """모델을 저장합니다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "feature_names": self.feature_names,
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "group_size": self.group_size,
                "clip_epsilon_low": self.clip_epsilon_low,
                "clip_epsilon_high": self.clip_epsilon_high,
                "entropy_coeff": self.entropy_coeff,
                "epochs_per_update": self.epochs_per_update,
            },
        }, path)
        logger.info(f"RL 모델 저장: {path}")

    def load(self, path: str) -> None:
        """모델을 로드합니다."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.state_dim = checkpoint["state_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.feature_names = checkpoint.get("feature_names", [])

        # 네트워크 재생성 (차원이 다를 수 있으므로)
        self.actor = ActorNetwork(self.state_dim, self.hidden_dim).to(self.device)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])

        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        config = checkpoint.get("config", {})
        for k, v in config.items():
            if hasattr(self, k):
                setattr(self, k, v)

        logger.info(f"RL 모델 로드: {path} (state_dim={self.state_dim})")
