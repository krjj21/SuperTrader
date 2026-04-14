"""
PPO (Proximal Policy Optimization) 트레이딩 에이전트
- Actor-Critic 아키텍처
- GAE (Generalized Advantage Estimation)
- Risk-adjusted reward 환경과 연동
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


class ActorCriticNetwork(nn.Module):
    """PPO Actor-Critic 네트워크 (공유 backbone + 분리 head)"""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_actions: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim // 2, n_actions)
        self.critic_head = nn.Linear(hidden_dim // 2, 1)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01 if m is self.actor_head else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """행동 확률 + 가치 추정을 반환합니다."""
        h = self.backbone(x)
        logits = self.actor_head(h)
        probs = F.softmax(logits, dim=-1)
        value = self.critic_head(h).squeeze(-1)
        return probs, value

    def get_action_value(
        self, states: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동의 log_prob, value, entropy를 반환합니다."""
        probs, values = self.forward(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def sample_action(self, state: torch.Tensor) -> tuple[int, float, float]:
        """행동 샘플링 → (action, log_prob, value)"""
        probs, value = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


# ── 레거시 호환: ActorNetwork alias ──
class ActorNetwork(ActorCriticNetwork):
    pass


class RLTimingModel:
    """PPO 기반 타이밍 모델 — 기존 모델 인터페이스 호환"""

    def __init__(
        self,
        state_dim: int = 43,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        epochs_per_update: int = 4,
        mini_batch_size: int = 64,
        # 아래는 레거시 호환 (무시됨)
        group_size: int = 8,
        clip_epsilon_low: float = 0.2,
        clip_epsilon_high: float = 0.28,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.epochs_per_update = epochs_per_update
        self.mini_batch_size = mini_batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ActorCriticNetwork(state_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.feature_names: list[str] = []

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE (Generalized Advantage Estimation)를 계산합니다.

        Returns:
            (advantages, returns) — 둘 다 (T,) shape
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            # A_t = δ_t + γ * λ * A_{t+1}
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        progress: float = 0.5,
    ) -> dict:
        """PPO 정책 + 가치 업데이트.

        Args:
            progress: 학습 진행도 (0.0 시작 ~ 1.0 종료) — entropy 스케줄링용

        Returns:
            학습 통계
        """
        # Advantage 정규화
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Entropy 스케줄링: 초기 2x → 후반 1x (탐색 점진 감소)
        eff_entropy_coeff = self.entropy_coeff * (2.0 - progress)

        n = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs_per_update):
            # 미니배치 셔플
            indices = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                end = min(start + self.mini_batch_size, n)
                idx = indices[start:end]

                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]

                # 현재 정책 평가
                new_log_probs, values, entropy = self.actor.get_action_value(
                    mb_states, mb_actions,
                )

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = F.mse_loss(values, mb_returns)

                # Entropy bonus (스케줄링 적용)
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    + eff_entropy_coeff * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def collect_episode(self, env, df: pd.DataFrame, deterministic: bool = False) -> dict:
        """에피소드를 수집합니다 (PPO rollout).

        Args:
            deterministic: True이면 argmax 정책 (평가용), False이면 확률적 샘플링 (학습용)
        """
        state = env.reset(df)
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

        done = False
        total_reward = 0.0
        ep_info = {}

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            with torch.no_grad():
                if deterministic:
                    probs, value = self.actor(state_tensor)
                    action = probs.argmax(dim=-1).item()
                    log_prob = torch.log(probs.squeeze(0)[action] + 1e-8).item()
                    value = value.item()
                else:
                    action, log_prob, value = self.actor.sample_action(state_tensor)

            next_state, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(float(done))

            total_reward += reward
            state = next_state
            ep_info = info

        rewards_arr = np.array(rewards, dtype=np.float32)
        values_arr = np.array(values, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)

        # GAE 계산
        advantages, returns = self.compute_gae(rewards_arr, values_arr, dones_arr)

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int64),
            "rewards": rewards_arr,
            "log_probs": np.array(log_probs, dtype=np.float32),
            "values": values_arr,
            "advantages": advantages,
            "returns": returns,
            "total_reward": total_reward,
            "portfolio_value": ep_info.get("portfolio_value", 1.0),
            "n_trades": ep_info.get("n_trades", 0),
        }

    # ── 레거시 호환: grpo_update → ppo_update 위임 ──
    def grpo_update(self, states, actions, advantages, old_log_probs):
        returns = advantages  # 대략적 호환
        return self.ppo_update(states, actions, old_log_probs, advantages, returns)

    def compute_group_advantage(self, rewards):
        mu = rewards.mean()
        sigma = rewards.std() + 1e-8
        return (rewards - mu) / sigma

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
        """기존 trainer.py 호환 인터페이스."""
        self.feature_names = X.columns.tolist()
        logger.info("RL 모델: train() — 실제 학습은 rl_trainer를 사용하세요")
        return {"accuracy": 0.0, "n_samples": len(X), "note": "use rl_trainer"}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """피처 DataFrame에서 시그널을 예측합니다.

        Returns:
            Series of {-1: SELL, 0: HOLD, 1: BUY}
        """
        self.actor.eval()
        predictions = pd.Series(0, index=X.index, dtype=int)

        mask = X.notna().all(axis=1)
        if not mask.any():
            return predictions

        X_clean = X[mask].values.astype(np.float32)
        position_state = np.zeros((len(X_clean), 3), dtype=np.float32)
        X_full = np.concatenate([X_clean, position_state], axis=1)

        with torch.no_grad():
            states = torch.FloatTensor(X_full).to(self.device)
            probs, _ = self.actor(states)
            actions = probs.argmax(dim=-1).cpu().numpy()

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
        buy_threshold: float = 0.08,
        sell_threshold: float = 0.05,
    ) -> int:
        """포지션 상태를 반영하여 단일 종목의 최신 시그널을 예측합니다.

        확률 기반 판단: argmax 대신 threshold를 사용하여
        과도하게 수렴한 정책에서도 시그널 생성 가능.
        """
        self.actor.eval()
        mask = X.notna().all(axis=1)
        if not mask.any():
            return 0

        last_valid = X[mask].iloc[[-1]].values.astype(np.float32)
        position_state = np.array([[
            1.0 if holding else 0.0,
            unrealized_pnl,
            min(holding_days / 20.0, 1.0),
        ]], dtype=np.float32)
        X_full = np.concatenate([last_valid, position_state], axis=1)

        with torch.no_grad():
            state = torch.FloatTensor(X_full).to(self.device)
            probs, _ = self.actor(state)
            p = probs.squeeze(0).cpu().numpy()  # [HOLD, BUY, SELL]

        self.actor.train()

        # 확률 기반 판단: threshold 초과 시 해당 행동 선택
        if not holding and p[1] > buy_threshold:
            return 1   # BUY
        if holding and p[2] > sell_threshold:
            return -1  # SELL
        return 0       # HOLD

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """행동 확률을 반환합니다."""
        self.actor.eval()
        mask = X.notna().all(axis=1)
        X_clean = X[mask].values.astype(np.float32)
        position_state = np.zeros((len(X_clean), 3), dtype=np.float32)
        X_full = np.concatenate([X_clean, position_state], axis=1)

        with torch.no_grad():
            states = torch.FloatTensor(X_full).to(self.device)
            probs, _ = self.actor(states)
            result = probs.cpu().numpy()

        self.actor.train()
        return result

    def save(self, path: str) -> None:
        """모델을 저장합니다."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "feature_names": self.feature_names,
            "model_type": "ppo",
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "entropy_coeff": self.entropy_coeff,
                "value_coeff": self.value_coeff,
                "max_grad_norm": self.max_grad_norm,
                "epochs_per_update": self.epochs_per_update,
                "mini_batch_size": self.mini_batch_size,
            },
        }, path)
        logger.info(f"RL 모델 저장: {path}")

    def load(self, path: str) -> None:
        """모델을 로드합니다. 레거시 GRPO 모델도 자동 변환합니다."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.state_dim = checkpoint["state_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.feature_names = checkpoint.get("feature_names", [])

        self.actor = ActorCriticNetwork(self.state_dim, self.hidden_dim).to(self.device)

        actor_sd = checkpoint["actor_state_dict"]
        is_legacy = any(k.startswith("net.") for k in actor_sd)

        if is_legacy:
            # 레거시 GRPO (ActorNetwork: net.0/2/4) → ActorCritic 변환
            new_sd = self.actor.state_dict()
            key_map = {
                "net.0.weight": "backbone.0.weight",
                "net.0.bias": "backbone.0.bias",
                "net.2.weight": "backbone.3.weight",
                "net.2.bias": "backbone.3.bias",
                "net.4.weight": "actor_head.weight",
                "net.4.bias": "actor_head.bias",
            }
            for old_key, new_key in key_map.items():
                if old_key in actor_sd and new_key in new_sd:
                    if actor_sd[old_key].shape == new_sd[new_key].shape:
                        new_sd[new_key] = actor_sd[old_key]
            self.actor.load_state_dict(new_sd)
            logger.info(f"레거시 GRPO → PPO 변환 로드: {path}")
        else:
            self.actor.load_state_dict(actor_sd)

        self.optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate,
        )
        if not is_legacy:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        config = checkpoint.get("config", {})
        for k, v in config.items():
            if hasattr(self, k):
                setattr(self, k, v)

        logger.info(f"RL 모델 로드: {path} (state_dim={self.state_dim})")
