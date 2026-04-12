"""
PPO 학습 파이프라인
- 다중 종목 에피소드 학습 (rollout 배치)
- Walk-forward 검증
- Sharpe ratio 기반 최적 모델 선택
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import torch

from src.config import get_config
from src.timing.rl_env import TradingEnv
from src.timing.rl_agent import RLTimingModel


def evaluate_rl_agent(
    agent: RLTimingModel,
    ohlcv_dict: dict[str, pd.DataFrame],
    commission_rate: float = 0.00015,
    tax_rate: float = 0.0023,
) -> dict:
    """RL 에이전트의 성능을 평가합니다."""
    env = TradingEnv(commission_rate=commission_rate, tax_rate=tax_rate)
    episode_returns = []
    episode_trades = []

    agent.actor.eval()
    for code, df in ohlcv_dict.items():
        if len(df) < 100:
            continue

        try:
            result = agent.collect_episode(env, df)
            episode_returns.append(result["portfolio_value"] - 1.0)
            episode_trades.append(result.get("n_trades", 0))
        except Exception:
            continue
    agent.actor.train()

    if not episode_returns:
        return {"sharpe": -999.0, "total_return": 0.0, "n_episodes": 0}

    returns = np.array(episode_returns)
    mean_ret = returns.mean()
    std_ret = returns.std() + 1e-8
    sharpe = mean_ret / std_ret * np.sqrt(250)

    return {
        "sharpe": float(sharpe),
        "total_return": float(mean_ret * 100),
        "max_drawdown": float(returns.min() * 100),
        "n_episodes": len(episode_returns),
        "win_rate": float((returns > 0).sum() / len(returns) * 100),
        "avg_trades": float(np.mean(episode_trades)) if episode_trades else 0.0,
    }


def train_rl_model(
    ohlcv_dict: dict[str, pd.DataFrame],
    save_path: str = "models/rl_timing.pt",
    val_ratio: float = 0.2,
) -> dict:
    """PPO로 RL 타이밍 모델을 학습합니다."""
    config = get_config()
    rl_config = config.timing.rl

    # 데이터 분할: 시간 기반 train/val
    train_dict = {}
    val_dict = {}

    for code, df in ohlcv_dict.items():
        if len(df) < 100:
            continue
        split_idx = int(len(df) * (1 - val_ratio))
        train_dict[code] = df.iloc[:split_idx].reset_index(drop=True)
        val_dict[code] = df.iloc[split_idx:].reset_index(drop=True)

    if not train_dict:
        return {"error": "no_data"}

    logger.info(f"PPO 학습 데이터: {len(train_dict)}종목, val: {len(val_dict)}종목")

    # 환경 및 에이전트 생성
    env = TradingEnv(
        commission_rate=config.backtest.commission_rate,
        tax_rate=config.backtest.tax_rate,
    )

    # state_dim 확인
    first_df = next(iter(train_dict.values()))
    state = env.reset(first_df)
    state_dim = len(state)

    agent = RLTimingModel(
        state_dim=state_dim,
        hidden_dim=rl_config.hidden_dim,
        learning_rate=rl_config.learning_rate,
        gamma=rl_config.gamma,
        clip_epsilon=getattr(rl_config, "clip_epsilon", 0.2),
        entropy_coeff=rl_config.entropy_coeff,
        epochs_per_update=rl_config.epochs_per_update,
    )

    codes = list(train_dict.keys())
    best_sharpe = -999.0
    best_state_dict = None
    patience = 0
    max_patience = 10  # early stopping

    n_episodes = rl_config.episodes
    logger.info(f"PPO 학습 시작: {n_episodes} 에피소드 × {len(codes)} 종목")

    for episode in range(n_episodes):
        # ── 다중 종목 rollout 수집 ──
        all_states, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []
        episode_rewards = []

        for code in codes:
            df = train_dict[code]
            if len(df) < 60:
                continue

            try:
                rollout = agent.collect_episode(env, df)
                all_states.append(rollout["states"])
                all_actions.append(rollout["actions"])
                all_log_probs.append(rollout["log_probs"])
                all_advantages.append(rollout["advantages"])
                all_returns.append(rollout["returns"])
                episode_rewards.append(rollout["total_reward"])
            except Exception as e:
                logger.debug(f"에피소드 실패 ({code}): {e}")
                continue

        if not episode_rewards:
            continue

        # ── 배치 PPO 업데이트 ──
        batch_states = torch.FloatTensor(
            np.concatenate(all_states),
        ).to(agent.device)
        batch_actions = torch.LongTensor(
            np.concatenate(all_actions),
        ).to(agent.device)
        batch_log_probs = torch.FloatTensor(
            np.concatenate(all_log_probs),
        ).to(agent.device)
        batch_advantages = torch.FloatTensor(
            np.concatenate(all_advantages),
        ).to(agent.device)
        batch_returns = torch.FloatTensor(
            np.concatenate(all_returns),
        ).to(agent.device)

        progress = episode / max(n_episodes - 1, 1)  # 0.0 → 1.0
        stats = agent.ppo_update(
            batch_states, batch_actions, batch_log_probs,
            batch_advantages, batch_returns,
            progress=progress,
        )

        avg_reward = np.mean(episode_rewards)

        # ── 주기적 검증 ──
        if (episode + 1) % 5 == 0 or episode == n_episodes - 1:
            val_metrics = evaluate_rl_agent(
                agent, val_dict,
                commission_rate=config.backtest.commission_rate,
                tax_rate=config.backtest.tax_rate,
            )
            val_sharpe = val_metrics["sharpe"]

            logger.info(
                f"Episode {episode+1}/{n_episodes}: "
                f"reward={avg_reward:.4f}, "
                f"p_loss={stats['policy_loss']:.4f}, "
                f"v_loss={stats['value_loss']:.4f}, "
                f"entropy={stats['entropy']:.3f}, "
                f"val_sharpe={val_sharpe:.3f}, "
                f"val_return={val_metrics['total_return']:.2f}%, "
                f"trades={val_metrics['avg_trades']:.1f}"
            )

            if val_sharpe > best_sharpe:
                best_sharpe = val_sharpe
                best_state_dict = {
                    k: v.clone() for k, v in agent.actor.state_dict().items()
                }
                patience = 0
                logger.info(f"  → Best Sharpe: {best_sharpe:.3f}")
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info(f"  → Early stopping (patience={max_patience})")
                    break

    # 최적 모델 복원 및 저장
    if best_state_dict is not None:
        agent.actor.load_state_dict(best_state_dict)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(save_path)

    # 최종 검증
    final_metrics = evaluate_rl_agent(
        agent, val_dict,
        commission_rate=config.backtest.commission_rate,
        tax_rate=config.backtest.tax_rate,
    )

    logger.info(
        f"PPO 학습 완료: sharpe={final_metrics['sharpe']:.3f}, "
        f"return={final_metrics['total_return']:.2f}%, "
        f"win_rate={final_metrics['win_rate']:.1f}%, "
        f"avg_trades={final_metrics['avg_trades']:.1f}"
    )

    return {
        "accuracy": final_metrics["win_rate"] / 100,
        "sharpe": final_metrics["sharpe"],
        "total_return": final_metrics["total_return"],
        "max_drawdown": final_metrics["max_drawdown"],
        "n_episodes": final_metrics["n_episodes"],
        "win_rate": final_metrics["win_rate"],
        "n_samples": sum(len(df) for df in train_dict.values()),
    }
