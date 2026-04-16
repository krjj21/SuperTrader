"""
PPO 학습 파이프라인
- 다중 종목 에피소드 학습 (rollout 배치)
- Walk-forward 검증
- Sharpe ratio 기반 최적 모델 선택
- GPU 가속 + 멀티프로세싱 rollout 수집
"""
from __future__ import annotations

import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from loguru import logger
import torch

from src.config import get_config
from src.timing.rl_env import TradingEnv
from src.timing.rl_agent import RLTimingModel


def _collect_single_episode_worker(args: tuple) -> dict | None:
    """단일 종목 에피소드 수집 (워커 스레드용 — CPU에서 실행)."""
    code, df, cpu_agent, env, deterministic = args
    if len(df) < 60:
        return None
    try:
        result = cpu_agent.collect_episode(env, df, deterministic=deterministic)
        result["code"] = code
        return result
    except Exception:
        return None


def _collect_rollouts_parallel(
    agent: RLTimingModel,
    train_dict: dict[str, pd.DataFrame],
    codes: list[str],
    commission_rate: float,
    tax_rate: float,
    n_workers: int = 8,
) -> tuple[list, list]:
    """멀티스레딩으로 rollout을 병렬 수집합니다.

    RL 환경은 CPU 연산이므로 ThreadPool로 GIL 우회 가능
    (numpy/torch no_grad는 GIL 해제).
    """
    from concurrent.futures import ThreadPoolExecutor

    # CPU 복사본으로 추론 (GPU 전송 오버헤드 제거)
    cpu_state_dict = {k: v.cpu() for k, v in agent.actor.state_dict().items()}
    cpu_agent = RLTimingModel(
        state_dim=agent.state_dim,
        hidden_dim=agent.hidden_dim,
    )
    cpu_agent.device = torch.device("cpu")
    cpu_agent.actor = cpu_agent.actor.cpu()
    cpu_agent.actor.load_state_dict(cpu_state_dict)
    cpu_agent.actor.eval()

    rollouts = []
    rewards = []

    # 스레드풀에서 종목별 rollout 병렬 수집
    args_list = [
        (code, train_dict[code], cpu_agent,
         TradingEnv(commission_rate=commission_rate, tax_rate=tax_rate),
         False)
        for code in codes if len(train_dict.get(code, [])) >= 60
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = pool.map(_collect_single_episode_worker, args_list)

    for r in results:
        if r is not None:
            rollouts.append(r)
            rewards.append(r["total_reward"])

    return rollouts, rewards


def evaluate_rl_agent(
    agent: RLTimingModel,
    ohlcv_dict: dict[str, pd.DataFrame],
    commission_rate: float = 0.00015,
    tax_rate: float = 0.0023,
) -> dict:
    """RL 에이전트의 성능을 평가합니다 (결정적 정책, CPU 병렬)."""
    from concurrent.futures import ThreadPoolExecutor

    # CPU 복사본으로 평가
    cpu_state_dict = {k: v.cpu() for k, v in agent.actor.state_dict().items()}
    cpu_agent = RLTimingModel(state_dim=agent.state_dim, hidden_dim=agent.hidden_dim)
    cpu_agent.device = torch.device("cpu")
    cpu_agent.actor = cpu_agent.actor.cpu()
    cpu_agent.actor.load_state_dict(cpu_state_dict)
    cpu_agent.actor.eval()

    episode_returns = []
    episode_trades = []

    args_list = [
        (code, df, cpu_agent,
         TradingEnv(commission_rate=commission_rate, tax_rate=tax_rate),
         True)
        for code, df in ohlcv_dict.items() if len(df) >= 100
    ]

    import os
    n_workers = min(os.cpu_count() or 4, len(args_list), 8)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = pool.map(_collect_single_episode_worker, args_list)

    for r in results:
        if r is not None:
            episode_returns.append(r["portfolio_value"] - 1.0)
            episode_trades.append(r.get("n_trades", 0))

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

    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU 감지: {gpu_name} ({gpu_mem:.1f}GB VRAM)")

    logger.info(f"PPO 학습 데이터: {len(train_dict)}종목, val: {len(val_dict)}종목 [{device_name}]")

    # 환경 및 에이전트 생성
    env = TradingEnv(
        commission_rate=config.backtest.commission_rate,
        tax_rate=config.backtest.tax_rate,
    )

    # state_dim 확인
    first_df = next(iter(train_dict.values()))
    state = env.reset(first_df)
    state_dim = len(state)

    # GPU VRAM에 맞춘 미니배치 크기 자동 조정
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        auto_mini_batch = min(512, max(64, int(gpu_mem_gb * 64)))  # ~6GB → 384
    else:
        auto_mini_batch = 64

    agent = RLTimingModel(
        state_dim=state_dim,
        hidden_dim=rl_config.hidden_dim,
        learning_rate=rl_config.learning_rate,
        gamma=rl_config.gamma,
        clip_epsilon=getattr(rl_config, "clip_epsilon", 0.2),
        entropy_coeff=rl_config.entropy_coeff,
        epochs_per_update=rl_config.epochs_per_update,
        mini_batch_size=auto_mini_batch,
    )
    logger.info(f"미니배치 크기: {auto_mini_batch} (자동 설정)")

    codes = list(train_dict.keys())
    best_sharpe = -999.0
    best_state_dict = None
    patience = 0
    max_patience = 10  # early stopping

    n_episodes = rl_config.episodes
    logger.info(f"PPO 학습 시작: {n_episodes} 에피소드 × {len(codes)} 종목")
    train_start_time = time.time()

    # 병렬 rollout 워커 수 결정
    import os
    n_workers = min(os.cpu_count() or 4, len(codes), 8)
    logger.info(f"Rollout 병렬 수집: {n_workers} workers (CPU), PPO 업데이트: {device_name}")

    for episode in range(n_episodes):
        ep_start = time.time()

        # ── 다중 종목 rollout 병렬 수집 (CPU) ──
        rollouts, episode_rewards = _collect_rollouts_parallel(
            agent, train_dict, codes,
            commission_rate=config.backtest.commission_rate,
            tax_rate=config.backtest.tax_rate,
            n_workers=n_workers,
        )

        if not episode_rewards:
            continue

        all_states = [r["states"] for r in rollouts]
        all_actions = [r["actions"] for r in rollouts]
        all_log_probs = [r["log_probs"] for r in rollouts]
        all_advantages = [r["advantages"] for r in rollouts]
        all_returns = [r["returns"] for r in rollouts]

        rollout_time = time.time() - ep_start

        # ── 배치 PPO 업데이트 ──
        update_start = time.time()
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
        update_time = time.time() - update_start

        avg_reward = np.mean(episode_rewards)
        batch_size = len(batch_states)

        # ── 주기적 검증 ──
        if (episode + 1) % 5 == 0 or episode == n_episodes - 1:
            val_metrics = evaluate_rl_agent(
                agent, val_dict,
                commission_rate=config.backtest.commission_rate,
                tax_rate=config.backtest.tax_rate,
            )
            val_sharpe = val_metrics["sharpe"]

            elapsed = time.time() - train_start_time
            eta = elapsed / (episode + 1) * (n_episodes - episode - 1)
            logger.info(
                f"Episode {episode+1}/{n_episodes}: "
                f"reward={avg_reward:.4f}, "
                f"p_loss={stats['policy_loss']:.4f}, "
                f"v_loss={stats['value_loss']:.4f}, "
                f"entropy={stats['entropy']:.3f}, "
                f"val_sharpe={val_sharpe:.3f}, "
                f"val_return={val_metrics['total_return']:.2f}%, "
                f"trades={val_metrics['avg_trades']:.1f}, "
                f"batch={batch_size}, "
                f"rollout={rollout_time:.1f}s, update={update_time:.1f}s, "
                f"ETA={eta/60:.0f}min"
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

    total_train_time = time.time() - train_start_time
    logger.info(
        f"PPO 학습 완료 ({total_train_time/60:.1f}분): "
        f"sharpe={final_metrics['sharpe']:.3f}, "
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
