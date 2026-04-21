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
    """단일 종목 에피소드 수집 (워커 스레드용 — CPU에서 실행).

    args = (code, df, cpu_agent, env, deterministic, sentiment_series)
    sentiment_series: None 이면 SAPPO 비활성.
    """
    if len(args) == 6:
        code, df, cpu_agent, env, deterministic, sentiment_series = args
    else:
        code, df, cpu_agent, env, deterministic = args
        sentiment_series = None
    if len(df) < 60:
        return None
    try:
        # env.reset 에 sentiment_series 를 주입하기 위한 hook
        # collect_episode 가 env.reset(df) 만 호출하면 sentiment 주입 불가 →
        # 여기서 env.reset 을 직접 호출한 뒤 collect_episode 는 이미 초기화된 env 로 돌림
        if sentiment_series is not None:
            env.reset(df, sentiment_series=sentiment_series)
            result = cpu_agent.collect_episode(env, df, deterministic=deterministic, reset_env=False)
        else:
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
    sentiment_lambda: float = 0.0,
    sentiment_map: dict[str, dict[str, float]] | None = None,
) -> tuple[list, list]:
    """멀티스레딩으로 rollout을 병렬 수집합니다.

    Args:
        sentiment_lambda: SAPPO 가중치 (0 이면 baseline PPO)
        sentiment_map: {code: {YYYYMMDD: score}} — 종목별 sentiment 시계열
    """
    from concurrent.futures import ThreadPoolExecutor

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

    sentiment_map = sentiment_map or {}
    args_list = [
        (code, train_dict[code], cpu_agent,
         TradingEnv(
             commission_rate=commission_rate, tax_rate=tax_rate,
             sentiment_lambda=sentiment_lambda,
         ),
         False,
         sentiment_map.get(code) if sentiment_lambda > 0 else None)
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
    sentiment_lambda: float = 0.0,
    sentiment_map: dict[str, dict[str, float]] | None = None,
    run_name: str | None = None,
    sentiment_source: str = "off",
) -> dict:
    """PPO/SAPPO 로 RL 타이밍 모델을 학습합니다.

    Args:
        sentiment_lambda: SAPPO 가중치. 0 이면 baseline PPO.
        sentiment_map: {code: {YYYYMMDD: score}} — sentiment_lambda > 0 일 때 사용.
        run_name: DB 기록용 식별자.
        sentiment_source: "off" / "news" / "mock" / "xgb_proxy" 등 라벨링용.
    """
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
            sentiment_lambda=sentiment_lambda,
            sentiment_map=sentiment_map,
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
        f"avg_trades={final_metrics['avg_trades']:.1f}, "
        f"sentiment_lambda={sentiment_lambda}"
    )

    # SAPPO 학습 런 DB 기록 (sentiment_lambda 와 무관하게 매 학습 기록 → baseline 비교 가능)
    try:
        from src.db.sappo_models import init_sappo_db, save_training_run
        init_sappo_db("data/trading.db")

        def _date_range(dfs: dict[str, pd.DataFrame]) -> tuple[str, str]:
            starts, ends = [], []
            for d in dfs.values():
                if "date" in d.columns and not d.empty:
                    if pd.api.types.is_datetime64_any_dtype(d["date"]):
                        starts.append(d["date"].iloc[0].strftime("%Y%m%d"))
                        ends.append(d["date"].iloc[-1].strftime("%Y%m%d"))
                    else:
                        starts.append(str(d["date"].iloc[0]).replace("-", ""))
                        ends.append(str(d["date"].iloc[-1]).replace("-", ""))
            return (min(starts) if starts else "", max(ends) if ends else "")

        train_s, train_e = _date_range(train_dict)
        val_s, val_e = _date_range(val_dict)

        auto_name = run_name or (
            f"{'sappo' if sentiment_lambda > 0 else 'baseline'}"
            f"_lambda{sentiment_lambda}_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        save_training_run(
            run_name=auto_name,
            lambda_value=sentiment_lambda,
            sentiment_source=sentiment_source,
            train_start=train_s, train_end=train_e,
            val_start=val_s, val_end=val_e,
            n_episodes=final_metrics.get("n_episodes", 0),
            val_sharpe=float(final_metrics.get("sharpe", 0.0)),
            val_return=float(final_metrics.get("total_return", 0.0)),
            val_mdd=float(final_metrics.get("max_drawdown", 0.0)),
            val_win_rate=float(final_metrics.get("win_rate", 0.0)),
            val_avg_trades=float(final_metrics.get("avg_trades", 0.0)),
            model_path=str(save_path),
            notes=f"train_time={total_train_time/60:.1f}min, best_sharpe={best_sharpe:.3f}",
        )
        logger.info(f"학습 런 기록: {auto_name}")
    except Exception as e:
        logger.warning(f"학습 런 DB 저장 실패: {e}")

    return {
        "accuracy": final_metrics["win_rate"] / 100,
        "sharpe": final_metrics["sharpe"],
        "total_return": final_metrics["total_return"],
        "max_drawdown": final_metrics["max_drawdown"],
        "n_episodes": final_metrics["n_episodes"],
        "win_rate": final_metrics["win_rate"],
        "n_samples": sum(len(df) for df in train_dict.values()),
        "sentiment_lambda": sentiment_lambda,
    }
