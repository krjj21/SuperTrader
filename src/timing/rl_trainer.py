"""
PPO 학습 파이프라인
- 다중 종목 에피소드 학습 (rollout 배치)
- Walk-forward 검증
- Sharpe ratio 기반 최적 모델 선택
- GPU 가속 + 멀티프로세싱 rollout 수집
"""
from __future__ import annotations

import math
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

    # 최소 30일이면 PPO env 가 동작 (train 필터는 100일이지만 val split 후 짧아질 수 있음)
    args_list = [
        (code, df, cpu_agent,
         TradingEnv(commission_rate=commission_rate, tax_rate=tax_rate),
         True)
        for code, df in ohlcv_dict.items() if len(df) >= 30
    ]

    if not args_list:
        logger.warning(
            f"evaluate_rl_agent: 평가 가능한 종목 0개 (입력 {len(ohlcv_dict)}종목, 모두 <30일). "
            f"평가를 건너뛰고 sharpe=-999 반환."
        )
        agent.actor.train()
        return {
            "sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
            "n_episodes": 0, "win_rate": 0.0, "avg_trades": 0.0,
        }

    import os
    n_workers = max(min(os.cpu_count() or 4, len(args_list), 8), 1)
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


def evaluate_rl_portfolio(
    agent: RLTimingModel,
    ohlcv_dict: dict[str, pd.DataFrame],
    top_n: int = 30,
    rebalance_freq_days: int = 22,
    commission_rate: float = 0.00015,
    tax_rate: float = 0.0023,
    initial_capital: int = 100_000_000,
) -> dict:
    """RL 에이전트를 *portfolio simulation* 환경에서 평가한다.

    evaluate_rl_agent() 가 단일종목 episode 평균 Sharpe 를 계산하는 반면,
    이 함수는 30종목 동시보유 + 월간 리밸런싱 + commission/tax 를 적용한
    portfolio Sharpe 를 계산한다 (production backtest 와 동일 환경).

    종목풀 선정: 각 리밸런싱 시점에서 직전 60일 누적 수익률 상위 top_n.
    (간이 모멘텀 — 학습 평가용. 라이브의 220-팩터 IC 가중과는 별개.)

    Returns:
        {sharpe, total_return, max_drawdown, win_rate, total_trades,
         n_rebalances, env_type='portfolio'}
    """
    # PortfolioBacktestEngine + FactorRLStrategy 재사용
    try:
        from backtest.portfolio_engine import PortfolioBacktestEngine
        from src.strategy.factor_rl import FactorRLStrategy
        import tempfile
        import os
    except Exception as e:
        logger.warning(f"evaluate_rl_portfolio 의존성 로드 실패: {e}")
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio", "error": str(e)}

    # 1) 모든 종목·날짜 union 정렬
    all_dates: set[str] = set()
    for code, df in ohlcv_dict.items():
        if "date" in df.columns:
            all_dates.update(df["date"].astype(str))
    sorted_dates = sorted(all_dates)
    if len(sorted_dates) < 60 + rebalance_freq_days:
        logger.warning(
            f"evaluate_rl_portfolio: 데이터 부족 ({len(sorted_dates)}일 < "
            f"{60 + rebalance_freq_days}일 필요)"
        )
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio"}

    # 2) 리밸런싱 날짜: 60일 워밍업 후 rebalance_freq_days 간격
    warmup = 60
    rebalance_dates: list[str] = []
    for i in range(warmup, len(sorted_dates), rebalance_freq_days):
        rebalance_dates.append(sorted_dates[i])
    if not rebalance_dates:
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio"}

    # 3) 각 리밸런싱일에서 직전 60일 누적 수익률 top_n 선정
    pool_history: dict[str, list[str]] = {}
    for rd in rebalance_dates:
        scores: dict[str, float] = {}
        for code, df in ohlcv_dict.items():
            if "date" not in df.columns or len(df) < warmup:
                continue
            df_idx = df[df["date"].astype(str) <= rd].tail(warmup)
            if len(df_idx) < warmup:
                continue
            close = df_idx["close"].astype(float).values
            if close[0] <= 0:
                continue
            scores[code] = float(close[-1] / close[0] - 1.0)
        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            pool_history[rd] = [c for c, _ in top]

    if not pool_history:
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio"}

    # 4) 임시 모델 파일로 저장 → FactorRLStrategy 가 load
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    try:
        agent.save(tmp.name)
        strategy = FactorRLStrategy(model_path=tmp.name)
    except Exception as e:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        logger.warning(f"evaluate_rl_portfolio: FactorRLStrategy 로드 실패: {e}")
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio", "error": str(e)}

    try:
        engine = PortfolioBacktestEngine(
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            tax_rate=tax_rate,
        )
        result = engine.run(strategy, ohlcv_dict, pool_history, rebalance_dates)
        m = result.get("metrics", {}) if isinstance(result, dict) else {}
        return {
            "sharpe": float(m.get("sharpe_ratio", -999.0)),
            "total_return": float(m.get("total_return", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
            "win_rate": float(m.get("win_rate", 0.0)),
            "total_trades": int(m.get("total_trades", 0)),
            "n_rebalances": len(rebalance_dates),
            "env_type": "portfolio",
        }
    except Exception as e:
        logger.warning(f"evaluate_rl_portfolio 실행 중 예외: {e}")
        return {"sharpe": -999.0, "total_return": 0.0, "max_drawdown": 0.0,
                "win_rate": 0.0, "total_trades": 0, "n_rebalances": 0,
                "env_type": "portfolio", "error": str(e)}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def train_rl_model(
    ohlcv_dict: dict[str, pd.DataFrame],
    save_path: str = "models/rl_timing.pt",
    val_ratio: float = 0.2,
    sentiment_lambda: float = 0.0,
    sentiment_map: dict[str, dict[str, float]] | None = None,
    run_name: str | None = None,
    sentiment_source: str = "off",
    seed: int = 42,
) -> dict:
    """PPO/SAPPO 로 RL 타이밍 모델을 학습합니다.

    Args:
        sentiment_lambda: SAPPO 가중치. 0 이면 baseline PPO.
        sentiment_map: {code: {YYYYMMDD: score}} — sentiment_lambda > 0 일 때 사용.
        run_name: DB 기록용 식별자.
        sentiment_source: "off" / "news" / "mock" / "xgb_proxy" 등 라벨링용.
        seed: torch/numpy seed. 시드 변동 ±0.3~0.5 Sharpe 흔함 — ensemble 위해 노출.
    """
    # 시드 고정 (PPO 정책 초기화 + rollout 샘플링 결정성)
    import random
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    best_score = -999.0          # portfolio composite score (sharpe + 0.05·win + 0.01·return - 0.02·|mdd|)
    best_state_dict = None
    patience = 0
    max_patience = 6  # 6 portfolio_eval (= 60 episode) 무개선 시 종료
    # NOTE: 2026-04-29 — single-stock val_sharpe 의 trades_adj_score 가 portfolio 환경에서
    # 무의미함이 v6 ensemble retrain 으로 입증 (val 8.6 / portfolio -0.07).
    # best 기준 자체를 portfolio 환경 metric 으로 변경.

    # Portfolio eval 주기 — 25 → 10 으로 강화 (best 신호 빈도 ↑, 학습 시간 +20min/seed)
    portfolio_eval_every = 10
    best_portfolio_sharpe = -999.0   # 참고용 (composite 의 sharpe 항만 별도 기록)

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
            avg_trades = val_metrics.get("avg_trades", 0.0) or 0.0

            # Trades-adjusted score: 매매 빈도가 폭증하는 정책에 페널티
            # score = sharpe / log(1 + trades_per_episode)
            # → trades 5회 → 1.79 분모 / trades 25 → 3.26 / trades 100 → 4.61
            # 같은 sharpe 면 trades 적은 정책이 우선
            trades_adj_score = float(val_sharpe) / float(math.log(1.0 + max(avg_trades, 0.0)) + 1.0)

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
                f"trades={avg_trades:.1f}, "
                f"adj_score={trades_adj_score:.3f}, "
                f"batch={batch_size}, "
                f"rollout={rollout_time:.1f}s, update={update_time:.1f}s, "
                f"ETA={eta/60:.0f}min"
            )

            # ── Portfolio eval (best 선택의 *유일한* 기준) — 매 portfolio_eval_every episode ──
            if portfolio_eval_every and (episode + 1) % portfolio_eval_every == 0:
                try:
                    pf_metrics = evaluate_rl_portfolio(
                        agent, val_dict,
                        commission_rate=config.backtest.commission_rate,
                        tax_rate=config.backtest.tax_rate,
                    )
                    pf_sharpe = float(pf_metrics.get("sharpe", -999.0))
                    pf_mdd = float(pf_metrics.get("max_drawdown", 0.0))
                    pf_win = float(pf_metrics.get("win_rate", 0.0))
                    pf_return = float(pf_metrics.get("total_return", 0.0))
                    pf_trades = int(pf_metrics.get("total_trades", 0))

                    # Composite score — 다중 metric 가중합 (사용자 요구)
                    # 가중치 직관:
                    #   sharpe (주된 위험조정 수익 척도, 단위 1.0)
                    #   + 0.05 × win_rate%  (50% → 2.5 / 65% → 3.25)
                    #   + 0.01 × return%   (30% → 0.30 / 50% → 0.50)
                    #   - 0.02 × |mdd%|    (10% → 0.20 / 20% → 0.40)
                    composite = (
                        pf_sharpe
                        + 0.05 * pf_win
                        + 0.01 * pf_return
                        - 0.02 * abs(pf_mdd)
                    )
                    if pf_sharpe > best_portfolio_sharpe:
                        best_portfolio_sharpe = pf_sharpe

                    logger.info(
                        f"  [Portfolio] sharpe={pf_sharpe:+.3f}, mdd={pf_mdd:.2f}%, "
                        f"win={pf_win:.1f}%, return={pf_return:+.2f}%, "
                        f"trades={pf_trades}, composite={composite:+.3f}, "
                        f"best_composite={best_score:+.3f}"
                    )

                    # ── Best 갱신: composite 가 가장 높을 때만 ──
                    if composite > best_score:
                        best_score = composite
                        best_state_dict = {
                            k: v.clone() for k, v in agent.actor.state_dict().items()
                        }
                        patience = 0
                        logger.info(
                            f"  → Best (composite={best_score:+.3f}, "
                            f"pf_sharpe={pf_sharpe:+.3f}, win={pf_win:.1f}%, "
                            f"return={pf_return:+.2f}%, mdd={pf_mdd:.2f}%)"
                        )
                    else:
                        patience += 1
                        if patience >= max_patience:
                            logger.info(
                                f"  → Early stopping (patience={max_patience} portfolio_evals, "
                                f"best_composite={best_score:+.3f})"
                            )
                            break
                except Exception as e:
                    logger.warning(f"  [Portfolio] eval 실패: {e}")
                    # eval 실패 시 patience 증가 안 함 (transient error 보호)

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


def train_rl_ensemble(
    ohlcv_dict: dict[str, pd.DataFrame],
    save_path: str = "models/rl_timing.pt",
    val_ratio: float = 0.2,
    sentiment_lambda: float = 0.0,
    sentiment_map: dict[str, dict[str, float]] | None = None,
    sentiment_source: str = "off",
    seeds: list[int] | None = None,
) -> dict:
    """N개 시드로 PPO 모델을 학습 후 *median Sharpe* 모델을 채택한다.

    배경 (2026-04-28 코드 리뷰): financial RL 에서 시드 간 Sharpe 분산 ±0.3~0.5 흔함.
    Triple gate 는 잘못된 모델 *교체* 를 막지만 *시드 변동* 자체는 줄이지 않는다.
    Ensemble = 시드 N개 학습 → 정렬 → 중간값 모델 채택. GPU 시간 N배 비용으로
    회귀 위험 큰 폭 감소.

    Args:
        seeds: 학습할 시드 리스트. None 또는 단일 원소 → train_rl_model 단독 호출.
    """
    if seeds is None or len(seeds) <= 1:
        single_seed = (seeds[0] if seeds else 42)
        return train_rl_model(
            ohlcv_dict, save_path=save_path, val_ratio=val_ratio,
            sentiment_lambda=sentiment_lambda, sentiment_map=sentiment_map,
            sentiment_source=sentiment_source, seed=single_seed,
        )

    import tempfile, os
    candidates: list[tuple[int, str, dict]] = []  # (seed, tmp_path, result)
    for i, seed in enumerate(seeds, 1):
        logger.info(f"[Ensemble {i}/{len(seeds)}] seed={seed} 학습 시작")
        tmp = tempfile.NamedTemporaryFile(
            suffix=f".seed{seed}.pt", delete=False,
            dir=str(Path(save_path).parent),
        )
        tmp.close()
        try:
            result = train_rl_model(
                ohlcv_dict, save_path=tmp.name, val_ratio=val_ratio,
                sentiment_lambda=sentiment_lambda, sentiment_map=sentiment_map,
                sentiment_source=sentiment_source, seed=seed,
            )
            if "error" in result:
                logger.warning(f"[Ensemble seed={seed}] 학습 실패: {result['error']}")
                try: os.unlink(tmp.name)
                except Exception: pass
                continue
            candidates.append((seed, tmp.name, result))
            logger.info(
                f"[Ensemble seed={seed}] sharpe={result.get('sharpe',0):.3f}, "
                f"return={result.get('total_return',0):.2f}%"
            )
        except Exception as e:
            logger.error(f"[Ensemble seed={seed}] 학습 예외: {e}")
            try: os.unlink(tmp.name)
            except Exception: pass

    if not candidates:
        return {"error": "all_seeds_failed"}

    # Sharpe 기준 정렬 후 중간값 선택 (홀수 → 정확한 median, 짝수 → lower median)
    candidates.sort(key=lambda x: x[2].get("sharpe", -999.0))
    median_idx = len(candidates) // 2
    if len(candidates) > 1 and len(candidates) % 2 == 0:
        median_idx -= 1  # lower median (보수적)
    chosen_seed, chosen_tmp, chosen_result = candidates[median_idx]

    sharpes = [r.get("sharpe", 0) for _, _, r in candidates]
    logger.info(
        f"[Ensemble] {len(candidates)}/{len(seeds)} seeds 성공, "
        f"Sharpe 분포 = {[f'{s:.3f}' for s in sharpes]} → median seed={chosen_seed} 채택"
    )

    # 채택된 모델을 save_path 로 이동
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(save_path).exists():
        Path(save_path).unlink()
    Path(chosen_tmp).rename(save_path)

    # 나머지 시드 임시 파일 정리
    for s, tmp_p, _ in candidates:
        if s != chosen_seed:
            try: Path(tmp_p).unlink(missing_ok=True)
            except Exception: pass

    chosen_result["ensemble_seeds"] = [s for s, _, _ in candidates]
    chosen_result["ensemble_sharpes"] = sharpes
    chosen_result["chosen_seed"] = chosen_seed
    return chosen_result
