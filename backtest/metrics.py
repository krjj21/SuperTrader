"""
백테스트 성과 지표 계산
- Sharpe, Sortino, Calmar, MDD, 승률 등
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_metrics(
    equity_curve: pd.Series,
    daily_returns: pd.Series,
    trades: list[dict],
    initial_capital: int,
) -> dict:
    """백테스트 결과에서 성과 지표를 계산합니다.

    Args:
        equity_curve: 일별 포트폴리오 가치 (date index)
        daily_returns: 일별 수익률
        trades: 라운드트립 거래 기록 리스트
        initial_capital: 초기 자본금

    Returns:
        성과 지표 딕셔너리
    """
    metrics: dict = {}

    # ── 기본 수익률 ──
    final_value = equity_curve.iloc[-1]
    metrics["initial_capital"] = initial_capital
    metrics["final_value"] = int(final_value)
    metrics["total_return"] = (final_value / initial_capital - 1.0) * 100

    # 연환산 수익률 (CAGR)
    n_days = len(equity_curve)
    n_years = n_days / 250  # 거래일 기준
    if n_years > 0 and final_value > 0:
        metrics["cagr"] = ((final_value / initial_capital) ** (1 / n_years) - 1.0) * 100
    else:
        metrics["cagr"] = 0.0

    # ── 리스크 지표 ──
    returns = daily_returns.dropna()

    # 샤프비율 (연환산, rf=3.5%)
    rf_daily = 0.035 / 250
    excess = returns - rf_daily
    if len(excess) > 1 and np.std(excess) > 0:
        metrics["sharpe_ratio"] = float(np.mean(excess) / np.std(excess) * np.sqrt(250))
    else:
        metrics["sharpe_ratio"] = 0.0

    # 소르티노비율
    downside = returns[returns < rf_daily] - rf_daily
    if len(downside) > 0 and np.std(downside) > 0:
        metrics["sortino_ratio"] = float(
            np.mean(excess) / np.std(downside) * np.sqrt(250)
        )
    else:
        metrics["sortino_ratio"] = 0.0

    # 최대 낙폭 (MDD)
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    metrics["max_drawdown"] = float(drawdown.min()) * 100  # 음수 퍼센트

    # MDD 지속 기간
    is_drawdown = drawdown < 0
    if is_drawdown.any():
        dd_groups = (~is_drawdown).cumsum()
        dd_durations = is_drawdown.groupby(dd_groups).sum()
        metrics["max_drawdown_duration"] = int(dd_durations.max())
    else:
        metrics["max_drawdown_duration"] = 0

    # Calmar 비율
    mdd_abs = abs(metrics["max_drawdown"])
    if mdd_abs > 0:
        metrics["calmar_ratio"] = metrics["cagr"] / mdd_abs
    else:
        metrics["calmar_ratio"] = 0.0

    # 연변동성
    if len(returns) > 1:
        metrics["annual_volatility"] = float(np.std(returns) * np.sqrt(250)) * 100
    else:
        metrics["annual_volatility"] = 0.0

    # ── 거래 지표 ──
    metrics["total_trades"] = len(trades)

    if trades:
        pnl_list = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p <= 0]

        metrics["win_rate"] = len(wins) / len(pnl_list) * 100
        metrics["avg_win"] = float(np.mean(wins)) if wins else 0.0
        metrics["avg_loss"] = float(np.mean(losses)) if losses else 0.0

        # Profit Factor
        total_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        total_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
        metrics["profit_factor"] = (
            total_profit / total_loss if total_loss > 0 else float("inf")
        )

        # 연속 승/패
        metrics["max_consecutive_wins"] = _max_consecutive(pnl_list, positive=True)
        metrics["max_consecutive_losses"] = _max_consecutive(pnl_list, positive=False)

        # 평균 보유 기간
        holding_days = [t.get("holding_days", 0) for t in trades]
        metrics["avg_holding_days"] = float(np.mean(holding_days)) if holding_days else 0.0
    else:
        metrics["win_rate"] = 0.0
        metrics["avg_win"] = 0.0
        metrics["avg_loss"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["max_consecutive_wins"] = 0
        metrics["max_consecutive_losses"] = 0
        metrics["avg_holding_days"] = 0.0

    # 시장 노출 비율
    if "exposure_days" in (trades[0] if trades else {}):
        total_exposure = sum(t.get("holding_days", 0) for t in trades)
        metrics["exposure_pct"] = total_exposure / n_days * 100 if n_days > 0 else 0.0
    else:
        metrics["exposure_pct"] = 0.0

    metrics["trading_days"] = n_days
    metrics["trading_years"] = round(n_years, 2)

    return metrics


def _max_consecutive(pnl_list: list[float], positive: bool) -> int:
    """최대 연속 수익/손실 횟수를 계산합니다."""
    max_count = 0
    current = 0
    for pnl in pnl_list:
        if (positive and pnl > 0) or (not positive and pnl <= 0):
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count
