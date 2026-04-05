"""
백테스트 리포트 및 시각화
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_equity_comparison(
    equity_curves: dict[str, pd.Series],
    title: str = "Strategy Comparison",
    save_path: str | None = None,
) -> None:
    """전략별 누적 수익률 차트를 그립니다."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # 상단: 누적 수익률
    ax1 = axes[0]
    for name, equity in equity_curves.items():
        normalized = equity / equity.iloc[0] * 100
        ax1.plot(normalized.index, normalized.values, label=name, linewidth=1.5)

    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel("Portfolio Value (Base=100)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 하단: 드로다운
    ax2 = axes[1]
    for name, equity in equity_curves.items():
        peak = equity.cummax()
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(drawdown.index, drawdown.values, alpha=0.3, label=name)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_comparison_table(comparison: pd.DataFrame) -> str:
    """성과 비교 테이블을 포맷팅합니다."""
    formatters = {
        "total_return": "{:.1f}%".format,
        "cagr": "{:.1f}%".format,
        "sharpe_ratio": "{:.2f}".format,
        "sortino_ratio": "{:.2f}".format,
        "max_drawdown": "{:.1f}%".format,
        "calmar_ratio": "{:.2f}".format,
        "win_rate": "{:.1f}%".format,
        "profit_factor": "{:.2f}".format,
        "total_trades": "{:.0f}".format,
        "avg_holding_days": "{:.1f}".format,
    }

    lines = ["\n" + "=" * 80]
    lines.append("전략 비교 결과")
    lines.append("=" * 80)

    for col in comparison.columns:
        fmt = formatters.get(col, "{:.2f}".format)
        lines.append(f"\n{col}:")
        for strategy in comparison.index:
            val = comparison.loc[strategy, col]
            lines.append(f"  {strategy:25s} {fmt(val)}")

    report = "\n".join(lines)
    print(report)
    return report
