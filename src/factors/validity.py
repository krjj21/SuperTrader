"""
팩터 유효성 검증 모듈
- IC (Information Coefficient): 스피어만 상관
- IR (Information Ratio): mean(IC) / std(IC)
- t검정: IC 유의성
- 팩터 감쇠: IC의 시계열 안정성
- 팩터 회전율
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


def calculate_ic(
    factor_values: pd.Series,
    forward_returns: pd.Series,
) -> float:
    """단일 기간의 IC (Information Coefficient)를 계산합니다.

    Spearman rank correlation between factor values and forward returns.
    """
    common = factor_values.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 10:
        return np.nan

    corr, _ = stats.spearmanr(
        factor_values.loc[common].values,
        forward_returns.loc[common].values,
    )
    return corr


def calculate_ic_series(
    factor_history: dict[str, pd.DataFrame],
    return_history: dict[str, pd.Series],
    factor_name: str,
) -> pd.Series:
    """팩터의 IC 시계열을 계산합니다.

    Args:
        factor_history: {date: factor_matrix} (code index, factor columns)
        return_history: {date: forward_return} (code index)
        factor_name: 팩터 이름

    Returns:
        IC 시계열 (date index)
    """
    ic_values = {}
    for date in factor_history:
        if date not in return_history:
            continue
        factor_df = factor_history[date]
        if factor_name not in factor_df.columns:
            continue

        ic = calculate_ic(factor_df[factor_name], return_history[date])
        if not np.isnan(ic):
            ic_values[date] = ic

    return pd.Series(ic_values, name=factor_name)


def calculate_factor_metrics(ic_series: pd.Series) -> dict:
    """IC 시계열에서 팩터 지표를 계산합니다.

    Returns:
        {mean_ic, ic_std, ir, t_stat, p_value, ic_positive_ratio}
    """
    ic = ic_series.dropna()
    n = len(ic)

    if n < 3:
        return {
            "mean_ic": np.nan, "ic_std": np.nan, "ir": np.nan,
            "t_stat": np.nan, "p_value": np.nan,
            "ic_positive_ratio": np.nan, "n_periods": n,
        }

    mean_ic = ic.mean()
    ic_std = ic.std()
    ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0

    # t검정 (mean IC != 0)
    t_stat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 1e-10 else 0.0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

    ic_positive_ratio = (ic > 0).mean()

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ir": ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "ic_positive_ratio": ic_positive_ratio,
        "n_periods": n,
    }


def calculate_factor_turnover(
    factor_history: dict[str, pd.DataFrame],
    factor_name: str,
    top_n: int = 30,
) -> float:
    """팩터의 평균 회전율을 계산합니다.

    회전율 = 상위 N종목 중 이전 기간 대비 변경된 종목 비율
    """
    dates = sorted(factor_history.keys())
    turnovers = []

    prev_top = set()
    for date in dates:
        df = factor_history[date]
        if factor_name not in df.columns:
            continue

        current_top = set(
            df[factor_name].dropna().nlargest(top_n).index.tolist()
        )

        if prev_top:
            changed = len(current_top - prev_top)
            turnover = changed / top_n if top_n > 0 else 0.0
            turnovers.append(turnover)

        prev_top = current_top

    return np.mean(turnovers) if turnovers else np.nan


def validate_all_factors(
    factor_history: dict[str, pd.DataFrame],
    return_history: dict[str, pd.Series],
    min_ir: float = 0.3,
    min_t_stat: float = 2.0,
    top_n: int = 30,
) -> pd.DataFrame:
    """모든 팩터의 유효성을 검증합니다.

    Returns:
        팩터 지표 DataFrame (factor index, metric columns)
    """
    # 첫 번째 기간의 팩터 이름 추출
    sample_df = next(iter(factor_history.values()))
    factor_names = sample_df.columns.tolist()

    results = []
    for name in factor_names:
        ic_series = calculate_ic_series(factor_history, return_history, name)
        metrics = calculate_factor_metrics(ic_series)
        metrics["factor_name"] = name
        metrics["turnover"] = calculate_factor_turnover(factor_history, name, top_n)
        metrics["is_valid"] = (
            abs(metrics["ir"]) >= min_ir
            and abs(metrics["t_stat"]) >= min_t_stat
        )
        results.append(metrics)

    report = pd.DataFrame(results).set_index("factor_name")
    report = report.sort_values("ir", ascending=False, key=abs)

    valid_count = report["is_valid"].sum()
    logger.info(
        f"팩터 유효성 검증 완료: {valid_count}/{len(report)} 통과 "
        f"(IR>={min_ir}, |t|>={min_t_stat})"
    )
    return report


def get_valid_factors(
    report: pd.DataFrame,
) -> list[str]:
    """유효한 팩터 이름 리스트를 반환합니다."""
    return report[report["is_valid"]].index.tolist()
