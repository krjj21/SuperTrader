"""
종목풀 구성 및 리밸런싱 모듈
- 복합 점수 기반 상위 N종목 선정
- 리밸런싱 이력 관리
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from src.config import get_config
from src.data.market_data import get_universe
from src.data.factor_data import get_sector_info, get_market_cap
from src.factors.calculator import compute_cross_sectional_factors
from src.factors.neutralizer import neutralize_factor_matrix
from src.factors.validity import validate_all_factors, get_valid_factors
from src.factors.composite import compute_composite_score


@dataclass
class StockPool:
    """종목풀"""
    date: str
    codes: list[str]
    scores: dict[str, float] = field(default_factory=dict)
    entered: list[str] = field(default_factory=list)   # 신규 편입
    exited: list[str] = field(default_factory=list)     # 퇴출


def build_stock_pool(
    date: str,
    factor_history: dict[str, pd.DataFrame] | None = None,
    return_history: dict[str, pd.Series] | None = None,
    factor_report: pd.DataFrame | None = None,
    previous_pool: StockPool | None = None,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
    return_factors: bool = False,
) -> StockPool | tuple[StockPool, pd.DataFrame]:
    """주어진 날짜의 종목풀을 구성합니다.

    Args:
        date: 기준일 (YYYYMMDD)
        factor_history: 팩터 히스토리 (유효성 검증용, None이면 스킵)
        return_history: 수익률 히스토리 (유효성 검증용)
        factor_report: 사전 계산된 팩터 리포트
        previous_pool: 이전 종목풀 (진입/퇴출 추적용)
        ohlcv_dict: 사전 로드된 OHLCV 데이터 (백테스트용, None이면 자동 로드)
    """
    config = get_config()

    # 1. 유니버스 구성
    universe = get_universe(date)
    if universe.empty:
        logger.warning(f"유니버스가 비어 있습니다: {date}")
        return StockPool(date=date, codes=[])

    codes = universe["code"].tolist()

    # 2. 크로스섹션 팩터 계산
    factor_df = compute_cross_sectional_factors(codes, date, ohlcv_dict=ohlcv_dict)
    if factor_df.empty:
        return StockPool(date=date, codes=[])

    # 3. 바이어스 보정
    industry = None
    log_mcap = None

    if config.factors.neutralize_industry:
        sector_df = get_sector_info(date)
        if not sector_df.empty:
            industry = sector_df.set_index("code")["sector"]

    if config.factors.neutralize_market_cap and "log_market_cap" in factor_df.columns:
        log_mcap = factor_df["log_market_cap"]

    factor_df = neutralize_factor_matrix(
        factor_df,
        industry=industry,
        log_market_cap=log_mcap,
        do_industry=config.factors.neutralize_industry,
        do_market_cap=config.factors.neutralize_market_cap,
    )

    # 4. 유효 팩터 결정
    if factor_report is not None:
        valid_factors = get_valid_factors(factor_report)
    elif factor_history is not None and return_history is not None:
        report = validate_all_factors(
            factor_history, return_history,
            min_ir=config.factors.min_ir,
        )
        valid_factors = get_valid_factors(report)
    else:
        # 팩터 리포트 없으면 모든 팩터 사용
        valid_factors = factor_df.columns.tolist()

    # 5. 복합 점수 계산
    composite = compute_composite_score(
        factor_df, valid_factors, factor_report,
        method=config.factors.composite_method,
    )

    # 6. 상위 N종목 선정
    top_n = config.factors.top_n
    top_codes = composite.nlargest(top_n).index.tolist()
    top_scores = composite.loc[top_codes].to_dict()

    # 7. 진입/퇴출 추적
    prev_codes = set(previous_pool.codes) if previous_pool else set()
    current_codes = set(top_codes)
    entered = list(current_codes - prev_codes)
    exited = list(prev_codes - current_codes)

    pool = StockPool(
        date=date,
        codes=top_codes,
        scores=top_scores,
        entered=entered,
        exited=exited,
    )

    logger.info(
        f"종목풀 구성: {len(top_codes)}종목 "
        f"(신규 {len(entered)}, 퇴출 {len(exited)}) "
        f"date={date}"
    )
    if return_factors:
        return pool, factor_df
    return pool
