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
    regime_label: str | None = None,
) -> StockPool | tuple[StockPool, pd.DataFrame]:
    """주어진 날짜의 종목풀을 구성합니다.

    Args:
        date: 기준일 (YYYYMMDD)
        factor_history: 팩터 히스토리 (유효성 검증용, None이면 스킵)
        return_history: 수익률 히스토리 (유효성 검증용)
        factor_report: 사전 계산된 팩터 리포트
        previous_pool: 이전 종목풀 (진입/퇴출 추적용)
        ohlcv_dict: 사전 로드된 OHLCV 데이터 (백테스트용, None이면 자동 로드)
        regime_label: regime 라벨 (예: "risk_on_trend"). 지정 시 카테고리별
            multiplier 가 IC 가중치 위에 곱해진다. config.regime.lambda_ <= 0
            이면 라벨이 와도 무시 (dark-launch).
    """
    config = get_config()

    # 1. 유니버스 구성 (PIT — ohlcv_dict 전달 시 D 시점 rolling-20 거래량 사용)
    universe = get_universe(date, ohlcv_dict=ohlcv_dict)
    if universe.empty:
        logger.warning(f"유니버스가 비어 있습니다: {date}")
        return StockPool(date=date, codes=[])

    codes = universe["code"].tolist()

    # 1-b. 투자자 매매 사전 필터 (foreign_filter_enabled=True 시)
    # sappo_investor_trading 에서 N영업일 누적 순매수 거래대금 상위 pct% 만 통과.
    # mode: foreign(외국인만) | foreign_organ(외국인+기관 합산)
    if getattr(config.factors, "foreign_filter_enabled", False):
        try:
            from src.db.sappo_models import (
                init_sappo_db,
                get_foreign_net_buy_cumulative,
                get_combined_net_buy_cumulative,
            )
            init_sappo_db(config.database.path)
            lookback = int(getattr(config.factors, "foreign_filter_lookback", 20))
            pct = float(getattr(config.factors, "foreign_filter_pct", 0.5))
            mode = str(getattr(config.factors, "investor_filter_mode", "foreign"))
            net_buy_fn = (
                get_combined_net_buy_cumulative if mode == "foreign_organ"
                else get_foreign_net_buy_cumulative
            )
            scores = []
            n_with_data = 0
            for code in codes:
                cum, n = net_buy_fn(code, date, lookback)
                # 데이터 부재 종목은 cum=0, n=0 → 중간 위치
                if n > 0:
                    n_with_data += 1
                scores.append((code, cum, n))
            if n_with_data == 0:
                logger.warning(
                    f"투자자 매매 데이터 0건 (date={date}, mode={mode}) — 필터 skip. "
                    f"scripts/fetch_foreign_buys.py --universe all 실행 필요"
                )
            else:
                scores.sort(key=lambda x: x[1], reverse=True)
                cutoff = max(1, int(len(scores) * pct))
                allowed = {code for code, _, _ in scores[:cutoff]}
                before = len(codes)
                codes = [c for c in codes if c in allowed]
                logger.info(
                    f"투자자 매매 필터({mode}): {before}→{len(codes)}종목 "
                    f"(상위 {pct*100:.0f}%, lookback={lookback}일, "
                    f"데이터 보유 {n_with_data}/{before})"
                )
                # universe도 함께 좁힘 (이후 단계에서 일관성)
                universe = universe[universe["code"].isin(codes)].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"투자자 매매 필터 적용 실패: {e} — skip")

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

    # 5. 복합 점수 계산 (regime 모드면 카테고리 multiplier 주입)
    category_weights: dict[str, float] | None = None
    if regime_label and getattr(config, "regime", None):
        if config.regime.enabled and config.regime.lambda_ > 0:
            from src.regime.weights import get_category_weights
            base = get_category_weights(regime_label)
            if base:
                lam = config.regime.lambda_
                # lambda 보간: 1 → 풀 적용, 0 → 1.0 (영향 없음)
                category_weights = {k: 1.0 + lam * (v - 1.0) for k, v in base.items()}

    composite = compute_composite_score(
        factor_df, valid_factors, factor_report,
        method=config.factors.composite_method,
        category_weights=category_weights,
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

    regime_note = f" regime={regime_label}" if regime_label and category_weights else ""
    logger.info(
        f"종목풀 구성: {len(top_codes)}종목 "
        f"(신규 {len(entered)}, 퇴출 {len(exited)}) "
        f"date={date}{regime_note}"
    )
    if return_factors:
        return pool, factor_df
    return pool
