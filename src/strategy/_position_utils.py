"""
포지션 관련 공통 유틸 (factor_rl / factor_hybrid 공유).

- `business_days_held(entry_date, reference_date=None)` — 영업일 기준 보유 일수
- `resolve_buy_date(code, avg_price)` — DB → 일봉 추정 → 오늘 순서로 매수일 조회
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd


def business_days_held(entry_date: str, reference_date: str | None = None) -> int:
    """매수일부터 기준일까지 영업일(월~금) 수.

    Args:
        entry_date: 매수일 (YYYYMMDD)
        reference_date: 기준일 (YYYYMMDD 또는 YYYY-MM-DD). 백테스트용. None 이면 현재 시각.
    """
    try:
        entry = datetime.strptime(entry_date, "%Y%m%d")
        today = (
            datetime.strptime(reference_date.replace("-", ""), "%Y%m%d")
            if reference_date else datetime.now()
        )
        days = 0
        current = entry
        while current < today:
            current += timedelta(days=1)
            if current.weekday() < 5:  # 월~금
                days += 1
        return days
    except Exception:
        return 0


def resolve_buy_date(code: str, avg_price: float) -> str:
    """매수일을 조회합니다. DB → 일봉 추정 → 오늘 순서.

    1. DB `holding_positions` 조회
    2. 실패 시 최근 45일 일봉에서 avg_price 와 가장 가까운 종가 날짜 추정 (+ DB 저장)
    3. 모두 실패 시 오늘
    """
    # 1. DB
    try:
        from src.db.models import get_holding_buy_date
        db_date = get_holding_buy_date(code)
        if db_date:
            return db_date
    except Exception:
        pass

    # 2. 일봉 기반 추정
    try:
        from src.data.market_data import get_ohlcv
        end = datetime.now().strftime("%Y%m%d")
        start = (datetime.now() - timedelta(days=45)).strftime("%Y%m%d")
        df = get_ohlcv(code, start, end)
        if df is not None and not df.empty and avg_price > 0:
            df = df.copy()
            df["diff"] = (df["close"] - avg_price).abs()
            best_idx = df["diff"].idxmin()
            buy_date = pd.Timestamp(df.loc[best_idx, "date"]).strftime("%Y%m%d")
            try:
                from src.db.models import save_holding
                save_holding(code, "", int(avg_price), 0, buy_date)
            except Exception:
                pass
            return buy_date
    except Exception:
        pass

    # 3. fallback
    return datetime.now().strftime("%Y%m%d")
