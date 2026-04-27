"""한국 주식시장 영업일/휴장일 판정.

- 주말(토/일)은 즉시 False (네트워크 호출 없음)
- 평일이지만 KOSPI(KS11)에 그날 OHLCV 행이 없으면 휴장으로 판정
  → 한국 공휴일 / 임시 휴장(예: 2026-12-31 연말휴장) 모두 자동 처리
- 결과는 메모리 dict 에 캐시 (프로세스 lifetime)
- FDR 호출 실패(네트워크 단절 등) 시: 보수적으로 *영업일로 가정* 하여
  reporter 가 침묵하지 않도록 함 → 운영자가 로그를 보고 수동 개입 가능

날짜 입력 포맷: "YYYYMMDD" 또는 datetime.date / datetime.datetime
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Union

from loguru import logger

DateLike = Union[str, date, datetime]

# (YYYYMMDD → bool open?)
_open_cache: dict[str, bool] = {}


def _to_yyyymmdd(d: DateLike) -> str:
    if isinstance(d, str):
        return d.replace("-", "")[:8]
    if isinstance(d, datetime):
        return d.strftime("%Y%m%d")
    if isinstance(d, date):
        return d.strftime("%Y%m%d")
    raise TypeError(f"Unsupported date type: {type(d)}")


def _to_date(d: DateLike) -> date:
    s = _to_yyyymmdd(d)
    return datetime.strptime(s, "%Y%m%d").date()


def _is_weekend(d: date) -> bool:
    return d.weekday() >= 5  # 5=토, 6=일


def _query_fdr_open(d: date) -> bool:
    """FDR 로 KS11 의 그날 OHLCV 가 존재하는지 확인.
    범위는 그 주 월~일을 받아서 d 가 포함되는지 체크 (단일 날짜 호출보다 안정적)."""
    try:
        import FinanceDataReader as fdr
        # 해당 주의 월요일 ~ 일요일 범위 조회
        monday = d - timedelta(days=d.weekday())
        sunday = monday + timedelta(days=6)
        df = fdr.DataReader(
            "KS11",
            monday.strftime("%Y-%m-%d"),
            sunday.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            return False
        # FDR index 가 datetime 일 때 .date() 로 비교
        idx_dates = {x.date() if hasattr(x, "date") else x for x in df.index}
        return d in idx_dates
    except Exception as e:
        logger.warning(
            f"[CALENDAR] FDR 조회 실패 ({d}): {e} — 보수적으로 영업일로 간주"
        )
        return True


def is_korean_market_open(d: DateLike | None = None) -> bool:
    """주어진 날짜가 KOSPI 영업일인가? (None=오늘 KST)

    - 토/일: False
    - 평일: KS11 데이터 존재 여부로 판정 (휴일 자동 감지)
    - 결과 캐시
    """
    target = _to_date(d) if d is not None else datetime.now().date()
    key = target.strftime("%Y%m%d")
    if key in _open_cache:
        return _open_cache[key]

    if _is_weekend(target):
        _open_cache[key] = False
        return False

    is_open = _query_fdr_open(target)
    _open_cache[key] = is_open
    if not is_open:
        logger.info(f"[CALENDAR] {key} 휴장일 감지 (KS11 데이터 없음)")
    return is_open


def is_market_holiday(d: DateLike | None = None) -> bool:
    """주어진 날짜가 휴장일인가? = NOT is_korean_market_open"""
    return not is_korean_market_open(d)


def reset_cache() -> None:
    """테스트/디버그 용 캐시 초기화."""
    _open_cache.clear()
