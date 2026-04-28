"""한국 주식시장 영업일/휴장일 판정.

판정 규칙 (2026-04-28 fix):
- 토/일: 즉시 False (네트워크 호출 없음)
- 평일 + 과거 날짜(또는 평일 + 16:00 이후 오늘): KS11 OHLCV 행 존재 여부로 휴장 자동 감지
- *평일 + 오늘 + 일중(<16:00 KST)*: FDR 가 KS11 일중 데이터를 제공하지 않으므로
  무조건 휴장으로 보면 정상 영업일에 매매가 차단됨. 따라서:
    · 같은 주 다른 영업일에 KS11 데이터가 존재 → FDR 정상 작동 → "오늘은 일중이라 데이터 미반영" → True
    · 같은 주 데이터가 전혀 없음 → FDR 자체가 안 받힘 → 보수적으로 True (운영자가 로그 보고 개입)
- FDR 호출 실패(네트워크 단절 등) 시: 보수적으로 True (sleeping silence 회피)

날짜 입력 포맷: "YYYYMMDD" 또는 datetime.date / datetime.datetime
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Union

from loguru import logger

DateLike = Union[str, date, datetime]

# 일중(장중) 판정 기준 — KST 16:00 (post-market 16:00 직후 FDR 가 KS11 일봉 채우기 시작)
INTRADAY_CUTOFF = time(16, 0)

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


def _fetch_week_ks11_dates(d: date) -> set[date] | None:
    """주어진 날짜가 속한 주 (월~일) KS11 데이터의 *날짜 집합* 반환.
    None = FDR 호출 실패."""
    try:
        import FinanceDataReader as fdr
        monday = d - timedelta(days=d.weekday())
        sunday = monday + timedelta(days=6)
        df = fdr.DataReader(
            "KS11",
            monday.strftime("%Y-%m-%d"),
            sunday.strftime("%Y-%m-%d"),
        )
        if df is None or df.empty:
            return set()
        return {x.date() if hasattr(x, "date") else x for x in df.index}
    except Exception as e:
        logger.warning(f"[CALENDAR] FDR 조회 실패 ({d}): {e}")
        return None


def is_korean_market_open(d: DateLike | None = None) -> bool:
    """주어진 날짜가 KOSPI 영업일인가? (None=오늘 KST)

    상세 규칙은 모듈 docstring 참고.
    """
    now = datetime.now()
    target = _to_date(d) if d is not None else now.date()
    key = target.strftime("%Y%m%d")
    if key in _open_cache:
        return _open_cache[key]

    # 1. 주말 즉시 False
    if _is_weekend(target):
        _open_cache[key] = False
        return False

    # 2. KS11 주간 데이터 조회
    week_dates = _fetch_week_ks11_dates(target)

    # 2-a. FDR 호출 실패 → 보수적으로 True (운영자가 로그 보고 개입)
    if week_dates is None:
        logger.warning(
            f"[CALENDAR] {key} FDR 호출 실패 — 평일이라 영업일로 간주"
        )
        # *오늘만* 캐시하지 않고 즉시 반환 (다음 호출 시 재시도)
        return True

    # 2-b. 그날 데이터 존재 → 영업일
    if target in week_dates:
        _open_cache[key] = True
        return True

    # 2-c. 그날 데이터 없음 → 일중 판정 분기
    is_today = (target == now.date())
    intraday = is_today and now.time() < INTRADAY_CUTOFF

    if intraday:
        # FDR 가 일중 KS11 데이터를 주지 않음. 같은 주 다른 영업일 데이터로 검증
        if week_dates:
            # 정상 작동 중인 FDR — "오늘은 일중이라 데이터 미반영" 으로 해석
            logger.info(
                f"[CALENDAR] {key} 일중 (now<{INTRADAY_CUTOFF}) — FDR 일중 미반영, 영업일로 가정 "
                f"(같은 주 영업일 {len(week_dates)}개 확인)"
            )
            # 일중 캐시는 위험 (16:00 이후에도 같은 결론 유지) → 캐시 안 함
            return True
        else:
            # FDR 가 그 주 전체 데이터 없음 → 정말 휴장이거나 데이터 갱신 안 됨
            # 보수적으로 영업일 가정 (캐시 안 함)
            logger.warning(
                f"[CALENDAR] {key} 일중인데 같은 주 KS11 데이터 0건 — 보수적으로 영업일 가정"
            )
            return True

    # 2-d. 평일 + 16:00 이후 + KS11 부재 → 진짜 휴장
    _open_cache[key] = False
    logger.info(f"[CALENDAR] {key} 휴장일 감지 (KS11 데이터 없음, post-16:00)")
    return False


def is_market_holiday(d: DateLike | None = None) -> bool:
    """주어진 날짜가 휴장일인가? = NOT is_korean_market_open"""
    return not is_korean_market_open(d)


def reset_cache() -> None:
    """테스트/디버그 용 캐시 초기화. 라이브 코드 변경 후 invalidation 용도."""
    _open_cache.clear()
