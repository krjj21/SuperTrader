"""
시장 데이터 수집 모듈
- FinanceDataReader 기반 유니버스 구성 (PIT 메타 활용)
- pykrx 기반 OHLCV 로드 (FDR 폴백)
"""
from __future__ import annotations

import functools
import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import FinanceDataReader as fdr
from loguru import logger
from pykrx import stock as pykrx_stock

from src.config import get_config

UNIVERSE_META_PATH = Path("data/universe_meta.csv")
UNIVERSE_CACHE_DIR = Path("data/universe_cache")


@functools.lru_cache(maxsize=1)
def _load_universe_meta() -> pd.DataFrame | None:
    """data/universe_meta.csv를 로드합니다 (없으면 None).

    스키마: [code, name, market, listed_date, delisted_date]
    listed_date / delisted_date 는 datetime64. delisted_date NaT = 활성.
    """
    if not UNIVERSE_META_PATH.exists():
        return None
    try:
        df = pd.read_csv(UNIVERSE_META_PATH, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        df["listed_date"] = pd.to_datetime(df["listed_date"], errors="coerce")
        df["delisted_date"] = pd.to_datetime(df["delisted_date"], errors="coerce")
        df = df[df["listed_date"].notna()].reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"universe_meta 로드 실패: {e}")
        return None


def _meta_signature() -> str:
    """meta 파일의 mtime+size 해시 — 캐시 무효화 키.

    파일이 없으면 'no-meta'.
    """
    if not UNIVERSE_META_PATH.exists():
        return "no-meta"
    st = UNIVERSE_META_PATH.stat()
    raw = f"{int(st.st_mtime)}:{st.st_size}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def _today_marcap() -> dict[str, int]:
    """오늘 KRX 시총 스냅샷 (PIT 한계 — 시총은 today-cap 사용).

    실패 시 빈 dict (시총 필터 비활성).
    """
    try:
        listing = fdr.StockListing("KRX")
        codes = listing["Code"].astype(str).str.zfill(6)
        caps = pd.to_numeric(listing.get("Marcap", 0), errors="coerce").fillna(0).astype("int64")
        return dict(zip(codes, caps))
    except Exception as e:
        logger.warning(f"오늘 시총 스냅샷 실패: {e} — 시총 필터 비활성")
        return {}


def _rolling_volume_at(ohlcv_dict: dict[str, pd.DataFrame], code: str, ts: pd.Timestamp, window: int = 20) -> int:
    df = ohlcv_dict.get(code)
    if df is None or df.empty or "date" not in df.columns:
        return 0
    dates = pd.to_datetime(df["date"])
    mask = dates <= ts
    if not mask.any():
        return 0
    vol = df.loc[mask, "volume"].tail(window)
    if vol.empty:
        return 0
    return int(vol.mean())


def _legacy_get_universe() -> pd.DataFrame:
    """date=None 또는 meta 부재 시 사용하는 기존 라이브 경로."""
    config = get_config().universe

    try:
        if config.market == "KOSPI":
            listing = fdr.StockListing("KOSPI")
        elif config.market == "KOSDAQ":
            listing = fdr.StockListing("KOSDAQ")
        else:
            kospi = fdr.StockListing("KOSPI")
            kosdaq = fdr.StockListing("KOSDAQ")
            listing = pd.concat([kospi, kosdaq], ignore_index=True)
    except Exception as e:
        logger.error(f"종목 리스트 로드 실패: {e}")
        return pd.DataFrame()

    if listing.empty:
        logger.warning("종목 리스트가 비어 있습니다")
        return pd.DataFrame()

    df = pd.DataFrame()
    df["code"] = listing["Code"].astype(str).str.zfill(6)
    df["name"] = listing["Name"]
    df["market"] = listing.get("Market", config.market)
    df["market_cap"] = pd.to_numeric(listing.get("Marcap", 0), errors="coerce").fillna(0).astype(int)
    df["volume"] = pd.to_numeric(listing.get("Volume", 0), errors="coerce").fillna(0).astype(int)

    df = df[df["market_cap"] >= config.min_market_cap]
    df = df[~df["name"].str.contains("스팩|리츠|ETF|ETN", na=False)]
    df = df[~df["code"].str[-1].isin(["5", "7", "9", "K", "L"])]

    filtered = df[df["volume"] >= config.min_avg_volume]
    if len(filtered) >= 30:
        df = filtered
        logger.info(f"유니버스 구성: {len(df)}종목 ({config.market}) [legacy]")
    else:
        logger.warning(
            f"유니버스 거래량 필터 통과 {len(filtered)}종목 — 시장 초반/거래 부족 의심. "
            f"volume 필터 스킵 (market_cap 만 적용, 결과 {len(df)}종목)"
        )
    return df.sort_values("market_cap", ascending=False).reset_index(drop=True)


def get_universe(
    date: str | None = None,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """투자 유니버스를 반환합니다.

    date=None: 기존 라이브 경로 (FDR 오늘 스냅샷).
    date 지정 + universe_meta.csv 존재: PIT 멤버십 (listed<=date<=(delisted or today))
        + 오늘 시총 + (가능하면) D 시점 rolling-20일 거래량.
    date 지정 + meta 부재: 단일 WARNING + legacy fallback.

    Args:
        date: 기준일 (YYYYMMDD 또는 YYYY-MM-DD)
        ohlcv_dict: 사전 로드된 OHLCV — 거래량 PIT 필터에 사용 (없으면 오늘 거래량 fallback)

    Returns:
        DataFrame [code, name, market, market_cap, volume]
    """
    if date is None:
        return _legacy_get_universe()

    meta = _load_universe_meta()
    if meta is None:
        logger.warning(
            "universe_meta.csv 부재 — legacy 경로 fallback. "
            "scripts/build_universe_meta.py 실행 권장."
        )
        return _legacy_get_universe()

    config = get_config().universe
    ts = pd.Timestamp(date.replace("-", ""))

    # 캐시 조회
    cache_disabled = bool(os.environ.get("SUPER_TRADER_DISABLE_UNIVERSE_CACHE"))
    cache_path: Path | None = None
    if not cache_disabled:
        key_raw = (
            f"{ts.strftime('%Y%m%d')}|{config.market}|{config.min_market_cap}|"
            f"{config.min_avg_volume}|{_meta_signature()}|"
            f"{'pit-vol' if ohlcv_dict else 'today-vol'}"
        ).encode()
        key = hashlib.sha1(key_raw).hexdigest()[:12]
        cache_path = UNIVERSE_CACHE_DIR / f"uni_{key}.csv"
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, dtype={"code": str})
                cached["code"] = cached["code"].str.zfill(6)
                return cached
            except Exception as e:
                logger.debug(f"universe_cache 로드 실패: {cache_path.name} - {e}")

    # PIT 멤버십 필터
    df = meta[
        (meta["listed_date"] <= ts)
        & (meta["delisted_date"].isna() | (meta["delisted_date"] >= ts))
    ].copy()

    # 시장 필터
    if config.market in ("KOSPI", "KOSDAQ"):
        df = df[df["market"] == config.market]

    # 이름/코드 제외 (방어적 재적용)
    df = df[~df["name"].astype(str).str.contains("스팩|리츠|ETF|ETN", na=False)]
    df = df[~df["code"].str[-1].isin(["5", "7", "9", "K", "L"])]

    if df.empty:
        logger.warning(f"PIT 유니버스 비어 있음 (date={date})")
        return pd.DataFrame(columns=["code", "name", "market", "market_cap", "volume"])

    # 시총 (오늘 스냅샷 — PIT 잔존 한계).
    # 폐지 종목은 cap_map에 없으므로 cap 필터를 면제 (Tier 3 미해결).
    # 그렇지 않으면 모든 폐지 종목이 cap=0으로 떨어져 생존편향이 그대로 남음.
    cap_map = _today_marcap()
    df["market_cap"] = df["code"].map(cap_map).fillna(0).astype("int64")
    if cap_map:
        is_delisted = df["delisted_date"].notna()
        active_pass = (~is_delisted) & (df["market_cap"] >= config.min_market_cap)
        df = df[active_pass | is_delisted]

    # 거래량 (가능하면 D 시점 rolling-20일 평균)
    if ohlcv_dict:
        df["volume"] = df["code"].apply(lambda c: _rolling_volume_at(ohlcv_dict, c, ts))
        filtered = df[df["volume"] >= config.min_avg_volume]
        if len(filtered) >= 30:
            df = filtered
        else:
            logger.warning(
                f"PIT 거래량 필터 통과 {len(filtered)}종목 (date={date}) — 필터 스킵"
            )
    else:
        df["volume"] = 0  # ohlcv 없이 호출된 경우 거래량 미평가

    df = df[["code", "name", "market", "market_cap", "volume"]]
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    logger.info(f"PIT 유니버스: {len(df)}종목 (date={date}, market={config.market})")

    if cache_path is not None:
        try:
            UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.debug(f"universe_cache 저장 실패: {e}")

    return df


def get_ohlcv(
    code: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """단일 종목 OHLCV를 조회합니다.

    Args:
        code: 종목코드 (예: "005930")
        start: 시작일 (YYYYMMDD 또는 YYYY-MM-DD)
        end: 종료일

    Returns:
        DataFrame [date, open, high, low, close, volume]
    """
    # 날짜 형식 변환
    s = start.replace("-", "")
    e = end.replace("-", "")
    start_fmt = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    end_fmt = f"{e[:4]}-{e[4:6]}-{e[6:8]}"

    try:
        df = fdr.DataReader(code, start_fmt, end_fmt)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        return df

    df = df.reset_index()
    # FDR 컬럼: Date, Open, High, Low, Close, Volume, Change
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "date":
            col_map[c] = "date"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif cl == "volume":
            col_map[c] = "volume"

    df = df.rename(columns=col_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"])

    # 숫자형 변환
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def filter_by_listing_date(
    ohlcv_dict: dict[str, pd.DataFrame],
    start: str,
    grace_days: int = 30,
) -> dict[str, pd.DataFrame]:
    """backtest 시작일 이후에 IPO 된 종목을 제외합니다 (방어적 net).

    각 종목의 OHLCV 가 `start` 이전(또는 grace_days 이내) 에 시작하면 유지.
    이후 IPO 는 lookahead/생존자 편향을 만드므로 제외.

    Note:
        data/universe_meta.csv 가 존재하면 get_universe(date) 가 이미 PIT
        멤버십을 강제하므로 이 함수는 사실상 no-op 이 된다. 그래도 보존하는
        이유는 (1) meta 부재 시 fallback, (2) FDR 데이터 자체에 비정상적인
        후방 상장 데이터가 섞여 있을 때의 안전장치.

    Args:
        ohlcv_dict: {종목코드: OHLCV DataFrame}
        start: 기준일 (YYYYMMDD 또는 YYYY-MM-DD)
        grace_days: 기준일 이후 grace_days 이내 상장한 종목도 제외

    Returns:
        필터링된 dict
    """
    start_fmt = pd.Timestamp(start.replace("-", ""))
    cutoff = start_fmt + pd.Timedelta(days=grace_days)

    kept = {}
    dropped_codes = []
    for code, df in ohlcv_dict.items():
        if df.empty or "date" not in df.columns:
            continue
        first = pd.to_datetime(df["date"].iloc[0])
        if first <= cutoff:
            kept[code] = df
        else:
            dropped_codes.append((code, first.strftime("%Y-%m-%d")))

    if dropped_codes:
        logger.info(
            f"IPO 이후 상장 종목 제외: {len(dropped_codes)}종목 "
            f"(예: {dropped_codes[:3]})"
        )
    return kept


def get_ohlcv_batch(
    codes: list[str],
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    """여러 종목의 OHLCV를 배치 로드합니다.

    Returns:
        {종목코드: OHLCV DataFrame} 딕셔너리
    """
    meta = _load_universe_meta()
    if meta is not None:
        delisted_in_codes = meta[meta["code"].isin(codes) & meta["delisted_date"].notna()]
        logger.info(
            f"PIT mode: {len(codes)} codes incl. {len(delisted_in_codes)} delisted"
        )
    result = {}
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            df = get_ohlcv(code, start, end)
            if not df.empty and len(df) >= 20:
                result[code] = df
        except Exception as e:
            logger.debug(f"OHLCV 로드 실패: {code} - {e}")

        # API rate limit 방지
        if (i + 1) % 5 == 0:
            time.sleep(0.5)

        if (i + 1) % 50 == 0:
            logger.info(f"OHLCV 로드 진행: {i+1}/{total} ({len(result)}종목 성공)")

    logger.info(f"OHLCV 배치 로드 완료: {len(result)}/{total}종목")
    return result


def get_close_prices(
    codes: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """여러 종목의 종가를 피벗 테이블로 반환합니다."""
    frames = []
    for code in codes:
        try:
            df = get_ohlcv(code, start, end)
            if not df.empty:
                s = df.set_index("date")["close"].rename(code)
                frames.append(s)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def get_returns(
    codes: list[str],
    start: str,
    end: str,
    period: int = 1,
) -> pd.DataFrame:
    """여러 종목의 수익률을 계산합니다."""
    prices = get_close_prices(codes, start, end)
    if prices.empty:
        return prices
    return prices.pct_change(period)
