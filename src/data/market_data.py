"""
시장 데이터 수집 모듈
- FinanceDataReader 기반 유니버스 구성
- pykrx 기반 OHLCV 로드 (FDR 폴백)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd
import FinanceDataReader as fdr
from loguru import logger
from pykrx import stock as pykrx_stock

from src.config import get_config


def get_universe(date: str | None = None) -> pd.DataFrame:
    """투자 유니버스를 반환합니다 (FinanceDataReader 기반).

    Args:
        date: 기준일 (YYYYMMDD). 현재는 최신 상장 종목 사용.

    Returns:
        DataFrame with columns: [code, name, market, market_cap, volume]
    """
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

    # 컬럼 정리
    df = pd.DataFrame()
    df["code"] = listing["Code"].astype(str).str.zfill(6)
    df["name"] = listing["Name"]
    df["market"] = listing.get("Market", config.market)
    df["market_cap"] = pd.to_numeric(listing.get("Marcap", 0), errors="coerce").fillna(0).astype(int)
    df["volume"] = pd.to_numeric(listing.get("Volume", 0), errors="coerce").fillna(0).astype(int)

    # 필터링: market_cap 먼저 적용
    df = df[df["market_cap"] >= config.min_market_cap]

    # 관리종목, 우선주, ETF 등 제외 (volume 필터 전에 먼저)
    df = df[~df["name"].str.contains("스팩|리츠|ETF|ETN", na=False)]
    df = df[~df["code"].str[-1].isin(["5", "7", "9", "K", "L"])]

    # 거래량 필터 — FDR Volume 은 "당일 실시간 거래량" 이라 시장 초반에는
    # 대부분 종목이 기준 미달. 결과가 기대치 이하면 필터 완화.
    filtered = df[df["volume"] >= config.min_avg_volume]
    if len(filtered) >= 30:
        df = filtered
        logger.info(f"유니버스 구성: {len(df)}종목 ({config.market})")
    else:
        logger.warning(
            f"유니버스 거래량 필터 통과 {len(filtered)}종목 — 시장 초반/거래 부족 의심. "
            f"volume 필터 스킵 (market_cap 만 적용, 결과 {len(df)}종목)"
        )
    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
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
    """backtest 시작일 이후에 IPO 된 종목을 제외합니다.

    각 종목의 OHLCV 가 `start` 이전(또는 grace_days 이내) 에 시작하면 유지.
    이후 IPO 는 lookahead/생존자 편향을 만드므로 제외.

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
