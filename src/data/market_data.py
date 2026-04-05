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

    # 필터링
    df = df[df["market_cap"] >= config.min_market_cap]
    df = df[df["volume"] >= config.min_avg_volume]

    # 관리종목, 우선주, ETF 등 제외
    df = df[~df["name"].str.contains("스팩|리츠|ETF|ETN", na=False)]
    # 우선주 코드: 끝이 5,7,9,K,L
    df = df[~df["code"].str[-1].isin(["5", "7", "9", "K", "L"])]

    df = df.sort_values("market_cap", ascending=False).reset_index(drop=True)
    logger.info(f"유니버스 구성: {len(df)}종목 ({config.market})")
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
