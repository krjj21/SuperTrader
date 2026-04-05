"""
펀더멘털 팩터 데이터 수집 모듈
- pykrx 기반 PER, PBR, EPS, 배당수익률, 시가총액
"""
from __future__ import annotations

import pandas as pd
from loguru import logger
from pykrx import stock as pykrx_stock


def get_fundamentals(date: str, market: str = "ALL") -> pd.DataFrame:
    """주어진 날짜의 펀더멘털 데이터를 반환합니다.

    Returns:
        DataFrame [code, per, pbr, eps, div_yield]
    """
    frames = []

    markets = []
    if market in ("ALL", "KOSPI"):
        markets.append("KOSPI")
    if market in ("ALL", "KOSDAQ"):
        markets.append("KOSDAQ")

    for mkt in markets:
        try:
            df = pykrx_stock.get_market_fundamental(date, market=mkt)
            if not df.empty:
                df["market"] = mkt
                frames.append(df)
        except Exception as e:
            logger.warning(f"펀더멘털 로드 실패: {mkt} {date} - {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames)
    result = result.reset_index()
    result.columns = ["code", "bps", "per", "pbr", "eps", "div_yield", "market"]
    return result


def get_market_cap(date: str, market: str = "ALL") -> pd.DataFrame:
    """시가총액 및 거래량 데이터를 반환합니다.

    Returns:
        DataFrame [code, market_cap, volume, traded_value, listed_shares]
    """
    frames = []

    markets = []
    if market in ("ALL", "KOSPI"):
        markets.append("KOSPI")
    if market in ("ALL", "KOSDAQ"):
        markets.append("KOSDAQ")

    for mkt in markets:
        try:
            df = pykrx_stock.get_market_cap(date, market=mkt)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            logger.warning(f"시가총액 로드 실패: {mkt} {date} - {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames).reset_index()
    result.columns = ["code", "close", "market_cap_change", "market_cap", "volume", "traded_value", "listed_shares"]
    return result[["code", "market_cap", "volume", "traded_value", "listed_shares"]]


def get_sector_info(date: str) -> pd.DataFrame:
    """종목별 섹터(업종) 정보를 반환합니다.

    Returns:
        DataFrame [code, sector]
    """
    records = []
    # KOSPI 업종별 종목 조회
    try:
        sectors = pykrx_stock.get_index_ticker_list(date, market="KOSPI")
        for sector_code in sectors:
            sector_name = pykrx_stock.get_index_ticker_name(sector_code)
            try:
                # 업종 구성종목
                comps = pykrx_stock.get_index_portfolio_deposit_file(sector_code, date)
                for stock_code in comps:
                    records.append({"code": stock_code, "sector": sector_name})
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"섹터 정보 로드 실패: {e}")

    if not records:
        return pd.DataFrame(columns=["code", "sector"])

    df = pd.DataFrame(records).drop_duplicates(subset="code", keep="first")
    return df
