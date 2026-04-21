"""
한국 주식 뉴스 수집기

1차 소스: Google News RSS
  URL 패턴: https://news.google.com/rss/search?q=<쿼리>&hl=ko&gl=KR&ceid=KR:ko
  - 종목명 기반 검색 (pykrx 에서 종목명 조회)
  - title/link/pubDate/source 제공
  - 본문은 각 기사 URL 방문 시 별도 파싱 필요 (선택적)

DB 저장: src/db/sappo_models.save_news (url unique 로 중복 방지)

Usage:
    from src.data.news_collector import collect_google_news
    collect_google_news("005930", "삼성전자")
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from xml.etree import ElementTree as ET
from typing import Any

import httpx
from loguru import logger

from src.db.sappo_models import save_news

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


def _fetch(url: str, params: dict | None = None, timeout: float = 10.0) -> str | None:
    try:
        with httpx.Client(headers=DEFAULT_HEADERS, timeout=timeout, follow_redirects=True) as c:
            r = c.get(url, params=params)
            r.raise_for_status()
            return r.text
    except Exception as e:
        logger.debug(f"GET 실패 {url}: {e}")
        return None


def _parse_rss_items(xml_text: str) -> list[dict[str, Any]]:
    """RSS XML 에서 items 를 dict 리스트로 반환."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning(f"RSS 파싱 실패: {e}")
        return []

    items: list[dict[str, Any]] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_raw = item.findtext("pubDate") or ""
        source_el = item.find("source")
        source_name = source_el.text if source_el is not None and source_el.text else ""
        description = (item.findtext("description") or "").strip()

        pub_dt: datetime | None = None
        if pub_raw:
            try:
                pub_dt = parsedate_to_datetime(pub_raw)
                if pub_dt.tzinfo is not None:
                    pub_dt = pub_dt.astimezone().replace(tzinfo=None)
            except Exception:
                pub_dt = None

        if not title or not link:
            continue

        items.append({
            "title": title,
            "url": link,
            "source": f"google:{source_name}" if source_name else "google",
            "published_at": pub_dt,
            "description": description,
        })
    return items


def collect_google_news(
    stock_code: str,
    stock_name: str,
    query_suffix: str = "주식",
    persist: bool = True,
) -> list[dict[str, Any]]:
    """종목명으로 Google News RSS 검색 → 수집 후 DB 저장.

    Args:
        stock_code: 6자리 종목코드 (DB 키)
        stock_name: 종목명 (검색 쿼리에 사용)
        query_suffix: 쿼리 부가어 ("주식" / "" / "증권" 등)
        persist: True 면 sappo_news 에 저장 (url 중복 skip)

    Returns:
        수집 dict 리스트
    """
    if not stock_name:
        logger.debug(f"종목명 없음 — skip: {stock_code}")
        return []

    q = f"{stock_name} {query_suffix}".strip()
    url = f"{GOOGLE_NEWS_RSS}?q={quote(q)}&hl=ko&gl=KR&ceid=KR:ko"
    xml_text = _fetch(url)
    if not xml_text:
        return []

    items = _parse_rss_items(xml_text)

    saved = 0
    for it in items:
        if persist:
            date_str = (
                it["published_at"].strftime("%Y%m%d")
                if it.get("published_at")
                else datetime.now().strftime("%Y%m%d")
            )
            try:
                rec = save_news(
                    stock_code=stock_code,
                    date=date_str,
                    source=it["source"],
                    title=it["title"],
                    body=it.get("description", ""),
                    url=it["url"],
                    published_at=it.get("published_at"),
                )
                if rec is not None:
                    saved += 1
            except Exception as e:
                logger.debug(f"save_news 실패 ({stock_code}): {e}")

    logger.info(f"구글 뉴스 수집: {stock_name}({stock_code}) {len(items)}건 (신규 저장 {saved})")
    return items


def collect_for_pool(
    codes: list[str],
    name_map: dict[str, str],
    sleep_sec: float = 0.5,
) -> dict[str, int]:
    """종목풀 전체 뉴스 수집. name_map: {code: name}. 반환: {code: 수집건수}."""
    result: dict[str, int] = {}
    for code in codes:
        name = name_map.get(code, "")
        if not name:
            try:
                from pykrx import stock as krx
                name = krx.get_market_ticker_name(code)
            except Exception:
                name = ""
        items = collect_google_news(code, name)
        result[code] = len(items)
        time.sleep(sleep_sec)
    return result


# ══════════════════════════════════════════════════════════════
# 본문 추출 (선택적; 구글 링크는 중간 redirect 라 비용↑)
# ══════════════════════════════════════════════════════════════
def fetch_body_best_effort(url: str, timeout: float = 8.0) -> str:
    """URL 방문해 본문 텍스트를 best-effort 로 추출. 실패 시 빈 문자열."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""
    html = _fetch(url, timeout=timeout)
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        # 공통 본문 후보
        candidates = [
            "article", "div#articleBodyContents", "div.article_body",
            "div#content", "div.news_body", "div#newsct_article",
            "div#news_body_area", "main",
        ]
        for sel in candidates:
            el = soup.select_one(sel)
            if el:
                text = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
                if len(text) > 200:
                    return text[:4000]
        # fallback: body 전체
        body = soup.select_one("body")
        if body:
            return re.sub(r"\s+", " ", body.get_text(" ", strip=True))[:4000]
    except Exception:
        pass
    return ""
