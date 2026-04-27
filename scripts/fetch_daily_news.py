"""
일일 뉴스 수집 + Sentiment 생성 cron 진입점

동작:
  1. 현재 종목풀(data/current_pool.json) 의 각 종목에 대해 Google News RSS 수집
  2. 오늘 날짜 기준 sentiment 생성 (Claude Haiku)
  3. sappo_news / sappo_sentiment_scores 테이블 갱신

Usage:
    # 오늘 종목풀에 대해
    python scripts/fetch_daily_news.py

    # 특정 날짜 (백필용)
    python scripts/fetch_daily_news.py --date 20260419

    # 종목코드 목록 직접 지정
    python scripts/fetch_daily_news.py --codes 005930 000660 035420
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.db.sappo_models import init_sappo_db
from src.data.news_collector import collect_google_news
from src.timing.sentiment_generator import generate_sentiment_for


def _load_current_pool() -> list[tuple[str, str]]:
    """data/current_pool.json 에서 [(code, name), ...] 추출."""
    from src.utils.json_io import load_json_with_fallback
    data = load_json_with_fallback("data/current_pool.json")
    if data is None:
        return []
    stocks = data.get("stocks", [])
    return [(s["code"], s.get("name", "")) for s in stocks]


from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

_NAME_LOOKUP_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def _resolve_name(code: str) -> str:
    try:
        from pykrx import stock as krx
    except Exception:
        return ""

    fut = _NAME_LOOKUP_EXECUTOR.submit(
        lambda: krx.get_market_ticker_name(code) or ""
    )
    try:
        return fut.result(timeout=5.0)
    except FuturesTimeout:
        logger.warning(f"pykrx 종목명 조회 timeout: {code}")
        return ""
    except Exception:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="일일 뉴스 수집 + sentiment 생성")
    parser.add_argument("--date", type=str, default=None, help="대상 날짜 YYYYMMDD (기본: 오늘)")
    parser.add_argument("--codes", type=str, nargs="*", default=None, help="종목코드 목록")
    parser.add_argument("--sleep", type=float, default=0.5, help="종목간 sleep (초)")
    parser.add_argument("--skip-news", action="store_true", help="뉴스 수집 건너뜀 (sentiment 만)")
    parser.add_argument("--skip-sentiment", action="store_true", help="sentiment 생성 건너뜀")
    parser.add_argument("--overwrite", action="store_true", help="기존 sentiment 덮어쓰기")
    parser.add_argument("--model", type=str, default=None, help="Claude 모델 ID override")
    args = parser.parse_args()

    load_config("config/settings.yaml")
    init_sappo_db("data/trading.db")

    target_date = args.date or datetime.now().strftime("%Y%m%d")

    if args.codes:
        pairs: list[tuple[str, str]] = [(c, _resolve_name(c)) for c in args.codes]
    else:
        pairs = _load_current_pool()

    if not pairs:
        logger.error("대상 종목 없음 — --codes 지정하거나 data/current_pool.json 필요")
        return 1

    logger.info(f"대상 {len(pairs)}종목, 날짜={target_date}")

    # 1. 뉴스 수집
    if not args.skip_news:
        t0 = time.time()
        for code, name in pairs:
            name = name or _resolve_name(code)
            if not name:
                logger.debug(f"종목명 없음, skip: {code}")
                continue
            try:
                collect_google_news(code, name)
            except Exception as e:
                logger.warning(f"뉴스 수집 실패 [{code}]: {e}")
            time.sleep(args.sleep)
        logger.info(f"뉴스 수집 완료: {time.time()-t0:.0f}초")

    # 2. Sentiment 생성
    if not args.skip_sentiment:
        t0 = time.time()
        saved = 0
        for code, _name in pairs:
            try:
                kwargs = {"overwrite": args.overwrite}
                if args.model:
                    kwargs["model"] = args.model
                rec = generate_sentiment_for(code, target_date, **kwargs)
                if rec is not None:
                    saved += 1
            except Exception as e:
                logger.warning(f"sentiment 실패 [{code}]: {e}")
            time.sleep(0.3)
        logger.info(f"sentiment 생성 완료: {saved}/{len(pairs)}, {time.time()-t0:.0f}초")

    return 0


if __name__ == "__main__":
    sys.exit(main())
