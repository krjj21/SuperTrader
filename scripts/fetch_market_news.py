"""
시장 뉴스 수집 + 시장 sentiment 산출 (Regime Detector 입력)

동작:
  1. 시장 키워드("코스피", "Fed", "금리" ...)로 Google News RSS 수집
  2. 가짜 종목코드 _MARKET_ 로 sappo_news 에 저장
  3. Claude Haiku 로 risk-on/off 점수화 (-1 ~ +1)
  4. sappo_sentiment_scores 에 stock_code='_MARKET_' 로 upsert

이 sentiment 는 src/regime/detector.py 가 HMM 결과를 검증/오버라이드할 때
입력으로 사용한다.

Usage:
    python scripts/fetch_market_news.py
    python scripts/fetch_market_news.py --date 20260427    # 백필
"""
from __future__ import annotations

import argparse
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


MARKET_CODE = "_MARKET_"

# 시장 risk-on/off 신호로 쓰일 키워드 — 시황·거시·정책에 집중
MARKET_KEYWORDS = [
    "코스피",
    "코스닥 시황",
    "한국 증시",
    "거시경제",
    "Fed 금리",
    "한국은행 기준금리",
    "환율 원달러",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="시장 뉴스 + sentiment 수집")
    parser.add_argument("--date", type=str, default=None, help="대상 날짜 YYYYMMDD")
    parser.add_argument("--sleep", type=float, default=0.4, help="검색어간 sleep (초)")
    parser.add_argument("--skip-news", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="기존 sentiment 덮어쓰기")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    load_config("config/settings.yaml")
    init_sappo_db("data/trading.db")

    target_date = args.date or datetime.now().strftime("%Y%m%d")
    logger.info(f"시장 뉴스 수집 시작: date={target_date}, keywords={len(MARKET_KEYWORDS)}")

    # 1. 뉴스 수집 — 모든 시장 키워드를 _MARKET_ 코드로 누적
    if not args.skip_news:
        t0 = time.time()
        total = 0
        for kw in MARKET_KEYWORDS:
            try:
                items = collect_google_news(MARKET_CODE, kw, query_suffix="")
                total += len(items)
            except Exception as e:
                logger.warning(f"뉴스 수집 실패 [{kw}]: {e}")
            time.sleep(args.sleep)
        logger.info(f"시장 뉴스 수집 완료: {total}건 ({time.time()-t0:.0f}초)")

    # 2. Sentiment 생성 — _MARKET_ 코드 + target_date
    if not args.skip_sentiment:
        kwargs = {"overwrite": args.overwrite}
        if args.model:
            kwargs["model"] = args.model
        try:
            rec = generate_sentiment_for(MARKET_CODE, target_date, **kwargs)
            if rec is not None:
                logger.info(
                    f"시장 sentiment: score={rec.score:+.2f} conf={rec.confidence:.2f} "
                    f"n={rec.n_articles} rationale={rec.rationale[:120]}"
                )
            else:
                logger.warning("시장 sentiment 생성 실패 (None 반환)")
        except Exception as e:
            logger.error(f"시장 sentiment 생성 예외: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
