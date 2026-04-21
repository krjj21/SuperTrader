"""
Sentiment 생성기 — 뉴스 → LLM → 점수

파이프라인:
  1. sappo_news 에서 (stock_code, date) 기사 조회
  2. 제목 + 본문(있으면) 압축 후 LLM 호출
  3. JSON 파싱 → score(-1~+1), confidence(0~1), rationale
  4. sappo_sentiment_scores 에 upsert

모델: Claude Haiku (비용 최소, 대량 처리). Sonnet 은 옵션.

Usage:
    from src.timing.sentiment_generator import generate_sentiment_for
    res = generate_sentiment_for("005930", "20260420")
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import httpx
from loguru import logger

from src.config import get_secrets
from src.db.sappo_models import (
    NewsArticle, SentimentScore,
    get_sappo_session, get_news_for, upsert_sentiment, get_sentiment,
)


API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # 저비용, 대량 처리
MAX_ARTICLES_PER_CALL = 10
MAX_TITLE_LEN = 200
MAX_BODY_LEN = 500

SYSTEM_PROMPT = """You are a Korean financial sentiment analyst.
주어진 종목의 최근 뉴스를 읽고, 가격에 영향을 미칠 *sentiment score* 를 산출한다.

출력은 반드시 유효한 JSON. 필드:
  score: float, -1.0 ~ +1.0
    -1.0 = 명확한 악재 (실적 악화 공시, 사고, 규제, 소송, 회계 의혹 등)
    +1.0 = 명확한 호재 (실적 상향, 대형 계약, 신제품, 수급 호재 등)
     0.0 = 중립 / 판단 불가 / 일반 시황
  confidence: float, 0.0~1.0 (기사 수 + 신호의 명확성으로 판단)
  rationale: str, 2문장 이내 한국어 요약

[엄격 규칙]
- "영향 있을 듯", "가능성" 등 재량 추론 금지 — 기사에 명시된 사실 기반.
- 기사가 없거나 중립이면 score=0, confidence=0.
- 같은 날 호재·악재 혼재면 우세 쪽 — 우세 불명확이면 0.
- 3~10개 기사만 참고 (초과분은 무시).
- JSON 외 텍스트 절대 출력 금지."""


def _compact_articles(articles: list[NewsArticle], limit: int = MAX_ARTICLES_PER_CALL) -> str:
    """LLM 입력용 기사 블록 구성."""
    lines: list[str] = []
    for i, a in enumerate(articles[:limit], 1):
        title = (a.title or "")[:MAX_TITLE_LEN]
        body = (a.body or "")[:MAX_BODY_LEN]
        source = a.source or "-"
        published = a.published_at.strftime("%Y-%m-%d %H:%M") if a.published_at else ""
        line = f"{i}. [{source} {published}] {title}"
        if body:
            line += f"\n   {body}"
        lines.append(line)
    return "\n".join(lines)


def _call_llm(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 400,
    timeout: float = 20.0,
) -> str:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(
            API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": max_tokens,
                "system": [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                ],
                "messages": [{"role": "user", "content": user_prompt}],
            },
        )
        r.raise_for_status()
        data = r.json()
    return data["content"][0]["text"].strip()


def _parse_response(text: str) -> dict[str, Any] | None:
    """LLM 응답 JSON 파싱. 코드블록/앞뒤 텍스트 제거 후 json.loads."""
    if not text:
        return None
    text = text.strip()
    # 코드블록 제거
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # 첫 번째 { ... } 추출
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON 파싱 실패: {e} — raw={text[:200]}")
        return None
    if not isinstance(obj, dict):
        return None
    # 필드 정규화
    try:
        score = float(obj.get("score", 0.0))
        conf = float(obj.get("confidence", 0.0))
    except (ValueError, TypeError):
        return None
    score = max(-1.0, min(1.0, score))
    conf = max(0.0, min(1.0, conf))
    rationale = str(obj.get("rationale", ""))[:500]
    return {"score": score, "confidence": conf, "rationale": rationale}


def generate_sentiment_for(
    stock_code: str,
    date: str,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
) -> SentimentScore | None:
    """특정 종목·날짜의 뉴스를 바탕으로 sentiment 를 생성·저장합니다.

    Args:
        stock_code: 6자리 종목코드
        date: YYYYMMDD
        overwrite: False 면 이미 저장된 값이 있으면 skip

    Returns:
        저장된 SentimentScore 또는 None (기사 없음/API 오류/skip 등)
    """
    if not overwrite:
        existing = get_sentiment(stock_code, date)
        if existing is not None:
            return existing

    articles = get_news_for(stock_code, date)
    if not articles:
        # 기사 없음 → 중립 0 저장 (학습 시 null 구멍 방지)
        return upsert_sentiment(
            stock_code=stock_code, date=date,
            score=0.0, confidence=0.0, n_articles=0,
            rationale="no news", model=model,
        )

    secrets = get_secrets()
    if not secrets.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY 미설정 — 중립 0 으로 저장")
        return upsert_sentiment(
            stock_code=stock_code, date=date,
            score=0.0, confidence=0.0, n_articles=len(articles),
            rationale="LLM 비활성 — 중립", model="",
        )

    user_prompt = (
        f"종목: {stock_code}\n"
        f"날짜: {date[:4]}-{date[4:6]}-{date[6:8]}\n"
        f"기사 ({min(len(articles), MAX_ARTICLES_PER_CALL)}건):\n"
        f"{_compact_articles(articles)}\n\n"
        f"출력: {{\"score\": float, \"confidence\": float, \"rationale\": \"...\"}}"
    )

    try:
        raw = _call_llm(secrets.anthropic_api_key, SYSTEM_PROMPT, user_prompt, model=model)
    except Exception as e:
        logger.warning(f"LLM 호출 실패 [{stock_code} {date}]: {e}")
        return None

    parsed = _parse_response(raw)
    if parsed is None:
        logger.warning(f"LLM 응답 파싱 실패 [{stock_code} {date}] — raw={raw[:150]}")
        return upsert_sentiment(
            stock_code=stock_code, date=date,
            score=0.0, confidence=0.0, n_articles=len(articles),
            rationale=f"parse_failed: {raw[:200]}", model=model,
        )

    rec = upsert_sentiment(
        stock_code=stock_code, date=date,
        score=parsed["score"], confidence=parsed["confidence"],
        n_articles=len(articles), rationale=parsed["rationale"],
        model=model,
    )
    logger.info(
        f"sentiment [{stock_code} {date}]: score={rec.score:+.2f} "
        f"conf={rec.confidence:.2f} n={len(articles)}"
    )
    return rec


def generate_for_pool(
    codes: list[str],
    date: str,
    model: str = DEFAULT_MODEL,
    overwrite: bool = False,
) -> dict[str, float]:
    """종목풀 전체에 대해 지정일 sentiment 생성. 반환: {code: score}."""
    out: dict[str, float] = {}
    for code in codes:
        rec = generate_sentiment_for(code, date, model=model, overwrite=overwrite)
        if rec is not None:
            out[code] = rec.score
    return out
