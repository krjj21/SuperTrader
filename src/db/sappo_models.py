"""
SAPPO (Sentiment-Augmented PPO) 관련 DB 모델

테이블 구조:
  - news                    원본 뉴스 기사 (종목·날짜·URL 유일)
  - sentiment_scores        종목별 일일 sentiment 점수
  - sappo_training_runs     RL 학습 런 기록 (λ, Sharpe 등)
  - sentiment_ic_metrics    sentiment → 미래수익률 IC 측정 스냅샷
  - sappo_weekly_metrics    주간 집계 (검증 리포트용 singleton-per-week)

기존 src/db/models.py 의 `Base` 와 별도 메타데이터를 가져 독립 DB 로 운용 가능
(기본적으로는 같은 trading.db 안에 공존).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    create_engine, event, UniqueConstraint, Index,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class SappoBase(DeclarativeBase):
    pass


# ══════════════════════════════════════════════════════════════
# 1. 뉴스 원본
# ══════════════════════════════════════════════════════════════
class NewsArticle(SappoBase):
    """종목별 뉴스 원본 기사."""
    __tablename__ = "sappo_news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False, index=True)
    date = Column(String(8), nullable=False, index=True)      # YYYYMMDD
    source = Column(String(30), default="")                    # naver/hankyung/mk/dart
    title = Column(Text, default="")
    body = Column(Text, default="")
    url = Column(String(500), unique=True)
    published_at = Column(DateTime, nullable=True)
    collected_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_sappo_news_code_date", "stock_code", "date"),
    )


# ══════════════════════════════════════════════════════════════
# 2. 일일 sentiment 점수
# ══════════════════════════════════════════════════════════════
class SentimentScore(SappoBase):
    """종목·날짜별 sentiment 점수 (LLM 생성).

    score: -1.0 (명확한 악재) ~ +1.0 (명확한 호재), 0.0 중립
    confidence: 0.0~1.0, 기사 수/신호 강도 기반
    """
    __tablename__ = "sappo_sentiment_scores"

    stock_code = Column(String(10), primary_key=True)
    date = Column(String(8), primary_key=True)                 # YYYYMMDD
    score = Column(Float, nullable=False, default=0.0)
    confidence = Column(Float, default=0.0)
    n_articles = Column(Integer, default=0)
    rationale = Column(Text, default="")
    model = Column(String(64), default="")                     # claude-haiku-4-5 등
    generated_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_sappo_sent_date", "date"),
    )


# ══════════════════════════════════════════════════════════════
# 3. RL 학습 런 기록
# ══════════════════════════════════════════════════════════════
class SappoTrainingRun(SappoBase):
    """PPO/SAPPO 학습 런 이력."""
    __tablename__ = "sappo_training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(100), unique=True)                # "sappo_lambda0.1_20260501"
    lambda_value = Column(Float, default=0.0)                  # SAPPO λ; 0.0 = baseline PPO
    sentiment_source = Column(String(30), default="off")       # off/news/mock/xgb_proxy
    train_start = Column(String(8))                            # YYYYMMDD
    train_end = Column(String(8))
    val_start = Column(String(8))
    val_end = Column(String(8))
    n_episodes = Column(Integer, default=0)
    val_sharpe = Column(Float, default=0.0)
    val_return = Column(Float, default=0.0)                    # percent
    val_mdd = Column(Float, default=0.0)
    val_win_rate = Column(Float, default=0.0)
    val_avg_trades = Column(Float, default=0.0)
    model_path = Column(String(200), default="")
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now, index=True)


# ══════════════════════════════════════════════════════════════
# 4. Sentiment IC 스냅샷
# ══════════════════════════════════════════════════════════════
class SentimentICMetric(SappoBase):
    """sentiment → 미래수익률 IC 측정 결과.

    주기적으로 실행해 sentiment 가 실제 예측력을 갖는지 추적.
    """
    __tablename__ = "sappo_sentiment_ic"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_date = Column(String(8), nullable=False, index=True)  # 측정 실행 날짜
    forward_days = Column(Integer, nullable=False)             # 1/5/20
    window_start = Column(String(8))                            # 측정 데이터 범위
    window_end = Column(String(8))
    n_samples = Column(Integer, default=0)
    ic = Column(Float, default=0.0)                            # Spearman
    ic_pvalue = Column(Float, default=1.0)
    hit_rate = Column(Float, default=0.0)                      # score 부호 일치율
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)


# ══════════════════════════════════════════════════════════════
# 5. 주간 집계 (검증 리포트용)
# ══════════════════════════════════════════════════════════════
class SappoWeeklyMetric(SappoBase):
    """주간 SAPPO 검증 집계. week_start (월요일 YYYYMMDD) 당 1개."""
    __tablename__ = "sappo_weekly_metrics"

    week_start = Column(String(8), primary_key=True)
    n_news = Column(Integer, default=0)
    n_sentiment_scores = Column(Integer, default=0)
    avg_sentiment = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    sentiment_ic_5d = Column(Float, default=0.0)
    sentiment_ic_pvalue = Column(Float, default=1.0)
    sentiment_ic_n = Column(Integer, default=0)
    baseline_sharpe = Column(Float, default=0.0)               # λ=0 학습 최고 Sharpe
    best_sappo_sharpe = Column(Float, default=0.0)             # 이번 주 SAPPO 최고
    best_sappo_lambda = Column(Float, default=0.0)
    sharpe_improvement = Column(Float, default=0.0)            # best_sappo - baseline
    n_training_runs = Column(Integer, default=0)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


# ══════════════════════════════════════════════════════════════
# DB 초기화 / 세션
# ══════════════════════════════════════════════════════════════
_engine = None
_SessionFactory = None


def init_sappo_db(db_path: str = "data/trading.db") -> None:
    """SAPPO 테이블을 기존 trading.db 에 추가로 생성합니다 (공존 운영)."""
    global _engine, _SessionFactory
    if _engine is not None:
        return
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)

    @event.listens_for(_engine, "connect")
    def set_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    SappoBase.metadata.create_all(_engine, checkfirst=True)
    _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def get_sappo_session() -> Session:
    if _SessionFactory is None:
        init_sappo_db()
    return _SessionFactory()


# ══════════════════════════════════════════════════════════════
# 헬퍼 함수 (간단한 CRUD)
# ══════════════════════════════════════════════════════════════
def save_news(
    stock_code: str,
    date: str,
    source: str,
    title: str,
    body: str,
    url: str,
    published_at: datetime | None = None,
) -> NewsArticle | None:
    """뉴스 1건 저장. url 중복 시 None 반환."""
    session = get_sappo_session()
    try:
        existing = session.query(NewsArticle).filter_by(url=url).first()
        if existing:
            return None
        rec = NewsArticle(
            stock_code=stock_code, date=date, source=source,
            title=title[:500], body=body, url=url,
            published_at=published_at,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()


def get_news_for(stock_code: str, date: str) -> list[NewsArticle]:
    session = get_sappo_session()
    try:
        return (
            session.query(NewsArticle)
            .filter_by(stock_code=stock_code, date=date)
            .all()
        )
    finally:
        session.close()


def upsert_sentiment(
    stock_code: str,
    date: str,
    score: float,
    confidence: float = 0.0,
    n_articles: int = 0,
    rationale: str = "",
    model: str = "",
) -> SentimentScore:
    session = get_sappo_session()
    try:
        existing = (
            session.query(SentimentScore)
            .filter_by(stock_code=stock_code, date=date)
            .first()
        )
        if existing:
            existing.score = score
            existing.confidence = confidence
            existing.n_articles = n_articles
            existing.rationale = rationale[:2000]
            existing.model = model
            existing.generated_at = datetime.now()
            session.commit()
            return existing
        rec = SentimentScore(
            stock_code=stock_code, date=date, score=score,
            confidence=confidence, n_articles=n_articles,
            rationale=rationale[:2000], model=model,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()


def get_sentiment(stock_code: str, date: str) -> SentimentScore | None:
    session = get_sappo_session()
    try:
        return (
            session.query(SentimentScore)
            .filter_by(stock_code=stock_code, date=date)
            .first()
        )
    finally:
        session.close()


def get_sentiment_series(stock_code: str, start: str, end: str) -> list[SentimentScore]:
    """start~end 기간의 sentiment 시계열 반환 (date 오름차순)."""
    session = get_sappo_session()
    try:
        return (
            session.query(SentimentScore)
            .filter(
                SentimentScore.stock_code == stock_code,
                SentimentScore.date >= start,
                SentimentScore.date <= end,
            )
            .order_by(SentimentScore.date.asc())
            .all()
        )
    finally:
        session.close()


def save_training_run(
    run_name: str,
    lambda_value: float,
    sentiment_source: str,
    train_start: str,
    train_end: str,
    val_start: str,
    val_end: str,
    n_episodes: int,
    val_sharpe: float,
    val_return: float,
    val_mdd: float,
    val_win_rate: float,
    val_avg_trades: float,
    model_path: str,
    notes: str = "",
) -> SappoTrainingRun:
    session = get_sappo_session()
    try:
        rec = SappoTrainingRun(
            run_name=run_name, lambda_value=lambda_value,
            sentiment_source=sentiment_source,
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            n_episodes=n_episodes,
            val_sharpe=val_sharpe, val_return=val_return, val_mdd=val_mdd,
            val_win_rate=val_win_rate, val_avg_trades=val_avg_trades,
            model_path=model_path, notes=notes,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()


def save_ic_metric(
    forward_days: int,
    window_start: str,
    window_end: str,
    n_samples: int,
    ic: float,
    ic_pvalue: float,
    hit_rate: float,
    notes: str = "",
) -> SentimentICMetric:
    session = get_sappo_session()
    try:
        rec = SentimentICMetric(
            eval_date=datetime.now().strftime("%Y%m%d"),
            forward_days=forward_days,
            window_start=window_start, window_end=window_end,
            n_samples=n_samples, ic=ic, ic_pvalue=ic_pvalue,
            hit_rate=hit_rate, notes=notes,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()


def upsert_weekly_metric(
    week_start: str,
    **fields,
) -> SappoWeeklyMetric:
    """주간 레코드 upsert. week_start(월요일 YYYYMMDD) 기준."""
    session = get_sappo_session()
    try:
        existing = session.query(SappoWeeklyMetric).filter_by(week_start=week_start).first()
        if existing:
            for k, v in fields.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            session.commit()
            return existing
        rec = SappoWeeklyMetric(week_start=week_start, **fields)
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()
