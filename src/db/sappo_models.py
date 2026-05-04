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
    Column, Integer, Float, String, Text, DateTime, Boolean,
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
# 6. Regime 라벨 (시장 국면)
# ══════════════════════════════════════════════════════════════
class RegimeLabel(SappoBase):
    """매일 장 시작 전 산출되는 시장 국면 라벨.
    label ∈ {risk_on_trend, high_vol_risk_off, mean_revert}
    """
    __tablename__ = "sappo_regime_labels"

    date = Column(String(8), primary_key=True)             # YYYYMMDD
    label = Column(String(20), nullable=False)
    hmm_state = Column(Integer, default=-1)                # GMM cluster idx (사후 라벨링 전)
    hmm_prob_risk_on = Column(Float, default=0.0)
    hmm_prob_risk_off = Column(Float, default=0.0)
    hmm_prob_revert = Column(Float, default=0.0)
    kospi_return_60d = Column(Float, default=0.0)          # 60일 누적 log-return
    kospi_vol_60d = Column(Float, default=0.0)             # 60일 일수익률 std (annualized)
    llm_score = Column(Float, nullable=True)               # 시장 뉴스 sentiment, -1~+1
    overridden_by_llm = Column(Boolean, default=False)     # LLM이 HMM 결과 강등/승격했는지
    notes = Column(Text, default="")
    generated_at = Column(DateTime, default=datetime.now)


# ══════════════════════════════════════════════════════════════
# 7. 외국인/기관/개인 매매 (KIS API FHKST01010900)
# ══════════════════════════════════════════════════════════════
class InvestorTrading(SappoBase):
    """종목별 투자자 매매 일별 데이터 (KIS API).

    Mid-cap 풀 선정 시 *외국인 20일 누적 순매수 상위 50%* 필터의 입력.
    """
    __tablename__ = "sappo_investor_trading"

    stock_code = Column(String(10), primary_key=True)
    date = Column(String(8), primary_key=True)              # YYYYMMDD
    close_price = Column(Integer, default=0)
    foreign_net_qty = Column(Integer, default=0)            # 외국인 순매수 수량
    foreign_net_amount = Column(Integer, default=0)         # 외국인 순매수 거래대금 (원)
    organ_net_qty = Column(Integer, default=0)              # 기관 순매수 수량
    organ_net_amount = Column(Integer, default=0)
    person_net_qty = Column(Integer, default=0)             # 개인 순매수 수량
    fetched_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index("idx_investor_date", "date"),
        Index("idx_investor_code_date", "stock_code", "date"),
    )


# ══════════════════════════════════════════════════════════════
# 8. 매크로 feature (USD/KRW + VIX) — A1+A2 라운드 6
# ══════════════════════════════════════════════════════════════
class MacroFeature(SappoBase):
    """일자별 매크로 지표 — Regime Detector 입력 보강용.

    sappo_regime_labels 와 1:1 (date PK) 관계. 별도 테이블로 둔 이유:
    - lifecycle 분리 (FDR 외부 의존성, 실패 시 KOSPI-only fallback)
    - 후일 macro feature 만 별도 분석할 때 join 만으로 가능
    """
    __tablename__ = "sappo_macro_features"

    date = Column(String(8), primary_key=True)              # YYYYMMDD
    usdkrw_close = Column(Float, default=0.0)
    usdkrw_log_ret = Column(Float, default=0.0)             # 1일 log-return
    usdkrw_vol_20d = Column(Float, default=0.0)             # 20일 rolling std
    vix_close = Column(Float, default=0.0)
    vix_log_ret = Column(Float, default=0.0)
    fetched_at = Column(DateTime, default=datetime.now)


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


def upsert_investor_trading(
    stock_code: str,
    date: str,
    close_price: int = 0,
    foreign_net_qty: int = 0,
    foreign_net_amount: int = 0,
    organ_net_qty: int = 0,
    organ_net_amount: int = 0,
    person_net_qty: int = 0,
) -> None:
    """투자자 매매 1건 upsert."""
    session = get_sappo_session()
    try:
        existing = (
            session.query(InvestorTrading)
            .filter_by(stock_code=stock_code, date=date)
            .first()
        )
        if existing:
            existing.close_price = close_price
            existing.foreign_net_qty = foreign_net_qty
            existing.foreign_net_amount = foreign_net_amount
            existing.organ_net_qty = organ_net_qty
            existing.organ_net_amount = organ_net_amount
            existing.person_net_qty = person_net_qty
            existing.fetched_at = datetime.now()
        else:
            rec = InvestorTrading(
                stock_code=stock_code, date=date,
                close_price=close_price,
                foreign_net_qty=foreign_net_qty,
                foreign_net_amount=foreign_net_amount,
                organ_net_qty=organ_net_qty,
                organ_net_amount=organ_net_amount,
                person_net_qty=person_net_qty,
            )
            session.add(rec)
        session.commit()
    finally:
        session.close()


def get_foreign_net_buy_cumulative(
    stock_code: str,
    end_date: str,
    days: int = 20,
) -> tuple[int, int]:
    """end_date 까지 직전 N영업일 외국인 순매수 누적값.

    Returns:
        (cum_amount, n_days_used) — 거래대금 누적(원), 실제 사용된 날짜 수
    """
    session = get_sappo_session()
    try:
        rows = (
            session.query(InvestorTrading)
            .filter(
                InvestorTrading.stock_code == stock_code,
                InvestorTrading.date <= end_date,
            )
            .order_by(InvestorTrading.date.desc())
            .limit(days)
            .all()
        )
        cum = sum(int(r.foreign_net_amount or 0) for r in rows)
        return cum, len(rows)
    finally:
        session.close()


def get_combined_net_buy_cumulative(
    stock_code: str,
    end_date: str,
    days: int = 20,
) -> tuple[int, int]:
    """end_date 까지 직전 N영업일 외국인+기관 순매수 누적값.

    한국 시장은 외국인·기관·개인 3축 — 외국인만 보면 신호의 절반만 활용.
    organ_net_amount 가 NULL 인 종목은 foreign 단독으로 fallback (graceful).

    Returns:
        (cum_amount, n_days_used)
    """
    session = get_sappo_session()
    try:
        rows = (
            session.query(InvestorTrading)
            .filter(
                InvestorTrading.stock_code == stock_code,
                InvestorTrading.date <= end_date,
            )
            .order_by(InvestorTrading.date.desc())
            .limit(days)
            .all()
        )
        cum = sum(
            int(r.foreign_net_amount or 0) + int(r.organ_net_amount or 0)
            for r in rows
        )
        return cum, len(rows)
    finally:
        session.close()


def upsert_regime_label(
    date: str,
    label: str,
    hmm_state: int,
    hmm_probs: tuple[float, float, float],
    kospi_return_60d: float,
    kospi_vol_60d: float,
    llm_score: float | None = None,
    overridden_by_llm: bool = False,
    notes: str = "",
) -> RegimeLabel:
    """오늘자 regime 라벨 upsert. hmm_probs 는 (risk_on, risk_off, revert) 순."""
    p_on, p_off, p_rev = hmm_probs
    session = get_sappo_session()
    try:
        existing = session.query(RegimeLabel).filter_by(date=date).first()
        if existing:
            existing.label = label
            existing.hmm_state = hmm_state
            existing.hmm_prob_risk_on = p_on
            existing.hmm_prob_risk_off = p_off
            existing.hmm_prob_revert = p_rev
            existing.kospi_return_60d = kospi_return_60d
            existing.kospi_vol_60d = kospi_vol_60d
            existing.llm_score = llm_score
            existing.overridden_by_llm = overridden_by_llm
            existing.notes = notes
            existing.generated_at = datetime.now()
            session.commit()
            return existing
        rec = RegimeLabel(
            date=date, label=label, hmm_state=hmm_state,
            hmm_prob_risk_on=p_on, hmm_prob_risk_off=p_off, hmm_prob_revert=p_rev,
            kospi_return_60d=kospi_return_60d, kospi_vol_60d=kospi_vol_60d,
            llm_score=llm_score, overridden_by_llm=overridden_by_llm, notes=notes,
        )
        session.add(rec)
        session.commit()
        session.refresh(rec)
        return rec
    finally:
        session.close()


def get_latest_regime() -> RegimeLabel | None:
    """가장 최근 regime 라벨 1건. 없으면 None."""
    session = get_sappo_session()
    try:
        return (
            session.query(RegimeLabel)
            .order_by(RegimeLabel.date.desc())
            .first()
        )
    finally:
        session.close()


def get_regime_for(date: str) -> RegimeLabel | None:
    session = get_sappo_session()
    try:
        return session.query(RegimeLabel).filter_by(date=date).first()
    finally:
        session.close()


def upsert_macro_feature(
    date: str,
    usdkrw_close: float = 0.0,
    usdkrw_log_ret: float = 0.0,
    usdkrw_vol_20d: float = 0.0,
    vix_close: float = 0.0,
    vix_log_ret: float = 0.0,
) -> None:
    """일자별 매크로 feature upsert."""
    session = get_sappo_session()
    try:
        existing = session.query(MacroFeature).filter_by(date=date).first()
        if existing:
            existing.usdkrw_close = float(usdkrw_close)
            existing.usdkrw_log_ret = float(usdkrw_log_ret)
            existing.usdkrw_vol_20d = float(usdkrw_vol_20d)
            existing.vix_close = float(vix_close)
            existing.vix_log_ret = float(vix_log_ret)
        else:
            session.add(MacroFeature(
                date=date,
                usdkrw_close=float(usdkrw_close),
                usdkrw_log_ret=float(usdkrw_log_ret),
                usdkrw_vol_20d=float(usdkrw_vol_20d),
                vix_close=float(vix_close),
                vix_log_ret=float(vix_log_ret),
            ))
        session.commit()
    finally:
        session.close()


def get_macro_features_window(end_date: str, days: int = 60) -> list[MacroFeature]:
    """end_date 까지 직전 N영업일 매크로 feature 시계열."""
    session = get_sappo_session()
    try:
        rows = (
            session.query(MacroFeature)
            .filter(MacroFeature.date <= end_date)
            .order_by(MacroFeature.date.desc())
            .limit(days)
            .all()
        )
        # 시간 오름차순 반환 (detector 가 시계열로 사용)
        rows.reverse()
        return rows
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
