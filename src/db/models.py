"""
데이터베이스 모듈
- SQLAlchemy 기반 ORM 모델
- 거래 이력, 포지션, 일일 수익 기록
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean,
    create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────────
class TradeLog(Base):
    """거래 이력"""
    __tablename__ = "trade_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), default="")
    side = Column(String(4), nullable=False)           # buy / sell
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    amount = Column(Integer, nullable=False)            # 거래금액
    order_no = Column(String(20), default="")
    strategy = Column(String(50), default="")
    signal_strength = Column(Float, default=0.0)
    signal_reason = Column(String(200), default="")
    stop_loss = Column(Integer, default=0)
    status = Column(String(20), default="filled")       # filled, cancelled, failed
    error_msg = Column(String(200), default="")
    created_at = Column(DateTime, default=datetime.now)


class DailyPnL(Base):
    """일일 손익"""
    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True, index=True)  # YYYYMMDD
    total_eval = Column(Integer, default=0)
    total_deposit = Column(Integer, default=0)
    total_pnl = Column(Integer, default=0)
    total_pnl_pct = Column(Float, default=0.0)
    num_positions = Column(Integer, default=0)
    num_trades = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class PositionLog(Base):
    """보유 포지션 스냅샷"""
    __tablename__ = "position_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, index=True)
    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(50), default="")
    quantity = Column(Integer, default=0)
    avg_price = Column(Integer, default=0)
    current_price = Column(Integer, default=0)
    pnl = Column(Integer, default=0)
    pnl_pct = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.now)


class SignalLog(Base):
    """시그널 이력 (LLM 검증 결과 포함)"""
    __tablename__ = "signal_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), default="")
    signal = Column(String(4), nullable=False)          # BUY / SELL
    decision = Column(String(10), nullable=False)        # 확정 / 보류
    reason = Column(String(1000), default="")
    signal_type = Column(String(10), default="llm")      # llm / summary
    created_at = Column(DateTime, default=datetime.now, index=True)


class RuntimeStatus(Base):
    """대시보드용 런타임 상태 스냅샷"""
    __tablename__ = "runtime_status"

    id = Column(Integer, primary_key=True, autoincrement=False)
    strategy = Column(String(50), default="")
    pool_size = Column(Integer, default=0)
    llm_enabled = Column(Boolean, default=False)
    kill_switch = Column(Boolean, default=False)
    check_interval = Column(Integer, default=0)
    daily_loss_limit = Column(Float, default=0.0)
    max_positions = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.now)


# ──────────────────────────────────────────────
# 데이터베이스 초기화
# ──────────────────────────────────────────────
_engine = None
_SessionFactory = None


def init_db(db_path: str = "data/trading.db") -> None:
    """데이터베이스를 초기화합니다."""
    global _engine, _SessionFactory

    if _engine is not None:
        return

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # WAL 모드 활성화 (동시 읽기/쓰기 성능)
    @event.listens_for(_engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Base.metadata.create_all(_engine, checkfirst=True)
    _SessionFactory = sessionmaker(bind=_engine)


def get_session() -> Session:
    """세션을 반환합니다."""
    if _SessionFactory is None:
        init_db()
    return _SessionFactory()


# ──────────────────────────────────────────────
# 리포지토리 함수
# ──────────────────────────────────────────────
def save_trade(
    stock_code: str,
    stock_name: str,
    side: str,
    quantity: int,
    price: int,
    order_no: str = "",
    strategy: str = "",
    signal_strength: float = 0.0,
    signal_reason: str = "",
    stop_loss: int = 0,
    status: str = "filled",
    error_msg: str = "",
) -> TradeLog:
    """거래를 기록합니다."""
    session = get_session()
    try:
        trade = TradeLog(
            stock_code=stock_code,
            stock_name=stock_name,
            side=side,
            quantity=quantity,
            price=price,
            amount=quantity * price,
            order_no=order_no,
            strategy=strategy,
            signal_strength=signal_strength,
            signal_reason=signal_reason,
            stop_loss=stop_loss,
            status=status,
            error_msg=error_msg,
        )
        session.add(trade)
        session.commit()
        session.refresh(trade)
        return trade
    finally:
        session.close()


def save_daily_pnl(
    total_eval: int,
    total_deposit: int,
    total_pnl: int,
    total_pnl_pct: float,
    num_positions: int,
    num_trades: int = 0,
) -> DailyPnL:
    """일일 손익을 기록합니다."""
    session = get_session()
    today = datetime.now().strftime("%Y%m%d")
    try:
        existing = session.query(DailyPnL).filter_by(date=today).first()
        if existing:
            existing.total_eval = total_eval
            existing.total_deposit = total_deposit
            existing.total_pnl = total_pnl
            existing.total_pnl_pct = total_pnl_pct
            existing.num_positions = num_positions
            existing.num_trades = num_trades
            session.commit()
            return existing

        record = DailyPnL(
            date=today,
            total_eval=total_eval,
            total_deposit=total_deposit,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            num_positions=num_positions,
            num_trades=num_trades,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def get_today_trades() -> list[TradeLog]:
    """오늘의 거래 내역을 조회합니다."""
    session = get_session()
    today = datetime.now().strftime("%Y%m%d")
    try:
        trades = (
            session.query(TradeLog)
            .filter(TradeLog.created_at >= datetime.strptime(today, "%Y%m%d"))
            .order_by(TradeLog.created_at.desc())
            .all()
        )
        return trades
    finally:
        session.close()


def save_runtime_status(
    strategy: str,
    pool_size: int,
    llm_enabled: bool,
    kill_switch: bool,
    check_interval: int,
    daily_loss_limit: float,
    max_positions: int,
) -> RuntimeStatus:
    """대시보드용 런타임 상태를 저장합니다."""
    session = get_session()
    try:
        status = session.get(RuntimeStatus, 1)
        if status is None:
            status = RuntimeStatus(id=1)
            session.add(status)

        status.strategy = strategy
        status.pool_size = pool_size
        status.llm_enabled = llm_enabled
        status.kill_switch = kill_switch
        status.check_interval = check_interval
        status.daily_loss_limit = daily_loss_limit
        status.max_positions = max_positions
        status.updated_at = datetime.now()

        session.commit()
        session.refresh(status)
        return status
    finally:
        session.close()


def get_runtime_status() -> RuntimeStatus | None:
    """가장 최근 런타임 상태를 조회합니다."""
    session = get_session()
    try:
        return session.get(RuntimeStatus, 1)
    finally:
        session.close()


def save_signal_log(
    stock_code: str,
    stock_name: str,
    signal: str,
    decision: str,
    reason: str = "",
    signal_type: str = "llm",
) -> SignalLog:
    """시그널 이력을 기록합니다."""
    session = get_session()
    try:
        log = SignalLog(
            stock_code=stock_code,
            stock_name=stock_name,
            signal=signal,
            decision=decision,
            reason=reason[:1000],
            signal_type=signal_type,
        )
        session.add(log)
        session.commit()
        session.refresh(log)
        return log
    finally:
        session.close()


def get_recent_signals(limit: int = 100, days: int = 7) -> list[SignalLog]:
    """최근 N일 이내의 시그널 이력을 조회합니다."""
    from datetime import timedelta
    session = get_session()
    try:
        cutoff = datetime.now() - timedelta(days=days)
        return (
            session.query(SignalLog)
            .filter(SignalLog.created_at >= cutoff)
            .order_by(SignalLog.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()
