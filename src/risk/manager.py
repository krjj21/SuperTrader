"""
리스크 관리 모듈
- 포지션 사이징 (ATR 기반)
- 종목별/일일 손실 한도
- 비상 정지 (kill switch)
"""
from __future__ import annotations

from datetime import datetime

from loguru import logger

from src.broker.account import AccountManager, AccountSummary
from src.config import get_config, RiskConfig
from src.strategy.base import TradeSignal, Signal


class RiskManager:
    """리스크 관리"""

    def __init__(self, account: AccountManager | None = None):
        self.config = get_config().risk
        schedule = get_config().schedule
        self._market_open = datetime.strptime(schedule.market_open, "%H:%M").time()
        self._market_close = datetime.strptime(schedule.market_close, "%H:%M").time()
        self.account = account or AccountManager()
        self._daily_pnl: float = 0.0
        self._daily_date: str = ""
        self._kill_switch: bool = False
        self._error_count: int = 0

    @property
    def is_trading_allowed(self) -> bool:
        """매매가 허용되는지 확인합니다 (kill switch + 장 운영 시간)."""
        if self._kill_switch:
            logger.warning("🚨 Kill switch 활성화 - 매매 중지")
            return False

        now = datetime.now()
        current_time = now.time()

        if current_time < self._market_open or current_time > self._market_close:
            logger.debug(f"장 운영 시간 외: {current_time} (운영: {self._market_open}~{self._market_close})")
            return False

        # 주말 체크 (토=5, 일=6)
        if now.weekday() >= 5:
            logger.debug("주말 — 매매 중지")
            return False

        return True

    def activate_kill_switch(self, reason: str = "") -> None:
        """비상 정지를 활성화합니다."""
        self._kill_switch = True
        logger.critical(f"🚨 KILL SWITCH 활성화: {reason}")

    def deactivate_kill_switch(self) -> None:
        """비상 정지를 해제합니다."""
        self._kill_switch = False
        self._error_count = 0
        logger.info("✅ Kill switch 해제")

    def record_error(self) -> None:
        """에러를 기록하고, 연속 3회 시 kill switch를 활성화합니다."""
        self._error_count += 1
        if self._error_count >= 3:
            self.activate_kill_switch(f"연속 에러 {self._error_count}회")

    def reset_error_count(self) -> None:
        self._error_count = 0

    def check_daily_loss_limit(self, current_pnl: float, total_assets: float) -> bool:
        """일일 손실 한도를 확인합니다."""
        today = datetime.now().strftime("%Y%m%d")
        if self._daily_date != today:
            self._daily_pnl = 0.0
            self._daily_date = today

        loss_pct = abs(current_pnl) / total_assets if total_assets > 0 else 0.0
        if current_pnl < 0 and loss_pct >= self.config.daily_loss_limit_pct:
            self.activate_kill_switch(
                f"일일 손실 한도 초과: {loss_pct:.2%} >= {self.config.daily_loss_limit_pct:.2%}"
            )
            return False
        return True

    def check_stop_loss(self, positions: list) -> list[str]:
        """손절 임계값(-stop_loss_pct)을 초과한 종목코드 리스트를 반환합니다.

        Args:
            positions: AccountSummary.positions

        Returns:
            손절해야 할 종목코드 리스트
        """
        threshold_pct = -round(self.config.stop_loss_pct * 100, 4)  # 예: -7.0
        stop_codes = []
        for pos in positions:
            if pos.pnl_pct <= threshold_pct:
                stop_codes.append(pos.stock_code)
                logger.warning(
                    f"손절 트리거: {pos.stock_name}({pos.stock_code}) "
                    f"{pos.pnl_pct:.2f}% ≤ {threshold_pct:.1f}%"
                )
        return stop_codes

    def calculate_position_size(
        self,
        signal: TradeSignal,
        available_cash: int,
        total_assets: int,
        current_positions: int,
    ) -> int:
        """포지션 사이즈(수량)를 계산합니다.

        Args:
            signal: 매매 시그널
            available_cash: 주문 가능 예수금
            total_assets: 총 평가금액
            current_positions: 현재 보유 종목 수

        Returns:
            매수 수량 (0이면 매수 불가)
        """
        if not self.is_trading_allowed:
            return 0

        if signal.signal != Signal.BUY or signal.price <= 0:
            return 0

        # 최대 동시 보유 종목 수 체크
        if current_positions >= self.config.max_total_positions:
            logger.info(
                f"최대 보유 종목 수 초과: {current_positions} >= {self.config.max_total_positions}"
            )
            return 0

        # 종목당 최대 투자 금액
        max_amount = int(total_assets * self.config.max_position_pct)

        # 실제 투자 가능 금액 (예수금과 비교)
        invest_amount = min(max_amount, available_cash)

        # ATR 기반 사이즈 조정 (손절까지의 위험 금액 고려)
        if signal.stop_loss > 0 and signal.price > signal.stop_loss:
            risk_per_share = signal.price - signal.stop_loss
            # 총 자산의 1%를 1회 거래 최대 위험으로 설정
            max_risk = total_assets * 0.01
            atr_based_qty = int(max_risk / risk_per_share)

            # 금액 기반 수량
            amount_based_qty = invest_amount // signal.price

            # 둘 중 작은 값
            quantity = min(atr_based_qty, amount_based_qty)
        else:
            quantity = invest_amount // signal.price

        # 최소 1주
        quantity = max(quantity, 0)

        if quantity > 0:
            logger.info(
                f"포지션 사이즈 계산: {signal.stock_code} "
                f"{quantity}주 × {signal.price:,}원 = {quantity * signal.price:,}원 "
                f"(한도: {max_amount:,}원)"
            )

        return quantity

    def validate_order(
        self,
        signal: TradeSignal,
        quantity: int,
        balance: AccountSummary,
    ) -> tuple[bool, str]:
        """주문 전 최종 검증을 수행합니다.

        Returns:
            (통과 여부, 사유)
        """
        if not self.is_trading_allowed:
            return False, "Kill switch 활성화 상태"

        if quantity <= 0:
            return False, "수량이 0 이하"

        if signal.signal == Signal.BUY:
            required_amount = signal.price * quantity
            if required_amount > balance.total_deposit:
                return False, f"예수금 부족: 필요 {required_amount:,}원 > 보유 {balance.total_deposit:,}원"

            # 일일 손실 한도 재확인
            if not self.check_daily_loss_limit(balance.total_pnl, balance.total_eval):
                return False, "일일 손실 한도 초과"

        return True, "OK"
