"""
Slack 알림 모듈
- 매매 체결 알림
- 에러/경고 알림
- 일일 리포트
- Kill switch 알림
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger

from src.broker.account import AccountSummary, Position
from src.broker.order import Order, OrderSide
from src.config import get_secrets
from src.strategy.base import TradeSignal, Signal


class SlackNotifier:
    """Slack 알림 발송"""

    def __init__(self):
        secrets = get_secrets()
        self.token = secrets.slack_bot_token
        self.channel = secrets.slack_channel
        # 매매 전용 채널 (체결/시그널/손절/주문실패). 미설정이면 기본 채널 fallback.
        self.trade_channel = secrets.slack_trade_channel or self.channel
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from slack_sdk import WebClient
                self._client = WebClient(token=self.token)
            except ImportError:
                logger.warning("slack-sdk가 설치되지 않았습니다. pip install slack-sdk")
        return self._client

    def _send(self, text: str, blocks: list[dict] | None = None, channel: str | None = None) -> bool:
        """메시지를 전송합니다. channel 미지정 시 self.channel 사용."""
        if not self.token:
            logger.debug(f"[Slack 미설정] {text}")
            return False

        target = channel or self.channel
        try:
            if self.client:
                self.client.chat_postMessage(
                    channel=target,
                    text=text,
                    blocks=blocks,
                )
                return True
        except Exception as e:
            logger.error(f"Slack 전송 실패 ({target}): {e}")
        return False

    # ──────────────────────────────────────────
    # 매매 알림
    # ──────────────────────────────────────────
    def notify_signal(self, signal: TradeSignal) -> bool:
        """매매 시그널 발생 알림"""
        if signal.signal == Signal.HOLD:
            return False

        emoji = "🟢" if signal.signal == Signal.BUY else "🔴"
        action = "매수" if signal.signal == Signal.BUY else "매도"

        text = (
            f"{emoji} *{action} 시그널* | {signal.stock_name} ({signal.stock_code})\n"
            f"• 가격: {signal.price:,}원\n"
            f"• 강도: {signal.strength:.0%}\n"
            f"• 근거: {signal.reason}\n"
        )
        if signal.stop_loss:
            text += f"• 손절가: {signal.stop_loss:,}원\n"

        return self._send(text, channel=self.trade_channel)

    def notify_order_filled(self, order: Order) -> bool:
        """체결 완료 알림"""
        emoji = "✅" if order.side == OrderSide.BUY else "💰"
        action = "매수" if order.side == OrderSide.BUY else "매도"

        text = (
            f"{emoji} *{action} 체결*\n"
            f"• 종목: {order.stock_code}\n"
            f"• 수량: {order.filled_qty:,}주\n"
            f"• 체결가: {order.filled_price:,}원\n"
            f"• 금액: {order.filled_qty * order.filled_price:,}원\n"
            f"• 주문번호: {order.order_no}"
        )
        return self._send(text, channel=self.trade_channel)

    def notify_order_failed(self, order: Order) -> bool:
        """주문 실패 알림"""
        text = (
            f"⚠️ *주문 실패*\n"
            f"• 종목: {order.stock_code}\n"
            f"• 사유: {order.error_msg}"
        )
        return self._send(text, channel=self.trade_channel)

    # ──────────────────────────────────────────
    # 시스템 알림
    # ──────────────────────────────────────────
    def notify_error(self, error: str, context: str = "") -> bool:
        """에러 알림"""
        text = f"🚨 *시스템 에러*\n• {error}"
        if context:
            text += f"\n• 컨텍스트: {context}"
        return self._send(text)

    def notify_stop_loss(self, position) -> bool:
        """손절 매도 알림"""
        text = (
            f"🛑 *손절 매도* | {position.stock_name} ({position.stock_code})\n"
            f"• 손익률: {position.pnl_pct:.2f}%\n"
            f"• 손익: {position.pnl:,}원\n"
            f"• 매입가→현재가: {position.avg_price:,} → {position.current_price:,}원\n"
            f"• 수량: {position.quantity:,}주"
        )
        return self._send(text, channel=self.trade_channel)

    def notify_kill_switch(self, reason: str) -> bool:
        """Kill switch 활성화 알림"""
        text = (
            f"🛑 *KILL SWITCH 활성화*\n"
            f"• 사유: {reason}\n"
            f"• 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• 모든 자동 매매가 중지되었습니다."
        )
        return self._send(text)

    def notify_start(self) -> bool:
        """시스템 시작 알림"""
        text = f"🚀 *자동매매 시스템 시작* | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return self._send(text)

    # ──────────────────────────────────────────
    # 리포트
    # ──────────────────────────────────────────
    def notify_daily_report(self, summary: AccountSummary) -> bool:
        """일일 리포트"""
        pnl_emoji = "📈" if summary.total_pnl >= 0 else "📉"

        lines = [
            f"{pnl_emoji} *일일 리포트* | {datetime.now().strftime('%Y-%m-%d')}",
            f"",
            f"• 총 평가금액: {summary.total_eval:,}원",
            f"• 예수금: {summary.total_deposit:,}원",
            f"• 평가손익: {summary.total_pnl:,}원 ({summary.total_pnl_pct:.2f}%)",
            f"• 보유 종목: {len(summary.positions)}개",
            f"",
        ]

        if summary.positions:
            lines.append("*보유 종목:*")
            for pos in summary.positions:
                pnl_sign = "+" if pos.pnl >= 0 else ""
                lines.append(
                    f"  • {pos.stock_name} ({pos.stock_code}): "
                    f"{pos.quantity}주 @ {pos.avg_price:,}원 → "
                    f"{pos.current_price:,}원 "
                    f"({pnl_sign}{pos.pnl:,}원, {pnl_sign}{pos.pnl_pct:.1f}%)"
                )

        text = "\n".join(lines)
        return self._send(text)

    def notify_daily_feedback(self, feedback: str) -> bool:
        """LLM 일일 매매 피드백 전송"""
        text = f"🤖 *AI 일일 매매 피드백*\n\n{feedback}"
        return self._send(text)
