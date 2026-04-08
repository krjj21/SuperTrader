"""
LLM 매매 시그널 검증 모듈
- XGBoost BUY/SELL 시그널을 LLM이 뉴스/기술적 맥락으로 검증
- 확정(confirm) 또는 보류(reject) 판단
- 장 종료 후 일일 매매 피드백 생성
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
from loguru import logger

from src.config import get_secrets


class SignalValidator:
    """LLM 기반 매매 시그널 검증"""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self):
        self.api_key = get_secrets().anthropic_api_key
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY 미설정 — LLM 검증 비활성화")

    @property
    def is_enabled(self) -> bool:
        return bool(self.api_key)

    def _build_context(self, stock_code: str, stock_name: str, df: pd.DataFrame) -> str:
        """OHLCV 데이터에서 기술적 컨텍스트를 구성합니다."""
        if len(df) < 20:
            return "데이터 부족"

        close = df["close"]
        volume = df["volume"]
        latest = close.iloc[-1]

        # 최근 수익률
        ret_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0

        # 이동평균
        ma5 = close.iloc[-5:].mean()
        ma20 = close.iloc[-20:].mean()
        ma60 = close.iloc[-60:].mean() if len(close) >= 60 else ma20

        # 거래량 변화
        vol_avg = volume.iloc[-20:].mean()
        vol_today = volume.iloc[-1]
        vol_ratio = vol_today / vol_avg if vol_avg > 0 else 1

        # RSI 간이 계산
        delta = close.diff().iloc[-14:]
        gain = delta.where(delta > 0, 0).mean()
        loss = (-delta.where(delta < 0, 0)).mean()
        rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 50

        # 최근 5일 가격
        recent_prices = ", ".join([f"{int(p):,}" for p in close.iloc[-5:]])

        context = (
            f"종목: {stock_name} ({stock_code})\n"
            f"현재가: {int(latest):,}원\n"
            f"최근 5일 종가: {recent_prices}\n"
            f"수익률: 1일 {ret_1d:+.1f}%, 5일 {ret_5d:+.1f}%, 20일 {ret_20d:+.1f}%\n"
            f"이동평균: MA5={int(ma5):,}, MA20={int(ma20):,}, MA60={int(ma60):,}\n"
            f"현재가 vs MA: {'MA5 위' if latest > ma5 else 'MA5 아래'}, "
            f"{'MA20 위' if latest > ma20 else 'MA20 아래'}, "
            f"{'MA60 위' if latest > ma60 else 'MA60 아래'}\n"
            f"RSI(14): {rsi:.0f}\n"
            f"거래량: 오늘 {int(vol_today):,} (20일 평균 대비 {vol_ratio:.1f}배)\n"
        )
        return context

    def validate_signal(
        self,
        stock_code: str,
        stock_name: str,
        signal: str,
        reason: str,
        df: pd.DataFrame,
    ) -> tuple[bool, str]:
        """ML 시그널을 LLM으로 검증합니다.

        Args:
            stock_code: 종목코드
            stock_name: 종목명
            signal: "BUY" 또는 "SELL"
            reason: ML 모델의 시그널 근거
            df: OHLCV DataFrame

        Returns:
            (confirmed: bool, explanation: str)
        """
        if not self.is_enabled:
            return True, "LLM 검증 비활성화 — 시그널 그대로 실행"

        context = self._build_context(stock_code, stock_name, df)

        system_prompt = """당신은 한국 주식 시장 전문 트레이딩 검증 에이전트입니다.
ML 모델이 생성한 매매 시그널을 기술적 맥락에서 검증합니다.

규칙:
1. 기술적 지표와 시그널의 일관성을 확인하세요
2. BUY 시그널 보류 조건 (하나라도 해당 시 보류):
   - RSI 80 이상 (과매수)
   - 현재가가 모든 이동평균(MA5, MA20, MA60) 아래에서 하락 추세
   - 거래량이 20일 평균 대비 0.3배 미만으로 급감
3. SELL 시그널은 리스크 관리 목적이므로 원칙적으로 확정하세요
   - 보류 가능한 유일한 경우: RSI 20 이하 과매도 + 거래량 급증(3배 이상) 반등 조짐
   - 상승 추세라는 이유만으로 SELL을 보류하지 마세요
4. 애매한 경우에는 ML 시그널을 존중하세요 (확정)
5. 반드시 첫 줄에 "확정" 또는 "보류" 중 하나만 쓰세요
6. 두 번째 줄부터 간단한 이유를 쓰세요 (3줄 이내)"""

        user_prompt = (
            f"ML 시그널: {signal}\n"
            f"ML 근거: {reason}\n\n"
            f"=== 기술적 컨텍스트 ===\n{context}"
        )

        try:
            with httpx.Client(timeout=15.0) as client:
                resp = client.post(
                    self.API_URL,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 200,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            content = data["content"][0]["text"].strip()
            lines = content.split("\n")
            first_line = lines[0].strip()

            confirmed = "확정" in first_line
            explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else first_line

            logger.info(
                f"LLM 검증 [{stock_name}] {signal}: "
                f"{'확정' if confirmed else '보류'} — {explanation[:50]}"
            )
            return confirmed, explanation

        except Exception as e:
            logger.warning(f"LLM 검증 실패 [{stock_name}]: {e} — 시그널 그대로 실행")
            return True, f"LLM 검증 오류: {e}"

    def generate_daily_feedback(
        self,
        trades: list,
        positions: list,
        total_pnl: int,
        total_pnl_pct: float,
        total_eval: int,
    ) -> str | None:
        """장 종료 후 당일 매매에 대한 LLM 피드백을 생성합니다.

        Args:
            trades: 당일 거래 내역 (TradeLog 리스트)
            positions: 현재 보유 포지션 (Position 리스트)
            total_pnl: 총 평가손익
            total_pnl_pct: 총 수익률(%)
            total_eval: 총 평가금액

        Returns:
            피드백 텍스트 또는 None (실패 시)
        """
        if not self.is_enabled:
            return None

        today = datetime.now().strftime("%Y-%m-%d")

        # 거래 내역 정리
        if trades:
            trade_lines = []
            for t in trades:
                trade_lines.append(
                    f"  - {t.side.upper()} {t.stock_name}({t.stock_code}) "
                    f"{t.quantity}주 × {t.price:,}원 | 근거: {t.signal_reason[:60]}"
                )
            trade_summary = "\n".join(trade_lines)
        else:
            trade_summary = "  (거래 없음)"

        # 보유 포지션 정리
        if positions:
            pos_lines = []
            winners, losers = 0, 0
            for p in positions:
                sign = "+" if p.pnl >= 0 else ""
                pos_lines.append(
                    f"  - {p.stock_name}({p.stock_code}) {p.quantity}주 "
                    f"평균 {p.avg_price:,} → 현재 {p.current_price:,} "
                    f"({sign}{p.pnl_pct:.1f}%)"
                )
                if p.pnl >= 0:
                    winners += 1
                else:
                    losers += 1
            pos_summary = "\n".join(pos_lines)
            win_lose = f"수익 {winners}종목 / 손실 {losers}종목"
        else:
            pos_summary = "  (보유 종목 없음)"
            win_lose = "보유 종목 없음"

        pnl_sign = "+" if total_pnl >= 0 else ""

        user_prompt = (
            f"=== {today} 매매 일지 ===\n\n"
            f"총 평가금액: {total_eval:,}원\n"
            f"총 평가손익: {pnl_sign}{total_pnl:,}원 ({pnl_sign}{total_pnl_pct:.2f}%)\n"
            f"승패: {win_lose}\n\n"
            f"[당일 거래]\n{trade_summary}\n\n"
            f"[보유 포지션]\n{pos_summary}"
        )

        system_prompt = """당신은 한국 주식 자동매매 시스템의 일일 리뷰어입니다.
오늘 하루의 매매 결과를 분석하여 피드백을 제공합니다.

반드시 다음 형식으로 작성하세요:

📊 오늘의 한 줄 평가
(한 문장으로 전체 요약)

✅ 잘한 점 (1~3개)
- 구체적인 매매 또는 판단에 대해

⚠️ 개선할 점 (1~3개)
- 구체적인 매매 또는 판단에 대해

💡 내일 전략 제안 (1~2개)
- 오늘 결과를 바탕으로 한 내일의 전략 힌트

규칙:
1. 거래가 없었으면 포지션 관리 관점에서 피드백하세요 (홀딩 판단, 손절/익절 타이밍 등)
2. 피상적인 칭찬이나 비판은 하지 마세요 — 구체적인 종목과 수치를 언급하세요
3. 200자 이내로 간결하게 작성하세요
4. 한국어로 작성하세요"""

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    self.API_URL,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 500,
                        "system": system_prompt,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            feedback = data["content"][0]["text"].strip()
            logger.info(f"일일 피드백 생성 완료 ({len(feedback)}자)")
            return feedback

        except Exception as e:
            logger.warning(f"일일 피드백 생성 실패: {e}")
            return None
