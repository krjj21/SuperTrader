"""
LLM 매매 시그널 검증 모듈
- XGBoost BUY/SELL 시그널을 LLM이 뉴스/기술적 맥락으로 검증
- 확정(confirm) 또는 보류(reject) 판단
- 장 종료 후 일일 매매 피드백 생성

검증 구조 (프롬프트 재량 해석 위험 방지):
  1. OHLCV 에서 규칙용 컨텍스트 계산
  2. LLM 호출 (자연어 판정 + 설명)
  3. 코드 기반 규칙 재계산
  4. LLM 과 규칙 불일치 시 규칙 우선 (rule-based post-check)
"""
from __future__ import annotations

from datetime import datetime

import httpx
import pandas as pd
from loguru import logger

from src.config import get_secrets


# ══════════════════════════════════════════════════════════════
# 결정론적 규칙 판정 (LLM 과 독립적으로 실행, 최종 판단 기준)
# ══════════════════════════════════════════════════════════════
def _extract_context(df: pd.DataFrame) -> dict | None:
    """OHLCV 에서 규칙 판정용 수치 컨텍스트를 추출합니다."""
    if df is None or len(df) < 20 or "close" not in df.columns:
        return None
    close = df["close"]
    last = float(close.iloc[-1])
    if close.iloc[-2] <= 0:
        return None
    ret_1d = (last / float(close.iloc[-2]) - 1.0) * 100.0
    ret_5d = (last / float(close.iloc[-5]) - 1.0) * 100.0 if len(close) >= 5 else 0.0
    ma5 = float(close.iloc[-5:].mean())
    ma20 = float(close.iloc[-20:].mean())
    delta = close.diff().iloc[-14:]
    gain = delta.where(delta > 0, 0.0).mean()
    loss = (-delta.where(delta < 0, 0.0)).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss) if loss > 0 else 50.0
    return {
        "last": last, "ma5": ma5, "ma20": ma20, "rsi": rsi,
        "ret_1d": ret_1d, "ret_5d": ret_5d,
    }


def _rule_check_buy(ctx: dict) -> tuple[bool, str]:
    """BUY 규칙 판정. 반환: (should_hold, reason)."""
    if ctx["rsi"] >= 80.0 and ctx["ret_1d"] >= 7.0:
        return True, f"과열+급등 RSI={ctx['rsi']:.0f}/1d={ctx['ret_1d']:+.1f}%"
    if ctx["rsi"] >= 90.0:
        return True, f"극단 과매수 RSI={ctx['rsi']:.0f}"
    if ctx["ret_1d"] >= 15.0:
        return True, f"1일 급등 1d={ctx['ret_1d']:+.1f}%"
    if ctx["ret_5d"] >= 25.0:
        return True, f"5일 누적급등 5d={ctx['ret_5d']:+.1f}%"
    return False, (
        f"임계값 미충족 RSI={ctx['rsi']:.0f} "
        f"1d={ctx['ret_1d']:+.1f}% 5d={ctx['ret_5d']:+.1f}%"
    )


def _rule_check_sell(ctx: dict) -> tuple[bool, str]:
    """SELL 규칙 판정. 반환: (should_hold, reason)."""
    if (
        ctx["last"] > ctx["ma20"]
        and ctx["last"] > ctx["ma5"]
        and ctx["rsi"] >= 50.0
        and ctx["ret_1d"] >= -1.0
        and ctx["ret_5d"] >= 1.0
    ):
        return True, (
            f"상승추세 유지 MA5/20 위 "
            f"RSI={ctx['rsi']:.0f} 1d={ctx['ret_1d']:+.1f}% 5d={ctx['ret_5d']:+.1f}%"
        )
    return False, (
        f"5조건 미충족 RSI={ctx['rsi']:.0f} "
        f"1d={ctx['ret_1d']:+.1f}% 5d={ctx['ret_5d']:+.1f}%"
    )


def apply_rule_check(signal: str, df: pd.DataFrame) -> tuple[bool, str] | None:
    """신호(BUY/SELL) 에 대해 규칙 판정을 반환합니다. (confirmed, reason) 또는 None (데이터 부족)."""
    ctx = _extract_context(df)
    if ctx is None:
        return None
    sig = signal.upper()
    if sig == "BUY":
        hold, reason = _rule_check_buy(ctx)
    elif sig == "SELL":
        hold, reason = _rule_check_sell(ctx)
    else:
        return True, "해당 없음"
    return (not hold), reason


class MockSignalValidator:
    """LLM 검증의 결정론적 시뮬레이션 (백테스트 비교용).

    SignalValidator 와 동일한 규칙 함수를 사용해 판정.
    """

    is_enabled = True

    def validate_signal(
        self,
        stock_code: str,
        stock_name: str,
        signal: str,
        reason: str,
        df: pd.DataFrame,
    ) -> tuple[bool, str]:
        result = apply_rule_check(signal, df)
        if result is None:
            return True, "데이터 부족 → 확정 (mock)"
        confirmed, rule_reason = result
        prefix = "확정" if confirmed else "보류"
        return confirmed, f"{prefix} — {rule_reason} (mock)"


class SignalValidator:
    """LLM 기반 매매 시그널 검증"""

    API_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str | None = None):
        self.api_key = get_secrets().anthropic_api_key
        self.model = model or self.DEFAULT_MODEL
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
PPO RL 모델이 생성한 매매 시그널을 기술적 맥락에서 최종 검증합니다.

핵심 원칙: 보수적 검증. 과열 매수는 차단하고, 약세 신호가 보이면 매도 확정. 손실 관리 우선.

[BUY 보류 조건 — 아래 중 하나라도 해당하면 보류]
   - RSI ≥ 80 AND 1일 수익률 ≥ +7% (과열 동반 급등)
   - RSI ≥ 90 (극단적 과매수, 단독)
   - 1일 수익률 ≥ +15% (단일일 급등)
   - 5일 수익률 ≥ +25% (단기 급등 누적)

[SELL 보류 조건 — 아래 5조건을 모두 충족하는 명백한 "상승추세 중 일시 눌림" 만 보류. 하나라도 미충족이면 확정]
   - 현재가 > MA20 (중기 상승추세) AND
   - 현재가 > MA5 (단기 상승추세) AND
   - RSI ≥ 50 (모멘텀 중립 이상) AND
   - 1일 수익률 ≥ -1% (거의 보합) AND
   - 5일 수익률 ≥ +1% (5일간 상승 유지)

SELL 확정 사유 예시 (하나라도 해당되면 확정):
   - MA5/MA20 이탈, RSI < 50, 1일 -1% 이하 하락, 5일 +1% 미만, 거래량 급증 + 음봉, 추세 약화

[엄격 해석 원칙 — 반드시 준수]
- 위에 명시된 임계값(RSI 80/90, 1일 +7/+15%, 5일 +25%, MA5/MA20, RSI 50, 1일 -1%, 5일 +1%) 외의 지표로 보류하지 마세요.
- "근접", "초입", "접근", "20일 누적", "거래량 동반" 등 재량 표현으로 임계값 미달 상황을 보류하지 마세요.
- 임계값에 정확히 도달하지 않았으면 "조건 미충족" 으로 확정입니다.
- BUY 는 네 보류 조건 중 하나라도 수치상 충족할 때만 보류, 미충족이면 확정.
- SELL 은 다섯 보류 조건 모두 수치상 충족할 때만 보류, 하나라도 미충족이면 확정.

판단 기준:
- 애매하면 "BUY 는 보류, SELL 은 확정" 쪽으로 (리스크 최소화)
- RSI 70~80 구간은 아직 과열 전이므로 다른 조건 확인 후 판단
- 상승추세 중 경미한 눌림이 아닌 이상 SELL 은 확정

반드시 첫 줄에 "확정" 또는 "보류" 중 하나만 쓰세요.
두 번째 줄부터 간단한 이유를 쓰세요 (2줄 이내)."""

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
                        "model": self.model,
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

            llm_confirmed = "확정" in first_line
            llm_explanation = "\n".join(lines[1:]).strip() if len(lines) > 1 else first_line

            display = f"{stock_code}|{stock_name or '-'}"

            # ── Rule-based post-check: LLM 의 재량 해석을 결정론적 규칙으로 보정 ──
            rule_result = apply_rule_check(signal, df)
            if rule_result is None:
                logger.info(
                    f"LLM 검증 [{display}] {signal}: "
                    f"{'확정' if llm_confirmed else '보류'} — {llm_explanation[:50]} (규칙 스킵: 데이터 부족)"
                )
                return llm_confirmed, llm_explanation

            rule_confirmed, rule_reason = rule_result
            if rule_confirmed != llm_confirmed:
                # 불일치 → 규칙 우선
                logger.warning(
                    f"LLM/규칙 불일치 [{display}] {signal}: "
                    f"LLM={'확정' if llm_confirmed else '보류'} "
                    f"→ 규칙={'확정' if rule_confirmed else '보류'} "
                    f"(규칙: {rule_reason} | LLM: {llm_explanation[:60]})"
                )
                final_explanation = f"[규칙 {rule_reason}] [LLM {llm_explanation[:80]}]"
                return rule_confirmed, final_explanation

            # 일치 → LLM 설명 유지
            logger.info(
                f"LLM 검증 [{display}] {signal}: "
                f"{'확정' if llm_confirmed else '보류'} — {llm_explanation[:50]}"
            )
            return llm_confirmed, llm_explanation

        except Exception as e:
            display = f"{stock_code}|{stock_name or '-'}"
            # LLM 실패 시에도 규칙 기반으로 판정
            rule_result = apply_rule_check(signal, df)
            if rule_result is not None:
                rule_confirmed, rule_reason = rule_result
                logger.warning(
                    f"LLM 검증 실패 [{display}] → 규칙 기반 판정: "
                    f"{'확정' if rule_confirmed else '보류'} ({rule_reason}) err={e}"
                )
                return rule_confirmed, f"LLM 오류→규칙: {rule_reason}"
            logger.warning(f"LLM 검증 실패 [{display}]: {e} — 시그널 그대로 실행")
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
                        "model": self.model,
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
