"""
Codex CLI 기반 리뷰 파이프라인

세 가지 모드를 지원한다:
- daily:    오늘자 (또는 지정일) 매매 결과를 DB에서 수집 → codex exec 리뷰
- backtest: 최신 백테스트 로그 + 비교 테이블을 codex exec 에 전달
- code:     변경 diff 를 codex review 에 전달 (기본 --uncommitted)

결과는 reports/codex_<mode>_<YYYY-MM-DD>.md 로 저장되며,
동일한 형식으로 라이브/백테스트/수동 호출 모두에서 재사용된다.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"


# ══════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════
def _output_path(mode: str, date: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORTS_DIR / f"codex_{mode}_{date}.md"


def _run_codex_exec(prompt: str, stdin_payload: str, output_file: Path, model: str | None) -> int:
    """codex exec 호출. stdin_payload 는 <stdin> 블록으로 prompt 에 붙는다."""
    cmd = ["codex", "exec", "--sandbox", "read-only", "-o", str(output_file)]
    if model:
        cmd.extend(["-m", model])
    cmd.append(prompt)

    print(f"[codex_review] 실행: {' '.join(cmd[:6])} ... (model={model or 'default'})")
    result = subprocess.run(
        cmd,
        input=stdin_payload,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode


def _run_codex_review(prompt: str, output_file: Path, base: str | None, uncommitted: bool, commit: str | None, model: str | None) -> int:
    """codex review 호출 — 전용 서브커맨드 사용."""
    cmd = ["codex", "review"]
    if model:
        cmd.extend(["-c", f"model={model}"])
    if uncommitted:
        cmd.append("--uncommitted")
    if base:
        cmd.extend(["--base", base])
    if commit:
        cmd.extend(["--commit", commit])
    cmd.append(prompt)

    print(f"[codex_review] 실행: {' '.join(cmd)}")
    # codex review 는 -o 옵션이 없어 stdout 을 캡처한다
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output_file.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result.returncode


# ══════════════════════════════════════════════════════════════
# daily 모드
# ══════════════════════════════════════════════════════════════
DAILY_PROMPT = """너는 퀀트 자동매매 시스템의 데일리 리뷰어다.

아래 <stdin> 에는 오늘자 매매 결과가 JSON 으로 담겨 있다. 다음 관점으로 한국어 마크다운 리포트를 작성해라:

1. **총평** (한 줄): 오늘 매매를 한 문장으로 요약
2. **핵심 지표**: 평가금액/손익/보유종목/거래건수 — 전일 대비 해석을 붙여라
3. **거래 분석**: 매수/매도 각각의 근거(signal_reason)를 보고, 과도한 확신이나 일관성 없는 신호가 있으면 지적
4. **포지션 건강도**: 집중도, 손실 중인 종목, 보유기간 리스크
5. **LLM 검증 결과 평가**: 시그널이 보류된 이유에 타당성이 있었는지 (signal_logs 의 reason 참조)
6. **내일을 위한 제안**: 구체적 액션 아이템 3개

간결하게. 각 섹션은 요점만. 감상평은 금지, 데이터 근거만 써라.
출력은 순수 마크다운. 다른 설명이나 메타코멘트 없이.
"""


def _collect_daily_data(date: str) -> dict:
    """DB 에서 해당 날짜의 매매 결과를 수집한다."""
    from src.config import load_config
    from src.db.models import (
        DailyPnL, TradeLog, PositionLog, HoldingPosition, SignalLog,
        get_session, init_db,
    )

    load_config(str(PROJECT_ROOT / "config" / "settings.yaml"))
    init_db("data/trading.db")
    session = get_session()

    # 날짜 범위
    date_start = datetime.strptime(date, "%Y%m%d")
    date_end = date_start + timedelta(days=1)

    try:
        pnl = session.query(DailyPnL).filter_by(date=date).first()
        prev_pnl = (
            session.query(DailyPnL)
            .filter(DailyPnL.date < date)
            .order_by(DailyPnL.date.desc())
            .first()
        )

        trades = (
            session.query(TradeLog)
            .filter(TradeLog.created_at >= date_start, TradeLog.created_at < date_end)
            .order_by(TradeLog.created_at.asc())
            .all()
        )

        holdings = session.query(HoldingPosition).all()

        positions = (
            session.query(PositionLog)
            .filter_by(date=date)
            .all()
        )

        signals = (
            session.query(SignalLog)
            .filter(SignalLog.created_at >= date_start, SignalLog.created_at < date_end)
            .order_by(SignalLog.created_at.asc())
            .all()
        )

        def pnl_dict(row):
            if row is None:
                return None
            return {
                "date": row.date,
                "total_eval": row.total_eval,
                "total_deposit": row.total_deposit,
                "total_pnl": row.total_pnl,
                "total_pnl_pct": row.total_pnl_pct,
                "num_positions": row.num_positions,
                "num_trades": row.num_trades,
            }

        return {
            "date": date,
            "pnl": pnl_dict(pnl),
            "prev_pnl": pnl_dict(prev_pnl),
            "trades": [
                {
                    "time": t.created_at.strftime("%H:%M:%S"),
                    "code": t.stock_code,
                    "name": t.stock_name,
                    "side": t.side,
                    "qty": t.quantity,
                    "price": t.price,
                    "amount": t.amount,
                    "strategy": t.strategy,
                    "strength": t.signal_strength,
                    "reason": t.signal_reason,
                    "status": t.status,
                }
                for t in trades
            ],
            "holdings": [
                {
                    "code": h.stock_code,
                    "name": h.stock_name,
                    "avg_price": h.avg_price,
                    "qty": h.quantity,
                    "buy_date": h.buy_date,
                }
                for h in holdings
            ],
            "positions": [
                {
                    "code": p.stock_code,
                    "name": p.stock_name,
                    "qty": p.quantity,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "pnl": p.pnl,
                    "pnl_pct": p.pnl_pct,
                }
                for p in positions
            ],
            "signals": [
                {
                    "time": s.created_at.strftime("%H:%M:%S"),
                    "code": s.stock_code,
                    "name": s.stock_name,
                    "signal": s.signal,
                    "decision": s.decision,
                    "reason": s.reason,
                    "type": s.signal_type,
                }
                for s in signals
            ],
        }
    finally:
        session.close()


def run_daily(date: str | None, model: str | None) -> int:
    target = date or datetime.now().strftime("%Y%m%d")
    data = _collect_daily_data(target)

    if not data.get("pnl") and not data.get("trades") and not data.get("signals"):
        print(f"[codex_review] {target} 일자의 DB 기록이 비어있습니다. 건너뜁니다.")
        return 0

    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    display_date = f"{target[:4]}-{target[4:6]}-{target[6:8]}"
    output = _output_path("daily", display_date)
    rc = _run_codex_exec(DAILY_PROMPT, payload, output, model)
    if rc == 0:
        print(f"[codex_review] 완료: {output}")
    return rc


# ══════════════════════════════════════════════════════════════
# backtest 모드
# ══════════════════════════════════════════════════════════════
BACKTEST_PROMPT = """너는 퀀트 백테스트 리뷰어다.

<stdin> 에는 최신 백테스트 로그와 전략 비교 결과가 담겨 있다. 다음을 분석해 한국어 마크다운 리포트를 작성해라:

1. **한 줄 결론**
2. **전략별 성과 해석**: 각 전략의 total_return / sharpe / mdd / win_rate / total_trades 기반. 숫자를 반드시 인용
3. **Sharpe 대비 MDD 균형**: 위험조정수익 관점에서 추천 전략
4. **경고 신호**: 과적합 의심, turnover 이상, trades 수 극단치 등
5. **다음 스텝 제안**: 파라미터 튜닝 방향 또는 전략 보강 아이디어 3가지

감상/추상적 표현 금지. 수치 기반으로만 판단해라. 출력은 순수 마크다운.
"""


def _find_latest_backtest_log() -> Path | None:
    """logs/ 에서 가장 최근 backtest 관련 로그를 찾는다."""
    if not LOGS_DIR.exists():
        return None
    candidates = sorted(
        [p for p in LOGS_DIR.glob("*backtest*.log") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # fallback: supertrader.log
    main_log = LOGS_DIR / "supertrader.log"
    return main_log if main_log.exists() else None


def _tail_file(path: Path, max_bytes: int = 256_000) -> str:
    """파일 끝부분을 읽는다 (대용량 로그 대응)."""
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(size - max_bytes)
            f.readline()  # 중간 줄 버리기
        return f.read().decode("utf-8", errors="replace")


def run_backtest_review(log_path: str | None, comparison_csv: str | None, model: str | None) -> int:
    log_file = Path(log_path) if log_path else _find_latest_backtest_log()
    if log_file is None or not log_file.exists():
        print("[codex_review] 백테스트 로그를 찾지 못했습니다. --log 로 경로를 지정하세요.", file=sys.stderr)
        return 1

    log_text = _tail_file(log_file)

    comparison_text = ""
    if comparison_csv and Path(comparison_csv).exists():
        comparison_text = Path(comparison_csv).read_text(encoding="utf-8", errors="replace")

    payload = (
        f"## 백테스트 로그 ({log_file.name})\n\n```\n{log_text}\n```\n"
    )
    if comparison_text:
        payload += f"\n## 전략 비교 테이블\n\n```\n{comparison_text}\n```\n"

    today = datetime.now().strftime("%Y-%m-%d")
    output = _output_path("backtest", today)
    rc = _run_codex_exec(BACKTEST_PROMPT, payload, output, model)
    if rc == 0:
        print(f"[codex_review] 완료: {output}")
    return rc


# ══════════════════════════════════════════════════════════════
# code 모드
# ══════════════════════════════════════════════════════════════
CODE_PROMPT = """너는 이 저장소(SuperTrader, 한국 주식 자동매매)의 코드 리뷰어다.
CLAUDE.md 의 아키텍처 원칙과 기존 모듈 구조를 기준으로 현재 변경분을 리뷰해라.

중점 항목:
1. 매매 안전성 (주문/리스크/kill switch/중복 매수 방지)
2. 데이터 파이프라인 일관성 (백테스트 ↔ 라이브)
3. 모델 학습/추론 경로 영향
4. DB 스키마 하위호환
5. 설정(config) 드리프트

출력은 한국어 마크다운. 파일:라인 인용 필수. 사소한 스타일 지적은 생략.
"""


def run_code_review(base: str, uncommitted: bool, commit: str | None, model: str | None) -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    output = _output_path("code", today)
    rc = _run_codex_review(CODE_PROMPT, output, base, uncommitted, commit, model)
    if rc == 0:
        print(f"[codex_review] 완료: {output}")
    return rc


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Codex CLI 기반 SuperTrader 결과 리뷰 파이프라인",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_daily = sub.add_parser("daily", help="오늘자(또는 지정일) 매매 리뷰")
    p_daily.add_argument("--date", type=str, default=None, help="YYYYMMDD (기본: 오늘)")
    p_daily.add_argument("--model", type=str, default=None, help="codex 모델 override")

    p_bt = sub.add_parser("backtest", help="백테스트 결과 리뷰")
    p_bt.add_argument("--log", type=str, default=None, help="로그 파일 경로 (기본: logs/ 최신)")
    p_bt.add_argument("--comparison", type=str, default=None, help="전략 비교 CSV 경로")
    p_bt.add_argument("--model", type=str, default=None, help="codex 모델 override")

    p_code = sub.add_parser("code", help="코드 변경분 리뷰")
    p_code.add_argument("--base", type=str, default=None, help="비교 base 브랜치 (예: main)")
    p_code.add_argument("--uncommitted", action="store_true", help="staged+unstaged+untracked 리뷰")
    p_code.add_argument("--commit", type=str, default=None, help="특정 커밋 SHA 리뷰")
    p_code.add_argument("--model", type=str, default=None, help="codex 모델 override")

    args = parser.parse_args()

    if args.mode == "daily":
        return run_daily(args.date, args.model)
    if args.mode == "backtest":
        return run_backtest_review(args.log, args.comparison, args.model)
    if args.mode == "code":
        if not (args.base or args.uncommitted or args.commit):
            args.uncommitted = True  # 기본값
        return run_code_review(args.base, args.uncommitted, args.commit, args.model)
    return 1


if __name__ == "__main__":
    sys.exit(main())
