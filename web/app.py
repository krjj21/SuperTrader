"""
SuperTrader 대시보드 웹서버
- 계좌 현황, 보유 종목, 수익률 실시간 조회
- 매매 알고리즘 파이프라인 시각화
- 시그널 로그 + 기술 지표 차트
- 30초 자동 새로고침
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from flask import Flask, jsonify, render_template

from src.config import load_config, get_config, get_secrets
from src.broker.kis_client import KISClient
from src.broker.account import AccountManager

load_config("config/settings.yaml")

app = Flask(__name__)

_client = None
_account_mgr = None


def _get_account_mgr():
    global _client, _account_mgr
    if _client is None:
        _client = KISClient()
        _account_mgr = AccountManager(_client)
    return _account_mgr


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    try:
        acct = _get_account_mgr()
        s = acct.get_balance()

        positions = []
        total_invested = 0
        for p in s.positions:
            invested = p.avg_price * p.quantity
            total_invested += invested
            positions.append({
                "code": p.stock_code,
                "name": p.stock_name,
                "quantity": p.quantity,
                "avg_price": p.avg_price,
                "current_price": p.current_price,
                "eval_amount": p.eval_amount,
                "pnl": p.pnl,
                "pnl_pct": round(p.pnl_pct, 2),
            })

        # 수익률순 정렬
        positions.sort(key=lambda x: x["pnl_pct"], reverse=True)

        return jsonify({
            "ok": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_eval": s.total_eval,
                "total_deposit": s.total_deposit,
                "total_pnl": s.total_pnl,
                "total_pnl_pct": round(s.total_pnl_pct, 2),
                "total_invested": total_invested,
                "num_positions": len(s.positions),
                "initial_capital": 10_000_000,
                "total_return_pct": round((s.total_eval / 10_000_000 - 1) * 100, 2),
            },
            "positions": positions,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/pipeline")
def api_pipeline():
    """파이프라인 상태 API"""
    try:
        config = get_config()
        # Kill Switch 상태는 로그에서 확인
        log_path = Path("logs/supertrader.log")
        kill_switch = False
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")[-5000:]
            kill_switch = "KILL SWITCH" in text and "Kill switch" not in text.split("KILL SWITCH")[-1]

        return jsonify({
            "ok": True,
            "strategy": config.strategy.name,
            "pool_size": config.factors.top_n,
            "llm_enabled": bool(get_secrets().anthropic_api_key),
            "kill_switch": kill_switch,
            "check_interval": config.schedule.check_interval_sec,
            "daily_loss_limit": config.risk.daily_loss_limit_pct * 100,
            "max_positions": config.risk.max_total_positions,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/signals")
def api_signals():
    """최근 시그널 로그 API — 로그 파일에서 LLM 검증 결과 파싱"""
    try:
        log_path = Path("logs/supertrader.log")
        if not log_path.exists():
            return jsonify({"ok": True, "signals": []})

        # Windows Python이 CP949로 로그를 쓸 수 있으므로 여러 인코딩 시도
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                text = log_path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.strip().split("\n")

        signals = []
        # LLM 검증 로그 패턴: "LLM 검증 [종목명] SIGNAL: 확정/보류 — 이유"
        # 종목명이 빈 경우도 허용
        pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| INFO .+LLM .+ \[(.*?)\] (BUY|SELL): (\S+) . (.+)"
        )

        for line in reversed(lines):
            m = pattern.search(line)
            if m:
                ts, stock, signal, decision, reason = m.groups()
                # 인코딩 이슈 대응: 확정/보류 판단
                confirmed = "확정" in decision or "Confirmed" in decision or "보류" not in decision
                signals.append({
                    "time": ts,
                    "stock": stock,
                    "signal": signal,
                    "decision": "확정" if confirmed else "보류",
                    "reason": reason[:100],
                })
            if len(signals) >= 30:
                break

        return jsonify({"ok": True, "signals": signals})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/feedback")
def api_feedback():
    """최근 일일 피드백 API — 로그에서 LLM 피드백 파싱"""
    try:
        log_path = Path("logs/supertrader.log")
        if not log_path.exists():
            return jsonify({"ok": True, "feedback": None})

        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                text = log_path.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = log_path.read_text(encoding="utf-8", errors="ignore")

        # 피드백 전송 완료 로그를 역순으로 찾기
        if "일일 피드백 전송 완료" in text:
            return jsonify({"ok": True, "feedback": "최근 피드백이 Slack으로 전송되었습니다."})

        return jsonify({"ok": True, "feedback": None})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/indicators/<code>")
def api_indicators(code: str):
    """종목별 기술 지표 API"""
    try:
        import FinanceDataReader as fdr
        from src.data.indicators import add_all_indicators

        df = fdr.DataReader(code, "2024-06-01")
        if df is None or df.empty:
            return jsonify({"ok": False, "error": "데이터 없음"}), 404

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df = add_all_indicators(df)

        # 최근 60일
        df = df.tail(60).copy()
        df["date"] = df["date"].astype(str)

        result = {
            "ok": True,
            "code": code,
            "dates": df["date"].tolist(),
            "close": df["close"].tolist(),
            "ma5": df["close"].rolling(5).mean().tolist(),
            "ma20": df["close"].rolling(20).mean().tolist(),
            "ma60": df["close"].rolling(60, min_periods=1).mean().tolist(),
            "rsi": df["rsi"].tolist() if "rsi" in df.columns else [],
            "macd": df["macd"].tolist() if "macd" in df.columns else [],
            "macd_signal": df["macd_signal"].tolist() if "macd_signal" in df.columns else [],
            "macd_hist": df["macd_hist"].tolist() if "macd_hist" in df.columns else [],
            "volume": df["volume"].tolist(),
        }
        # NaN을 null로 변환
        for key in result:
            if isinstance(result[key], list):
                result[key] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in result[key]]

        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
