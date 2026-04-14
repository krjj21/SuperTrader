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
from src.db.models import init_db, get_runtime_status, get_recent_signals

load_config("config/settings.yaml")
init_db(get_config().database.path)

app = Flask(__name__)

_client = None
_account_mgr = None
_stock_name_cache: dict[str, str] = {}


def _lookup_stock_name(code: str) -> str:
    """종목코드 → 종목명 (pykrx 캐시)"""
    if not code or not code.isdigit() or len(code) != 6:
        return ""
    if code in _stock_name_cache:
        return _stock_name_cache[code]
    try:
        from pykrx import stock as krx
        name = krx.get_market_ticker_name(code) or ""
        _stock_name_cache[code] = name
        return name
    except Exception:
        _stock_name_cache[code] = ""
        return ""


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

        initial_capital = (s.total_eval - s.asset_change) if s.asset_change is not None else 10_000_000

        # 실현손익 조회
        realized = acct.get_realized_pnl()

        return jsonify({
            "ok": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_eval": s.total_eval,
                "total_deposit": s.total_deposit,
                "available_cash": s.available_cash,
                "total_pnl": s.asset_change,
                "total_pnl_pct": round(s.asset_change_pct, 2),
                "total_invested": total_invested,
                "num_positions": len(s.positions),
                "initial_capital": initial_capital,
                "total_return_pct": round(s.asset_change_pct, 2),
                "asset_change": s.asset_change,
                "asset_change_pct": round(s.asset_change_pct, 2),
                "realized_pnl": realized["realized_pnl"],
                "unrealized_pnl": s.total_pnl,
                "total_sell": realized["total_sell"],
                "total_buy": realized["total_buy"],
                "n_trades": realized["n_trades"],
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
        runtime = get_runtime_status()

        return jsonify({
            "ok": True,
            "strategy": runtime.strategy if runtime else config.strategy.name,
            "pool_size": runtime.pool_size if runtime else config.factors.top_n,
            "llm_enabled": runtime.llm_enabled if runtime else bool(get_secrets().anthropic_api_key),
            "kill_switch": runtime.kill_switch if runtime else False,
            "check_interval": runtime.check_interval if runtime else config.schedule.check_interval_sec,
            "daily_loss_limit": runtime.daily_loss_limit if runtime else config.risk.daily_loss_limit_pct * 100,
            "max_positions": runtime.max_positions if runtime else config.risk.max_total_positions,
            "updated_at": (
                runtime.updated_at.strftime("%Y-%m-%d %H:%M:%S")
                if runtime and runtime.updated_at else None
            ),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/signals")
def api_signals():
    """최근 시그널 이력 API — DB 우선, 로그 fallback"""
    try:
        from flask import request
        days = int(request.args.get("days", 7))
        limit = int(request.args.get("limit", 100))
        days = min(max(days, 1), 90)
        limit = min(max(limit, 1), 500)

        # DB에서 조회
        rows = get_recent_signals(limit=limit, days=days)
        if rows:
            signals = []
            for r in rows:
                signals.append({
                    "time": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "",
                    "stock_code": r.stock_code or "",
                    "stock_name": r.stock_name or "-",
                    "signal": r.signal,
                    "decision": r.decision,
                    "reason": r.reason if r.reason else "",
                    "type": r.signal_type or "llm",
                })
            return jsonify({"ok": True, "signals": signals, "source": "db"})

        # DB에 데이터 없으면 로그 파일 fallback
        signals = _parse_signals_from_log(limit=limit)
        return jsonify({"ok": True, "signals": signals, "source": "log"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def _parse_signals_from_log(limit: int = 100) -> list[dict]:
    """로그 파일에서 시그널을 파싱합니다 (DB 없을 때 fallback)."""
    log_path = Path("logs/supertrader.log")
    if not log_path.exists():
        return []

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
    llm_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| INFO .+LLM .+ \[(.*?)\] (BUY|SELL): (\S+) . (.+)"
    )
    reject_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| INFO .+LLM 보류: (.+?) (BUY|SELL) . (.+)"
    )
    summary_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ \| INFO .+시그널 체크 완료: (\d+)종목 . BUY:(\d+) SELL:(\d+) HOLD:(\d+) 오류:(\d+)"
    )

    def parse_stock_field(s: str) -> tuple[str, str]:
        if not s:
            return "", "-"
        s = s.strip()
        if "|" in s:
            code, name = s.split("|", 1)
            code, name = code.strip(), name.strip()
            if code.isdigit() and len(code) == 6 and (not name or name == "-"):
                name = _lookup_stock_name(code) or "-"
            return code, name or "-"
        if s.isdigit() and len(s) == 6:
            return s, _lookup_stock_name(s) or "-"
        return "", s

    for line in reversed(lines):
        m = llm_pattern.search(line)
        if m:
            ts, stock, signal, decision, reason = m.groups()
            code, name = parse_stock_field(stock)
            confirmed = "확정" in decision or "Confirmed" in decision or "보류" not in decision
            signals.append({
                "time": ts, "stock_code": code, "stock_name": name,
                "signal": signal, "decision": "확정" if confirmed else "보류",
                "reason": reason, "type": "llm",
            })
            if len(signals) >= limit:
                break
            continue

        m = reject_pattern.search(line)
        if m:
            ts, stock, signal, reason = m.groups()
            code, name = parse_stock_field(stock)
            signals.append({
                "time": ts, "stock_code": code, "stock_name": name,
                "signal": signal, "decision": "보류",
                "reason": reason, "type": "llm",
            })
            if len(signals) >= limit:
                break
            continue

        m = summary_pattern.search(line)
        if m:
            ts, total, buy, sell, hold, err = m.groups()
            signals.append({
                "time": ts, "stock_code": "", "stock_name": f"{total}종목 스캔",
                "signal": "SCAN", "decision": f"B:{buy} S:{sell} H:{hold}",
                "reason": f"오류: {err}" if int(err) > 0 else "정상", "type": "summary",
            })
            if len(signals) >= limit:
                break

    return signals


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


@app.route("/api/returns")
def api_returns():
    """일일 수익률 시계열 API — DailyPnL 테이블에서 조회"""
    try:
        from src.db.models import init_db, get_session, DailyPnL

        init_db()
        session = get_session()
        rows = session.query(DailyPnL).order_by(DailyPnL.date).all()
        session.close()

        if not rows:
            return jsonify({"ok": True, "dates": [], "total_eval": [], "total_pnl_pct": []})

        initial_eval = rows[0].total_eval - rows[0].total_pnl if rows[0].total_eval else 10_000_000
        if initial_eval <= 0:
            initial_eval = 10_000_000

        dates = []
        total_eval = []
        total_pnl_pct = []
        cumulative_return = []

        for r in rows:
            dates.append(f"{r.date[:4]}-{r.date[4:6]}-{r.date[6:]}" if len(r.date) == 8 else r.date)
            total_eval.append(r.total_eval)
            total_pnl_pct.append(round(r.total_pnl_pct, 2))
            cum_ret = round((r.total_eval / initial_eval - 1) * 100, 2) if r.total_eval else 0
            cumulative_return.append(cum_ret)

        return jsonify({
            "ok": True,
            "dates": dates,
            "total_eval": total_eval,
            "total_pnl_pct": total_pnl_pct,
            "cumulative_return": cumulative_return,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/pool")
def api_pool():
    """현재 팩터 종목풀 API"""
    try:
        import json
        pool_path = Path("data/current_pool.json")
        if not pool_path.exists():
            return jsonify({"ok": True, "stocks": [], "date": None, "count": 0})

        raw = pool_path.read_bytes()
        for enc in ("utf-8", "cp949", "euc-kr"):
            try:
                text = raw.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        else:
            text = raw.decode("utf-8", errors="ignore")

        data = json.loads(text)
        # 종목명이 비어있으면 pykrx로 보충
        for s in data.get("stocks", []):
            if not s.get("name"):
                s["name"] = _lookup_stock_name(s["code"])
        return jsonify({"ok": True, **data})
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
