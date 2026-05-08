"""ETF Index Timing — regime 기반 KODEX 200/인버스 매매 (백테스트 only, MVP).

정책:
  risk_on_trend → KODEX 200 (069500) long
  high_vol_risk_off → KODEX 인버스 (114800) long (시장 short proxy)
  mean_revert → 현금

regime source: sappo_regime_labels (5-06 backfill 1729일).
ETF OHLCV: FDR.
거래비용: 수수료 0.015% × 2, ETF 증권거래세 0%, 슬리피지 0.15% × 2 → 회당 ~0.33%.

비교: KOSPI buy-and-hold (069500 만 보유) vs Index Timing.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db.sappo_models import init_sappo_db, get_sappo_session, RegimeLabel  # noqa: E402

ETF_LONG = "069500"      # KODEX 200
ETF_SHORT = "114800"     # KODEX 인버스
INITIAL_CAPITAL = 10_000_000
COMMISSION = 0.00015
SLIPPAGE = 0.0015         # 호가 한 단계 가정

REGIME_TO_ETF = {
    "risk_on_trend": ETF_LONG,
    "high_vol_risk_off": ETF_SHORT,
    "mean_revert": None,
}


def _fetch_etf_close(code: str, start: str, end: str) -> pd.Series:
    """FDR 로 ETF 종가 시계열 fetch."""
    import FinanceDataReader as fdr
    df = fdr.DataReader(code, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"ETF {code} 데이터 부재")
    df = df.rename(columns=str.lower)
    return df["close"].rename(code)


def _simulate(
    regime_by_date: dict[str, str],
    long_close: pd.Series,
    short_close: pd.Series,
    label: str,
    policy: dict[str, str | None],
) -> dict:
    """주어진 정책으로 시뮬레이션. 매일 regime 보고 ETF 회전."""
    cash = float(INITIAL_CAPITAL)
    holding_code: str | None = None
    holding_qty = 0
    holding_buy_price = 0.0

    equity_history: list[tuple[str, float]] = []
    trades: list[dict] = []
    daily_returns: list[float] = []
    prev_equity = float(INITIAL_CAPITAL)

    long_dates_set = set(long_close.index.strftime("%Y%m%d"))

    for date_str in sorted(regime_by_date.keys()):
        if date_str not in long_dates_set:
            continue
        ts = pd.to_datetime(date_str)
        regime = regime_by_date[date_str]
        target = policy.get(regime)

        # 현재가 (해당 ETF)
        long_p = float(long_close.loc[ts]) if ts in long_close.index else 0.0
        short_p = float(short_close.loc[ts]) if ts in short_close.index else 0.0

        # 1. 회전: 현재 보유와 target 다르면 매도
        if holding_code is not None and holding_code != target:
            cur_p = long_p if holding_code == ETF_LONG else short_p
            if cur_p > 0 and holding_qty > 0:
                eff = cur_p * (1.0 - SLIPPAGE)
                revenue = holding_qty * eff
                comm = revenue * COMMISSION
                cash += revenue - comm
                pnl_pct = (eff / holding_buy_price - 1) * 100 if holding_buy_price > 0 else 0
                trades.append({
                    "date": date_str, "side": "sell", "code": holding_code,
                    "qty": holding_qty, "price": eff, "pnl_pct": pnl_pct,
                })
                holding_code = None
                holding_qty = 0
                holding_buy_price = 0.0

        # 2. 매수: target 있고 미보유
        if target is not None and holding_code != target:
            target_p = long_p if target == ETF_LONG else short_p
            if target_p > 0:
                eff = target_p * (1.0 + SLIPPAGE)
                qty = int(cash * 0.99 / (eff * (1 + COMMISSION)))  # 99% 투입
                if qty > 0:
                    cost = qty * eff
                    comm = cost * COMMISSION
                    cash -= cost + comm
                    holding_code = target
                    holding_qty = qty
                    holding_buy_price = eff
                    trades.append({
                        "date": date_str, "side": "buy", "code": target,
                        "qty": qty, "price": eff, "pnl_pct": 0,
                    })

        # 3. equity 평가 (당일 종가 기준)
        eq = cash
        if holding_code == ETF_LONG and holding_qty > 0:
            eq += holding_qty * long_p
        elif holding_code == ETF_SHORT and holding_qty > 0:
            eq += holding_qty * short_p
        equity_history.append((date_str, eq))
        # 일일 수익률
        if prev_equity > 0:
            daily_returns.append(eq / prev_equity - 1)
        prev_equity = eq

    if not equity_history:
        return {}

    # 마지막 날 청산 후 metrics
    final_eq = equity_history[-1][1]
    eq_series = pd.Series([eq for _, eq in equity_history])
    rets = pd.Series(daily_returns)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    rf_daily = 0.035 / 250
    excess = rets - rf_daily
    n_years = len(eq_series) / 250
    return {
        "label": label,
        "total_return": (final_eq / INITIAL_CAPITAL - 1) * 100,
        "cagr": ((final_eq / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0,
        "sharpe_ratio": float(excess.mean() / excess.std() * np.sqrt(250)) if excess.std() > 0 else 0,
        "max_drawdown": float(dd.min()) * 100,
        "total_trades": len(trades),
        "n_days": len(eq_series),
        "final_equity": int(final_eq),
    }


def main():
    init_sappo_db("data/trading.db")
    s = get_sappo_session()
    try:
        rows = s.query(RegimeLabel).order_by(RegimeLabel.date).all()
        regime_by_date = {r.date: r.label for r in rows}
    finally:
        s.close()
    if not regime_by_date:
        logger.error("sappo_regime_labels 비어있음 — backfill 먼저 실행")
        return

    dates = sorted(regime_by_date.keys())
    start = pd.to_datetime(dates[0]).strftime("%Y-%m-%d")
    end = pd.to_datetime(dates[-1]).strftime("%Y-%m-%d")
    logger.info(f"regime 라벨: {len(dates)}일 ({start} ~ {end})")

    # ETF OHLCV
    logger.info(f"ETF 데이터 로드: {ETF_LONG}, {ETF_SHORT}")
    long_close = _fetch_etf_close(ETF_LONG, start, end)
    try:
        short_close = _fetch_etf_close(ETF_SHORT, start, end)
    except Exception as e:
        logger.warning(f"KODEX 인버스 로드 실패 — {e}, long-only 폴백")
        short_close = pd.Series(dtype=float)
    logger.info(f"long n={len(long_close)}, short n={len(short_close)}")

    # 정책 후보들
    policies = {
        "buy_hold_KODEX200": {  # 비교용: KODEX 200 만 항상 보유
            "risk_on_trend": ETF_LONG,
            "high_vol_risk_off": ETF_LONG,
            "mean_revert": ETF_LONG,
        },
        "long_only": {  # risk_on 만 long, 그 외 현금
            "risk_on_trend": ETF_LONG,
            "high_vol_risk_off": None,
            "mean_revert": None,
        },
        "long_short": {  # 풀 정책
            "risk_on_trend": ETF_LONG,
            "high_vol_risk_off": ETF_SHORT,
            "mean_revert": None,
        },
    }

    results = []
    for label, policy in policies.items():
        if label == "long_short" and short_close.empty:
            logger.warning(f"{label} skip — short ETF 데이터 부재")
            continue
        r = _simulate(regime_by_date, long_close, short_close, label, policy)
        if r:
            results.append(r)
            logger.info(
                f"[{label}] return={r['total_return']:+.2f}%, "
                f"CAGR={r['cagr']:+.2f}%, Sharpe={r['sharpe_ratio']:.2f}, "
                f"MDD={r['max_drawdown']:+.2f}%, trades={r['total_trades']}"
            )

    if not results:
        logger.error("결과 없음")
        return

    df = pd.DataFrame(results).set_index("label")
    df = df[["total_return", "cagr", "sharpe_ratio", "max_drawdown", "total_trades", "n_days", "final_equity"]]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/etf_index_timing_{stamp}.csv"
    md_path = ROOT / f"reports/etf_index_timing_{stamp}.md"
    df.to_csv(csv_path, encoding="utf-8")
    md_path.write_text(
        f"# ETF Index Timing (regime-driven) — {stamp}\n\n"
        f"기간: {start} ~ {end}, regime 라벨 {len(dates)}일\n\n"
        f"## 정책 비교\n\n```\n{df.to_string()}\n```\n",
        encoding="utf-8",
    )
    logger.info(f"\n결과:\n{df.to_string()}")
    logger.info(f"저장: {csv_path}")

    # Slack
    try:
        from src.notification.slack_bot import SlackNotifier
        n = SlackNotifier()
        if n.token:
            best = df["total_return"].idxmax()
            best_r = df.loc[best, "total_return"]
            best_s = df.loc[best, "sharpe_ratio"]
            best_m = df.loc[best, "max_drawdown"]
            buy_hold_r = df.loc["buy_hold_KODEX200", "total_return"] if "buy_hold_KODEX200" in df.index else 0
            alpha = best_r - buy_hold_r
            msg = (
                "📈 *ETF Index Timing — regime-driven 백테스트* (MVP)\n"
                f"기간: {start} ~ {end}, regime {len(dates)}일\n"
                "```\n"
                f"{df.to_string()}\n"
                "```\n"
                f"🏆 best policy: *{best}* — return {best_r:+.2f}%, "
                f"Sharpe {best_s:.2f}, MDD {best_m:+.2f}%\n"
                f"📊 vs buy_hold_KODEX200: alpha {alpha:+.2f}%p\n"
                f"리포트: `{csv_path.name}`"
            )
            n._send(msg)
            logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 예외: {e}")


if __name__ == "__main__":
    main()
