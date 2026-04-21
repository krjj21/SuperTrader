"""
주간 SAPPO 검증 리포트

매주 토요일 실행 (APScheduler 또는 cron):
  1. 지난 7일 뉴스/sentiment 수집 통계 집계
  2. Sentiment → N일 forward return IC 측정 (논문 검증 지표)
  3. baseline(λ=0) vs SAPPO(λ>0) 학습 런 성과 비교
  4. sappo_weekly_metrics 테이블에 upsert
  5. 결과를 Slack 에 전송 (--slack)

Usage:
    python scripts/weekly_sappo_report.py
    python scripts/weekly_sappo_report.py --slack
    python scripts/weekly_sappo_report.py --week-start 20260413
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.db.sappo_models import (
    init_sappo_db, get_sappo_session,
    NewsArticle, SentimentScore, SappoTrainingRun,
    save_ic_metric, upsert_weekly_metric,
)
from src.data.market_data import get_ohlcv


def _week_start(ref: datetime | None = None) -> datetime:
    """ref 가 속한 주의 월요일을 반환."""
    d = ref or datetime.now()
    return d - timedelta(days=d.weekday())


def _collect_aggregates(week_start_dt: datetime) -> dict:
    """지난 7일 sappo_news/sappo_sentiment_scores 집계."""
    session = get_sappo_session()
    try:
        end_dt = week_start_dt + timedelta(days=7)
        ws, we = week_start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
        n_news = session.query(NewsArticle).filter(
            NewsArticle.date >= ws, NewsArticle.date < we,
        ).count()

        sentiments = session.query(SentimentScore).filter(
            SentimentScore.date >= ws, SentimentScore.date < we,
        ).all()
        n_sent = len(sentiments)
        avg_s = float(sum(s.score for s in sentiments) / n_sent) if n_sent else 0.0
        avg_c = float(sum(s.confidence for s in sentiments) / n_sent) if n_sent else 0.0
    finally:
        session.close()

    return {
        "week_start": week_start_dt.strftime("%Y%m%d"),
        "window_end": end_dt.strftime("%Y%m%d"),
        "n_news": n_news,
        "n_sentiment_scores": n_sent,
        "avg_sentiment": avg_s,
        "avg_confidence": avg_c,
    }


def _compute_sentiment_ic(forward_days: int = 5, look_back_days: int = 60) -> dict:
    """sentiment → forward return IC 측정. 최근 look_back_days 범위 전체에서."""
    try:
        from scipy import stats
    except ImportError:
        return {"ic": 0.0, "ic_pvalue": 1.0, "n_samples": 0, "hit_rate": 0.0}

    session = get_sappo_session()
    cutoff_recent = (datetime.now() - timedelta(days=forward_days)).strftime("%Y%m%d")
    cutoff_old = (datetime.now() - timedelta(days=look_back_days)).strftime("%Y%m%d")
    try:
        sentiments = (
            session.query(SentimentScore)
            .filter(
                SentimentScore.date >= cutoff_old,
                SentimentScore.date <= cutoff_recent,
                SentimentScore.confidence > 0.0,  # 진짜 생성된 신호만
            )
            .all()
        )
    finally:
        session.close()

    if not sentiments:
        return {"ic": 0.0, "ic_pvalue": 1.0, "n_samples": 0, "hit_rate": 0.0,
                "window_start": cutoff_old, "window_end": cutoff_recent}

    # 종목별 OHLCV 캐시
    ohlcv: dict[str, pd.DataFrame] = {}
    codes = sorted({s.stock_code for s in sentiments})
    start = cutoff_old
    end = (datetime.now() + timedelta(days=forward_days + 5)).strftime("%Y%m%d")
    for code in codes:
        try:
            df = get_ohlcv(code, start, end)
            if df is not None and not df.empty:
                ohlcv[code] = df.sort_values("date").reset_index(drop=True)
        except Exception:
            pass

    xs, ys = [], []
    for s in sentiments:
        df = ohlcv.get(s.stock_code)
        if df is None:
            continue
        target = pd.Timestamp(s.date)
        future = df[df["date"] >= target]
        if len(future) < forward_days + 1:
            continue
        p0 = float(future["close"].iloc[0])
        p1 = float(future["close"].iloc[forward_days])
        if p0 <= 0:
            continue
        fwd = (p1 / p0 - 1.0) * 100.0
        xs.append(s.score)
        ys.append(fwd)

    n = len(xs)
    if n < 10 or len(set(xs)) < 2:
        return {"ic": 0.0, "ic_pvalue": 1.0, "n_samples": n, "hit_rate": 0.0,
                "window_start": cutoff_old, "window_end": cutoff_recent}

    ic, pvalue = stats.spearmanr(xs, ys)
    # hit rate: sentiment 부호와 수익률 부호 일치 비율
    hit = sum(1 for x, y in zip(xs, ys) if (x > 0 and y > 0) or (x < 0 and y < 0)) / max(
        sum(1 for x in xs if x != 0.0), 1
    )

    return {
        "ic": float(ic) if ic == ic else 0.0,   # NaN 방어
        "ic_pvalue": float(pvalue) if pvalue == pvalue else 1.0,
        "n_samples": n,
        "hit_rate": float(hit),
        "window_start": cutoff_old,
        "window_end": cutoff_recent,
        "forward_days": forward_days,
    }


def _compare_training_runs(week_start_dt: datetime) -> dict:
    """지난 7일 학습 런 조회 → baseline(λ=0) 과 SAPPO(λ>0) 최고 Sharpe 비교."""
    end_dt = week_start_dt + timedelta(days=7)
    session = get_sappo_session()
    try:
        runs = (
            session.query(SappoTrainingRun)
            .filter(
                SappoTrainingRun.created_at >= week_start_dt,
                SappoTrainingRun.created_at < end_dt,
            )
            .all()
        )
    finally:
        session.close()

    baseline = [r for r in runs if abs(r.lambda_value) < 1e-9]
    sappo = [r for r in runs if r.lambda_value > 0.0]

    best_base = max((r.val_sharpe for r in baseline), default=0.0) if baseline else 0.0
    best_sappo = 0.0
    best_lambda = 0.0
    if sappo:
        best_run = max(sappo, key=lambda r: r.val_sharpe)
        best_sappo = best_run.val_sharpe
        best_lambda = best_run.lambda_value

    return {
        "n_training_runs": len(runs),
        "baseline_sharpe": float(best_base),
        "best_sappo_sharpe": float(best_sappo),
        "best_sappo_lambda": float(best_lambda),
        "sharpe_improvement": float(best_sappo - best_base) if sappo and baseline else 0.0,
    }


def _send_slack(report: dict) -> None:
    """Slack 에 주간 리포트 전송."""
    try:
        from slack_sdk import WebClient
        from src.config import get_secrets
        token = get_secrets().slack_bot_token
        if not token:
            logger.warning("SLACK_BOT_TOKEN 미설정")
            return
        client = WebClient(token=token)
        text = _format_slack(report)
        client.chat_postMessage(channel="C0AT0BM1AHF", text=text)
        logger.info("Slack 전송 완료")
    except Exception as e:
        logger.warning(f"Slack 전송 실패: {e}")


def _format_slack(r: dict) -> str:
    improv = r["sharpe_improvement"]
    arrow = "⬆️" if improv > 0 else ("⬇️" if improv < 0 else "➡️")
    signif = ""
    if r["sentiment_ic_pvalue"] < 0.05:
        signif = " **"
    elif r["sentiment_ic_pvalue"] < 0.1:
        signif = " *"
    return (
        f":calendar: *주간 SAPPO 검증 — {r['week_start']} ~ {r['window_end']}*\n\n"
        f"*뉴스·Sentiment 수집*\n"
        f"• 뉴스 건수: *{r['n_news']:,}*\n"
        f"• Sentiment 생성: *{r['n_sentiment_scores']:,}* (평균 score {r['avg_sentiment']:+.2f}, confidence {r['avg_confidence']:.2f})\n\n"
        f"*Sentiment → {r.get('forward_days', 5)}일 Forward Return IC*\n"
        f"• IC: *{r['sentiment_ic_5d']:+.3f}*{signif} (p={r['sentiment_ic_pvalue']:.3f}, N={r['sentiment_ic_n']})\n"
        f"• Hit rate: {r['hit_rate']*100:.1f}%\n\n"
        f"*학습 런 비교 (이 주)*\n"
        f"• 총 런: {r['n_training_runs']}\n"
        f"• Baseline (λ=0) 최고 Sharpe: {r['baseline_sharpe']:.3f}\n"
        f"• SAPPO 최고 Sharpe: {r['best_sappo_sharpe']:.3f} (λ={r['best_sappo_lambda']})\n"
        f"• 개선: {arrow} {improv:+.3f}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="주간 SAPPO 검증 리포트")
    parser.add_argument("--week-start", type=str, default=None, help="주 시작(월요일) YYYYMMDD (기본: 이번주)")
    parser.add_argument("--forward", type=int, default=5, help="IC forward 일수")
    parser.add_argument("--look-back", type=int, default=60, help="IC 측정 look-back 일수")
    parser.add_argument("--slack", action="store_true", help="결과를 Slack 에 전송")
    args = parser.parse_args()

    load_config("config/settings.yaml")
    init_sappo_db("data/trading.db")

    if args.week_start:
        week_start_dt = datetime.strptime(args.week_start, "%Y%m%d")
    else:
        week_start_dt = _week_start()
    logger.info(f"리포트 대상 주 시작: {week_start_dt.strftime('%Y-%m-%d')} (월)")

    agg = _collect_aggregates(week_start_dt)
    ic = _compute_sentiment_ic(forward_days=args.forward, look_back_days=args.look_back)
    runs = _compare_training_runs(week_start_dt)

    report = {
        **agg,
        "sentiment_ic_5d": ic["ic"],
        "sentiment_ic_pvalue": ic["ic_pvalue"],
        "sentiment_ic_n": ic["n_samples"],
        "hit_rate": ic["hit_rate"],
        "forward_days": ic.get("forward_days", args.forward),
        **runs,
    }

    # DB 에 IC 스냅샷 기록
    save_ic_metric(
        forward_days=args.forward,
        window_start=ic.get("window_start", ""),
        window_end=ic.get("window_end", ""),
        n_samples=ic["n_samples"],
        ic=ic["ic"],
        ic_pvalue=ic["ic_pvalue"],
        hit_rate=ic["hit_rate"],
        notes=f"weekly report for {report['week_start']}",
    )

    # 주간 집계 upsert
    upsert_weekly_metric(
        week_start=report["week_start"],
        n_news=report["n_news"],
        n_sentiment_scores=report["n_sentiment_scores"],
        avg_sentiment=report["avg_sentiment"],
        avg_confidence=report["avg_confidence"],
        sentiment_ic_5d=report["sentiment_ic_5d"],
        sentiment_ic_pvalue=report["sentiment_ic_pvalue"],
        sentiment_ic_n=report["sentiment_ic_n"],
        baseline_sharpe=report["baseline_sharpe"],
        best_sappo_sharpe=report["best_sappo_sharpe"],
        best_sappo_lambda=report["best_sappo_lambda"],
        sharpe_improvement=report["sharpe_improvement"],
        n_training_runs=report["n_training_runs"],
    )

    # 콘솔 출력
    print("\n" + "=" * 60)
    print(f"  주간 SAPPO 검증 리포트 ({report['week_start']}~{report['window_end']})")
    print("=" * 60)
    for k, v in report.items():
        print(f"  {k:<25s}: {v}")
    print("=" * 60)

    if args.slack:
        _send_slack(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
