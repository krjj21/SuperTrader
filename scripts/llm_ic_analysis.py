"""
LLM 검증 결정의 정보계수(IC) 측정

signal_logs 테이블의 LLM 확정/보류 판정과 실제 N일 forward return 을 비교해
LLM 검증이 실제 수익에 알파를 더하는지 정량 측정한다.

IC 해석:
  + : LLM 결정이 수익에 예측력 있음
  0 : 노이즈, 알파 없음
  - : LLM 결정이 역으로 작용 (제거 권장)

한계:
  - 확정만 실제 체결. 보류는 counterfactual (실제 수익 관측 불가)
    → forward_return 은 "신호 이후 가격이 어떻게 움직였는지" 이지
       "실제 P&L" 이 아님. 판단 품질의 proxy.
  - 데이터 기간이 짧을수록 샘플 편향 큼

Usage:
    python scripts/llm_ic_analysis.py
    python scripts/llm_ic_analysis.py --forward 5 --days 30
    python scripts/llm_ic_analysis.py --forward 1 --out reports/llm_ic_1d.csv
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
from src.db.models import init_db, get_session, SignalLog
from src.data.market_data import get_ohlcv


def compute_forward_return(code: str, signal_date: datetime, forward_days: int) -> float | None:
    """signal_date 이후 forward_days 영업일 종가 기준 수익률(%) 을 반환."""
    start = signal_date.strftime("%Y%m%d")
    end = (signal_date + timedelta(days=forward_days * 2 + 5)).strftime("%Y%m%d")
    try:
        df = get_ohlcv(code, start, end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.sort_values("date").reset_index(drop=True)
    target = pd.Timestamp(signal_date.strftime("%Y%m%d"))
    future = df[df["date"] >= target]
    if len(future) < forward_days + 1:
        return None
    p0 = float(future["close"].iloc[0])
    p1 = float(future["close"].iloc[forward_days])
    if p0 <= 0:
        return None
    return (p1 / p0 - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 결정 IC 분석")
    parser.add_argument("--forward", type=int, default=5, help="forward 영업일")
    parser.add_argument("--days", type=int, default=30, help="조회 look-back 일수")
    parser.add_argument("--out", type=str, default=None, help="원자료 CSV 저장 경로")
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    args = parser.parse_args()

    load_config(args.config)
    init_db("data/trading.db")

    cutoff_recent = datetime.now() - timedelta(days=args.forward)
    cutoff_oldest = datetime.now() - timedelta(days=args.days)

    session = get_session()
    try:
        rows = (
            session.query(SignalLog)
            .filter(
                SignalLog.created_at >= cutoff_oldest,
                SignalLog.created_at <= cutoff_recent,
                SignalLog.signal_type == "llm",
            )
            .order_by(SignalLog.created_at.asc())
            .all()
        )
    finally:
        session.close()

    logger.info(
        f"조회: LLM 시그널 {len(rows)}건 ({cutoff_oldest.date()} ~ {cutoff_recent.date()}, forward={args.forward}일)"
    )

    if not rows:
        logger.warning("분석 대상 없음")
        return 1

    # 종목별 OHLCV 캐시 (API 중복 호출 방지)
    ohlcv_cache: dict[str, pd.DataFrame] = {}
    min_date = min(r.created_at for r in rows).strftime("%Y%m%d")
    max_date = (max(r.created_at for r in rows) + timedelta(days=args.forward * 2 + 5)).strftime("%Y%m%d")
    unique_codes = sorted({r.stock_code for r in rows if r.stock_code})
    logger.info(f"종목 OHLCV 로드: {len(unique_codes)}종목 ({min_date} ~ {max_date})")
    for i, code in enumerate(unique_codes):
        try:
            df = get_ohlcv(code, min_date, max_date)
            if df is not None and not df.empty:
                ohlcv_cache[code] = df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.debug(f"OHLCV 로드 실패: {code} - {e}")
        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(unique_codes)}")

    records = []
    for r in rows:
        df = ohlcv_cache.get(r.stock_code)
        if df is None:
            continue
        target = pd.Timestamp(r.created_at.strftime("%Y%m%d"))
        future = df[df["date"] >= target]
        if len(future) < args.forward + 1:
            continue
        p0 = float(future["close"].iloc[0])
        p1 = float(future["close"].iloc[args.forward])
        if p0 <= 0:
            continue
        fwd = (p1 / p0 - 1.0) * 100.0
        records.append({
            "time": r.created_at,
            "code": r.stock_code,
            "name": r.stock_name,
            "signal": r.signal,
            "decision": r.decision,
            "forward_return": fwd,
        })

    if not records:
        logger.warning("forward return 계산 가능한 건 없음")
        return 1

    df = pd.DataFrame(records)
    logger.info(f"분석 대상: {len(df)}건")

    print(f"\n{'=' * 60}")
    print(f"  LLM 결정 IC 분석 (forward={args.forward}일, N={len(df)}건)")
    print(f"{'=' * 60}\n")

    # 시그널별 상세
    for signal in ["BUY", "SELL"]:
        sub = df[df["signal"] == signal]
        if sub.empty:
            continue

        print(f"\n[{signal}] 총 {len(sub)}건")
        print("-" * 40)

        grouped = sub.groupby("decision")["forward_return"].agg(["count", "mean", "std", "min", "max"])
        grouped = grouped.round(2)
        print(grouped.to_string())

        # Hit rate (BUY: forward>0, SELL 확정: forward<0, SELL 보류: forward>0)
        print(f"\nHit rate:")
        for decision in sub["decision"].unique():
            dsub = sub[sub["decision"] == decision]["forward_return"]
            if len(dsub) == 0:
                continue
            if signal == "BUY":
                hit = (dsub > 0).mean()
                target_desc = "수익(>0)"
            else:  # SELL
                if decision == "확정":
                    hit = (dsub < 0).mean()
                    target_desc = "손실회피(<0)"
                else:
                    hit = (dsub > 0).mean()
                    target_desc = "보유유지가 유리(>0)"
            print(f"  {decision:6s}: {hit*100:>5.1f}% (n={len(dsub)}, {target_desc})")

        # Spearman IC: decision → forward return
        # BUY: 확정=+1, 보류=-1 (LLM 이 양수 방향 예측 의도)
        # SELL: 확정=-1, 보류=+1 (LLM 이 음수 방향 예측 의도, 즉 확정시 가격 하락 기대)
        if signal == "BUY":
            smap = {"확정": 1, "보류": -1}
        else:
            smap = {"확정": -1, "보류": 1}

        sub2 = sub[sub["decision"].isin(smap.keys())].copy()
        sub2["score"] = sub2["decision"].map(smap)
        if len(sub2) >= 10 and sub2["score"].nunique() >= 2:
            try:
                from scipy import stats
                ic, pvalue = stats.spearmanr(sub2["score"], sub2["forward_return"])
                signif = "**" if pvalue < 0.05 else ("*" if pvalue < 0.1 else "")
                print(f"\nSpearman IC: {ic:+.4f} (p={pvalue:.3f}, n={len(sub2)}) {signif}")
                if abs(ic) < 0.02:
                    interpret = "거의 알파 없음"
                elif ic > 0:
                    interpret = "LLM 결정이 수익 방향으로 예측력 있음 (적합)"
                else:
                    interpret = "LLM 결정이 반대 방향 — 제거 검토 필요"
                print(f"  → {interpret}")
            except Exception as e:
                print(f"\nIC 계산 실패: {e}")

    # 전체 시그널 무관 baseline
    print("\n" + "=" * 60)
    print("Baseline (전체 시그널 평균)")
    print("-" * 60)
    print(f"  mean forward return: {df['forward_return'].mean():.2f}%")
    print(f"  hit rate (>0):       {(df['forward_return'] > 0).mean()*100:.1f}%")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n원자료 저장: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
