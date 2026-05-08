"""풀 진단 — 현재 30종목이 *momentum 강한가, 방어적인가*.

검증:
1. 현재 풀 30종목의 60d / 20d / 5d return 분포
2. KOSPI universe (337종목) 평균 / 분포와 비교
3. 60d momentum 상위 50종목 vs 풀 overlap (몇 % 가 진짜 momentum 종목)
4. 풀 종목의 평균 변동성 (ATR/price) 와 universe 평균 비교
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.market_data import get_universe, get_ohlcv_batch  # noqa: E402
from src.data.indicators import calc_atr  # noqa: E402


def main():
    # 1. 현재 라이브 풀
    pool_path = ROOT / "data/current_pool.json"
    with open(pool_path, encoding="utf-8") as f:
        pool_json = json.load(f)
    pool_codes = [s["code"] for s in pool_json["stocks"]]
    pool_date = pool_json["date"]
    logger.info(f"라이브 풀: {len(pool_codes)}종목 (date={pool_date})")

    # 2. KOSPI universe (오늘 시점)
    today = datetime.now().strftime("%Y%m%d")
    uni = get_universe(today)
    if uni.empty:
        logger.error("universe 비어있음")
        return
    uni_codes = uni["code"].tolist()
    logger.info(f"KOSPI universe: {len(uni_codes)}종목")

    # 3. OHLCV 60일치 (universe 전체)
    end = datetime.now()
    start = (end - pd.Timedelta(days=120)).strftime("%Y%m%d")
    end = end.strftime("%Y%m%d")
    logger.info(f"OHLCV 로드 ({start} ~ {end}): {len(uni_codes)}종목")
    ohlcv = get_ohlcv_batch(uni_codes, start, end)
    logger.info(f"OHLCV 로드 완료: {len(ohlcv)}종목")

    # 4. 종목별 60d / 20d / 5d return + ATR ratio 계산
    rows = []
    for code in uni_codes:
        df = ohlcv.get(code)
        if df is None or len(df) < 60:
            continue
        c = df["close"].values
        if c[-60] <= 0 or c[-20] <= 0 or c[-5] <= 0:
            continue
        ret_60d = (c[-1] / c[-60] - 1) * 100
        ret_20d = (c[-1] / c[-20] - 1) * 100
        ret_5d = (c[-1] / c[-5] - 1) * 100
        # ATR ratio
        try:
            atr = calc_atr(df["high"], df["low"], df["close"], period=14).iloc[-1]
            atr_ratio = float(atr / c[-1] * 100) if c[-1] > 0 else 0
        except Exception:
            atr_ratio = 0
        rows.append({
            "code": code,
            "in_pool": code in pool_codes,
            "ret_60d": ret_60d,
            "ret_20d": ret_20d,
            "ret_5d": ret_5d,
            "atr_ratio": atr_ratio,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("분석 가능 종목 없음")
        return
    pool_df = df[df["in_pool"]]
    uni_df = df  # universe 전체

    # 5. 분포 비교
    print()
    print("=" * 75)
    print("풀(30) vs Universe 비교 (60일 기준)")
    print("=" * 75)
    print(f"{'metric':20s} | {'pool mean':>10s} | {'uni mean':>10s} | {'pool-uni':>9s}")
    print("-" * 75)
    for col in ["ret_60d", "ret_20d", "ret_5d", "atr_ratio"]:
        pm = pool_df[col].mean()
        um = uni_df[col].mean()
        print(f"{col:20s} | {pm:>+9.2f}% | {um:>+9.2f}% | {pm-um:>+8.2f}%")

    # 6. 60d momentum 상위 50 종목 vs 풀 overlap
    top50_by_mom = uni_df.nlargest(50, "ret_60d")["code"].tolist()
    overlap_n = len(set(top50_by_mom) & set(pool_codes))
    print()
    print(f"60d momentum top 50 ∩ 풀 30: {overlap_n}/30 = {overlap_n/30*100:.0f}%")
    print(f"60d momentum top 100 ∩ 풀 30: "
          f"{len(set(uni_df.nlargest(100, 'ret_60d')['code']) & set(pool_codes))}/30")

    # 7. 풀 종목별 detail (60d 정렬)
    print()
    print("=" * 75)
    print(f"풀 30종목 momentum 상세 (60d 정렬)")
    print("=" * 75)
    pool_sorted = pool_df.sort_values("ret_60d", ascending=False)
    print(f"{'code':>8s} | {'60d':>8s} | {'20d':>8s} | {'5d':>8s} | {'ATR%':>6s}")
    for _, r in pool_sorted.iterrows():
        print(f"{r['code']:>8s} | {r['ret_60d']:>+7.2f}% | {r['ret_20d']:>+7.2f}% | "
              f"{r['ret_5d']:>+7.2f}% | {r['atr_ratio']:>5.2f}%")

    # 8. 진단
    print()
    print("=" * 75)
    print("진단")
    print("=" * 75)
    pool_mom_60 = pool_df["ret_60d"].mean()
    uni_mom_60 = uni_df["ret_60d"].mean()
    pool_atr = pool_df["atr_ratio"].mean()
    uni_atr = uni_df["atr_ratio"].mean()
    if pool_mom_60 < uni_mom_60:
        print(f"❌ 풀 60d momentum ({pool_mom_60:+.2f}%) < universe 평균 ({uni_mom_60:+.2f}%)")
        print(f"   → 풀이 *추세 약한 종목* 위주 (방어적)")
    else:
        print(f"✓ 풀 60d momentum ({pool_mom_60:+.2f}%) >= universe ({uni_mom_60:+.2f}%)")
    if pool_atr < uni_atr:
        print(f"❌ 풀 ATR ({pool_atr:.2f}%) < universe ({uni_atr:.2f}%)")
        print(f"   → 풀이 *변동성 낮은 종목* 위주 → alpha 작음")
    if overlap_n < 10:
        print(f"❌ momentum top 50 와 overlap 낮음 ({overlap_n}/30)")
        print(f"   → IC composite 가 momentum 신호 무시")

    # 결과 CSV
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ROOT / f"reports/pool_momentum_{stamp}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"저장: {csv_path}")


if __name__ == "__main__":
    main()
