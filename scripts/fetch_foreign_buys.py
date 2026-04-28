"""
외국인 매매 데이터 수집 (KIS API FHKST01010900)

- 종목별 최근 30일 투자자(외국인/기관/개인) 매매동향 조회
- sappo_investor_trading 테이블에 upsert (stock_code, date) PK
- Mid-cap 풀 선정의 "외국인 20일 누적 순매수 상위 50%" 필터에 사용

Usage:
    # 현재 종목풀 + 시총 rank 30~150 universe 모두 수집
    python scripts/fetch_foreign_buys.py

    # 특정 종목 코드 지정
    python scripts/fetch_foreign_buys.py --codes 005930 035420

    # KOSPI 전체 (느림: ~900종목 × 0.1초 ≈ 90초)
    python scripts/fetch_foreign_buys.py --universe all
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.broker.kis_client import KISClient
from src.db.sappo_models import init_sappo_db, upsert_investor_trading


def _load_pool_codes() -> list[str]:
    """data/current_pool.json 에서 [code, ...] 추출."""
    from src.utils.json_io import load_json_with_fallback
    data = load_json_with_fallback("data/current_pool.json")
    if data is None:
        return []
    return [s["code"] for s in data.get("stocks", [])]


def _load_universe_codes(market: str = "KOSPI") -> list[str]:
    """get_universe() 로 PIT 유니버스 전체 (시총 필터 없이 KOSPI 전체)."""
    from src.data.market_data import get_universe
    today = datetime.now().strftime("%Y%m%d")
    df = get_universe(today)
    if df.empty:
        return []
    return df["code"].tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="외국인 매매 데이터 수집 (KIS API)")
    parser.add_argument("--codes", type=str, nargs="*", default=None,
                        help="종목코드 목록 (기본: 현재 풀)")
    parser.add_argument("--universe", choices=["pool", "all"], default="pool",
                        help="대상: pool (data/current_pool.json) | all (KOSPI 유니버스)")
    parser.add_argument("--sleep", type=float, default=0.12,
                        help="API 호출 간 sleep (초). KIS rate 10/sec 보호")
    args = parser.parse_args()

    load_config("config/settings.yaml")
    init_sappo_db("data/trading.db")

    if args.codes:
        codes = args.codes
        logger.info(f"--codes 지정: {len(codes)}종목")
    elif args.universe == "all":
        codes = _load_universe_codes()
        logger.info(f"KOSPI 유니버스: {len(codes)}종목")
    else:
        codes = _load_pool_codes()
        logger.info(f"현재 풀: {len(codes)}종목")

    if not codes:
        logger.error("대상 종목 없음 — --codes 또는 --universe 지정")
        return 1

    client = KISClient()
    t0 = time.time()
    n_ok = 0
    n_rows = 0
    n_err = 0

    for i, code in enumerate(codes, 1):
        try:
            records = client.get_investor_trading(code)
        except Exception as e:
            logger.warning(f"[{i}/{len(codes)}] {code} 조회 실패: {e}")
            n_err += 1
            time.sleep(args.sleep)
            continue

        for r in records:
            try:
                upsert_investor_trading(
                    stock_code=code,
                    date=r["date"],
                    close_price=r.get("close", 0),
                    foreign_net_qty=r.get("foreign_net_qty", 0),
                    foreign_net_amount=r.get("foreign_net_amount", 0),
                    organ_net_qty=r.get("organ_net_qty", 0),
                    organ_net_amount=r.get("organ_net_amount", 0),
                    person_net_qty=r.get("person_net_qty", 0),
                )
                n_rows += 1
            except Exception as e:
                logger.debug(f"upsert 실패 [{code} {r.get('date')}]: {e}")
        n_ok += 1

        # Progress every 50 stocks
        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (len(codes) - i)
            logger.info(
                f"  진행 {i}/{len(codes)} ({elapsed:.0f}s, ETA {eta:.0f}s) "
                f"성공 {n_ok} 행 {n_rows} 실패 {n_err}"
            )
        time.sleep(args.sleep)

    elapsed = time.time() - t0
    logger.info(
        f"외국인 매매 수집 완료: {n_ok}/{len(codes)}종목, "
        f"{n_rows}행 저장, {n_err}건 실패, {elapsed:.0f}초"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
