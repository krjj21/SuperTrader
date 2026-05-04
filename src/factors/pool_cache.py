"""
종목풀 히스토리 캐시

동일한 config(기간/유니버스/팩터모듈/리밸런싱주기 등) 이면
`pool_history` 는 결정적이므로 파일로 저장해 재사용한다.

캐시 키: config 관련 필드들을 sha1 해시해 12자리 prefix 사용.
저장 위치: data/pool_cache/pool_<key>.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from src.config import get_config

CACHE_DIR = Path("data/pool_cache")
UNIVERSE_META_PATH = Path("data/universe_meta.csv")


def _meta_signature() -> str:
    if not UNIVERSE_META_PATH.exists():
        return "no-meta"
    st = UNIVERSE_META_PATH.stat()
    raw = f"{int(st.st_mtime)}:{st.st_size}".encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def _payload() -> dict:
    cfg = get_config()
    return {
        "start": cfg.backtest.start_date,
        "end": cfg.backtest.end_date,
        "factor_module": cfg.factors.factor_module,
        "rebalance_freq": cfg.factors.rebalance_freq,
        "top_n": cfg.factors.top_n,
        "composite_method": cfg.factors.composite_method,
        "ic_lookback": cfg.factors.ic_lookback,
        "min_ir": cfg.factors.min_ir,
        "neutralize_industry": cfg.factors.neutralize_industry,
        "neutralize_market_cap": cfg.factors.neutralize_market_cap,
        "market": cfg.universe.market,
        "min_market_cap": cfg.universe.min_market_cap,
        "min_avg_volume": cfg.universe.min_avg_volume,
        "exclude_sectors": cfg.universe.exclude_sectors,
        "pit_universe": True,
        "meta_signature": _meta_signature(),
        # B1: 투자자 필터 모드 변경 시 풀 구성이 달라지므로 캐시 무효화
        "foreign_filter_enabled": getattr(cfg.factors, "foreign_filter_enabled", False),
        "foreign_filter_pct": getattr(cfg.factors, "foreign_filter_pct", 0.5),
        "foreign_filter_lookback": getattr(cfg.factors, "foreign_filter_lookback", 20),
        "investor_filter_mode": getattr(cfg.factors, "investor_filter_mode", "foreign"),
    }


def cache_key() -> str:
    raw = json.dumps(_payload(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def cache_path(key: str | None = None) -> Path:
    return CACHE_DIR / f"pool_{key or cache_key()}.json"


def load(key: str | None = None, expected_dates: list[str] | None = None) -> dict[str, list[str]] | None:
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        pool_history = data.get("pool_history")
        if not isinstance(pool_history, dict) or not pool_history:
            return None
        if expected_dates is not None and set(pool_history.keys()) != set(expected_dates):
            logger.warning(
                f"종목풀 캐시 날짜 불일치 (캐시 {len(pool_history)}개 vs 요청 {len(expected_dates)}개)"
            )
            return None
        logger.info(f"종목풀 캐시 사용: {path.name} ({len(pool_history)}개 리밸런싱)")
        return pool_history
    except Exception as e:
        logger.warning(f"종목풀 캐시 로드 실패: {e}")
        return None


def save(pool_history: dict[str, list[str]], key: str | None = None, extra_meta: dict | None = None) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(key)
    payload = {
        "pool_history": pool_history,
        "meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "n_rebalances": len(pool_history),
            "config": _payload(),
            **(extra_meta or {}),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"종목풀 캐시 저장: {path.name}")
    return path
