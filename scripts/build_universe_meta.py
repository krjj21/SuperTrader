"""
Build point-in-time universe metadata.

Combines FDR `KRX-DESC` (active stocks with ListingDate) and `KRX-DELISTING`
(historical stocks with both ListingDate and DelistingDate) into a single
parquet at data/universe_meta.parquet, plus a sidecar JSON with build info.

Usage:
    /mnt/e/SuperTrader/venv/Scripts/python.exe scripts/build_universe_meta.py [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import FinanceDataReader as fdr
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_CSV = ROOT / "data" / "universe_meta.csv"
OUT_META = ROOT / "data" / "universe_meta.meta.json"

EXCLUDE_NAME_PATTERN = r"스팩|리츠|ETF|ETN"
EXCLUDE_CODE_SUFFIXES = {"5", "7", "9", "K", "L"}
KEEP_MARKETS = {"KOSPI", "KOSDAQ"}


def _normalize_active() -> pd.DataFrame:
    raw = fdr.StockListing("KRX-DESC")
    if raw.empty:
        raise RuntimeError("KRX-DESC returned empty — FDR upstream issue")
    df = pd.DataFrame()
    df["code"] = raw["Code"].astype(str).str.zfill(6)
    df["name"] = raw["Name"].astype(str)
    df["market"] = raw["Market"].astype(str)
    df["listed_date"] = pd.to_datetime(raw["ListingDate"], errors="coerce")
    df["delisted_date"] = pd.NaT
    return df


def _normalize_delisted() -> pd.DataFrame:
    raw = fdr.StockListing("KRX-DELISTING")
    if raw.empty:
        raise RuntimeError("KRX-DELISTING returned empty — FDR upstream issue")
    df = pd.DataFrame()
    df["code"] = raw["Symbol"].astype(str).str.zfill(6)
    df["name"] = raw["Name"].astype(str)
    df["market"] = raw["Market"].astype(str)
    df["listed_date"] = pd.to_datetime(raw["ListingDate"], errors="coerce")
    df["delisted_date"] = pd.to_datetime(raw["DelistingDate"], errors="coerce")
    return df


def _apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["market"].isin(KEEP_MARKETS)]
    df = df[~df["name"].str.contains(EXCLUDE_NAME_PATTERN, na=False)]
    df = df[~df["code"].str[-1].isin(EXCLUDE_CODE_SUFFIXES)]
    df = df[df["listed_date"].notna()]
    return df


def build() -> pd.DataFrame:
    active = _apply_exclusions(_normalize_active())
    delisted = _apply_exclusions(_normalize_delisted())

    n_active_raw = len(active)
    n_delisted_raw = len(delisted)

    combined = pd.concat([active, delisted], ignore_index=True)
    combined = combined.sort_values("listed_date", ascending=False)
    combined = combined.drop_duplicates(subset=["code"], keep="first")
    combined = combined.sort_values(["market", "code"]).reset_index(drop=True)

    logger.info(
        f"universe_meta: active={n_active_raw} delisted={n_delisted_raw} "
        f"combined={len(combined)} (post-dedup)"
    )
    return combined


def write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    out = df.copy()
    out["listed_date"] = out["listed_date"].dt.strftime("%Y-%m-%d")
    out["delisted_date"] = out["delisted_date"].dt.strftime("%Y-%m-%d")
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="rebuild even if file exists")
    args = parser.parse_args()

    if OUT_CSV.exists() and not args.force:
        logger.info(f"{OUT_CSV} already exists — skipping (use --force to rebuild)")
        return 0

    df = build()
    write_atomic(df, OUT_CSV)

    meta = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "fdr_version": getattr(fdr, "__version__", "unknown"),
        "n_total": int(len(df)),
        "n_active": int(df["delisted_date"].isna().sum()),
        "n_delisted": int(df["delisted_date"].notna().sum()),
        "earliest_listed": str(df["listed_date"].min().date()) if not df.empty else None,
        "latest_delisted": (
            str(df["delisted_date"].max().date())
            if df["delisted_date"].notna().any() else None
        ),
    }
    with OUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"wrote {OUT_CSV} ({meta['n_total']} rows)")
    logger.info(f"  active={meta['n_active']} delisted={meta['n_delisted']}")
    logger.info(f"  earliest_listed={meta['earliest_listed']} latest_delisted={meta['latest_delisted']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
