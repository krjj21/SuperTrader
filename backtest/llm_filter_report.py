"""
LLM 필터 차단 효과 리포트

백테스트 엔진이 기록한 LLM 검증 결정(확정/보류)에 대해
- 차단군(보류) vs 통과군(확정) forward return 비교
- 필터 알파(= 통과군 평균 - 차단군 평균, SELL 은 부호 반전)
- 시그널별 hit rate / Spearman IC
를 계산하고 Markdown 리포트를 생성한다.

Forward return 기준: **T+1 시가 진입 → T+1+H 종가 청산**.
엔진이 T일 시그널을 T+1 시가로 체결하는 것과 동일 기준이라 실제 체결 수익과 일치한다.
(T+1 시가가 없으면 T+1 종가로 fallback — 엔진의 _get_open_cached 와 동일.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from loguru import logger


HORIZONS_DEFAULT: tuple[int, ...] = (1, 5, 20)


# ══════════════════════════════════════════════════════════
# Forward return 부착
# ══════════════════════════════════════════════════════════
def _build_price_cache(ohlcv_dict: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """종목별 date→(open, close) 매핑을 구축. 엔진의 astype(str) 과 동일한 키."""
    cache: dict[str, dict] = {}
    for code, df in ohlcv_dict.items():
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            continue
        dates = df["date"].astype(str).tolist()
        closes = df["close"].astype(float).tolist()
        opens = (
            df["open"].astype(float).tolist()
            if "open" in df.columns
            else list(closes)
        )
        cache[code] = {
            "dates": dates,
            "closes": closes,
            "opens": opens,
            "idx": {d: i for i, d in enumerate(dates)},
        }
    return cache


def attach_forward_returns(
    decisions: Sequence[dict],
    ohlcv_dict: dict[str, pd.DataFrame],
    horizons: Iterable[int] = HORIZONS_DEFAULT,
) -> list[dict]:
    """각 결정에 fwd_{h}d (%) 를 붙여 반환. forward 데이터가 없으면 None.

    기준: p0 = T+1 open (없으면 T+1 close), p1 = T+1+h close.
    엔진이 T일 시그널을 T+1 시가에 체결하는 것과 동일.
    시그널이 마지막 거래일이면 T+1 이 없어 스킵.
    """
    caches = _build_price_cache(ohlcv_dict)
    horizons = tuple(horizons)
    out: list[dict] = []
    for d in decisions:
        cache = caches.get(d["code"])
        if cache is None:
            continue
        idx = cache["idx"].get(d["date"])
        if idx is None:
            continue
        closes = cache["closes"]
        opens = cache["opens"]
        # T+1 진입: 다음 거래일이 없으면 스킵
        entry_idx = idx + 1
        if entry_idx >= len(closes):
            continue
        p0 = opens[entry_idx] if opens[entry_idx] > 0 else closes[entry_idx]
        if p0 <= 0:
            continue
        row = dict(d)
        for h in horizons:
            # T+1+h 종가
            exit_idx = entry_idx + h
            if exit_idx < len(closes) and closes[exit_idx] > 0:
                row[f"fwd_{h}d"] = (closes[exit_idx] / p0 - 1.0) * 100.0
            else:
                row[f"fwd_{h}d"] = None
        out.append(row)
    return out


# ══════════════════════════════════════════════════════════
# 요약 통계 계산
# ══════════════════════════════════════════════════════════
def _group_stats(df: pd.DataFrame, col: str) -> dict:
    """forward return 컬럼에 대해 count/mean/hit_rate 계산. NaN 은 제외."""
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return {"count": 0, "mean": float("nan"), "hit_rate": float("nan")}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "hit_rate": float((series > 0).mean()),
    }


def build_summary(
    decisions_with_returns: Sequence[dict],
    horizons: Iterable[int] = HORIZONS_DEFAULT,
) -> dict:
    """결정 데이터(+ forward return)로 요약 통계를 구축한다.

    반환 구조:
      {
        "total": N, "blocked": ..., "confirmed": ...,
        "per_signal": {
            "BUY": {"blocked": n, "confirmed": n, "horizons": {5: {...}}, "filter_alpha": {5: 알파%}},
            "SELL": { ... },
        },
        "spearman_ic": {"BUY": {5: (ic, p)}, "SELL": {...}},
      }
    """
    horizons = tuple(horizons)
    df = pd.DataFrame(list(decisions_with_returns))
    if df.empty:
        return {
            "total": 0, "blocked": 0, "confirmed": 0,
            "per_signal": {}, "spearman_ic": {},
        }

    total = int(len(df))
    blocked = int((df["decision"] == "보류").sum())
    confirmed = int((df["decision"] == "확정").sum())

    per_signal: dict[str, dict] = {}
    spearman: dict[str, dict] = {}

    for sig in ("BUY", "SELL"):
        sub = df[df["signal"] == sig]
        if sub.empty:
            continue

        n_blocked = int((sub["decision"] == "보류").sum())
        n_confirmed = int((sub["decision"] == "확정").sum())

        horizon_stats: dict[int, dict] = {}
        filter_alpha: dict[int, float] = {}
        for h in horizons:
            col = f"fwd_{h}d"
            if col not in sub.columns:
                continue
            conf_stats = _group_stats(sub[sub["decision"] == "확정"], col)
            block_stats = _group_stats(sub[sub["decision"] == "보류"], col)
            horizon_stats[h] = {"confirmed": conf_stats, "blocked": block_stats}

            # filter alpha: 통과군과 차단군의 평균 차이 (SELL 은 부호 반전)
            if conf_stats["count"] > 0 and block_stats["count"] > 0:
                diff = conf_stats["mean"] - block_stats["mean"]
                filter_alpha[h] = diff if sig == "BUY" else -diff

        per_signal[sig] = {
            "total": int(len(sub)),
            "blocked": n_blocked,
            "confirmed": n_confirmed,
            "horizons": horizon_stats,
            "filter_alpha": filter_alpha,
        }

        # Spearman IC: 결정(BUY 확정=+1, BUY 보류=-1 / SELL 반대) vs forward return
        smap = {"확정": 1, "보류": -1} if sig == "BUY" else {"확정": -1, "보류": 1}
        sub2 = sub[sub["decision"].isin(smap.keys())].copy()
        if len(sub2) >= 10 and sub2["decision"].nunique() >= 2:
            sub2["score"] = sub2["decision"].map(smap)
            ic_per_h: dict[int, tuple] = {}
            try:
                from scipy import stats as sp_stats
            except ImportError:
                sp_stats = None
            if sp_stats is not None:
                for h in horizons:
                    col = f"fwd_{h}d"
                    if col not in sub2.columns:
                        continue
                    pair = sub2[["score", col]].dropna()
                    if len(pair) >= 10 and pair["score"].nunique() >= 2:
                        try:
                            ic, p = sp_stats.spearmanr(pair["score"], pair[col])
                            ic_per_h[h] = (float(ic), float(p))
                        except Exception:
                            continue
            if ic_per_h:
                spearman[sig] = ic_per_h

    return {
        "total": total,
        "blocked": blocked,
        "confirmed": confirmed,
        "per_signal": per_signal,
        "spearman_ic": spearman,
    }


# ══════════════════════════════════════════════════════════
# Markdown 포맷팅
# ══════════════════════════════════════════════════════════
def format_markdown(
    summary: dict,
    strategy: str,
    filter_mode: str,
    horizons: Iterable[int] = HORIZONS_DEFAULT,
) -> str:
    horizons = tuple(horizons)
    lines: list[str] = []
    lines.append(f"# LLM 필터 차단 효과 리포트 — `{strategy}` ({filter_mode})")
    lines.append("")
    lines.append(
        f"- 총 검증 시그널: **{summary['total']:,}건**  "
        f"(확정 {summary['confirmed']:,} / 보류 {summary['blocked']:,})"
    )
    if summary["total"] == 0:
        lines.append("")
        lines.append("> 검증된 시그널 없음 — 필터가 비활성이거나 시그널이 발생하지 않음.")
        return "\n".join(lines)

    for sig in ("BUY", "SELL"):
        ps = summary["per_signal"].get(sig)
        if not ps:
            continue
        lines.append("")
        lines.append(f"## {sig} — 총 {ps['total']:,}건 (확정 {ps['confirmed']:,} / 보류 {ps['blocked']:,})")
        lines.append("")
        # horizon 별 평균 / hit rate 테이블
        header = "| horizon | 그룹 | n | 평균 forward (%) | hit rate (>0) |"
        lines.append(header)
        lines.append("|---|---|---:|---:|---:|")
        for h in horizons:
            hs = ps["horizons"].get(h)
            if not hs:
                continue
            for group_key, label in (("confirmed", "확정"), ("blocked", "보류")):
                s = hs[group_key]
                if s["count"] == 0:
                    lines.append(f"| {h}d | {label} | 0 | — | — |")
                else:
                    lines.append(
                        f"| {h}d | {label} | {s['count']} | "
                        f"{s['mean']:+.2f} | {s['hit_rate']*100:.1f}% |"
                    )

        # 필터 알파
        if ps["filter_alpha"]:
            lines.append("")
            lines.append("**필터 알파 (+ 일수록 필터가 차단으로 수익을 개선)**")
            lines.append("")
            alpha_parts = [f"{h}d: `{v:+.2f}%`" for h, v in sorted(ps["filter_alpha"].items())]
            lines.append("- " + " / ".join(alpha_parts))

        # Spearman IC
        ic = summary["spearman_ic"].get(sig)
        if ic:
            lines.append("")
            lines.append("**Spearman IC (결정 부호 vs forward return)**")
            lines.append("")
            for h in sorted(ic.keys()):
                val, p = ic[h]
                signif = "**" if p < 0.05 else ("*" if p < 0.1 else "")
                lines.append(f"- {h}d: IC={val:+.4f}, p={p:.3f} {signif}")

    lines.append("")
    lines.append("_Forward return 기준: T+1 시가 진입 → T+1+H 종가 청산 (엔진 체결과 동일)._")
    lines.append("_BUY 필터 알파가 양수면 차단 시그널의 체결 가정 수익이 실제로 낮았다는 뜻 (필터 유효)._")
    lines.append("_SELL 필터 알파가 양수면 차단 후 유지했을 때 수익이 더 좋았다는 뜻 (필터가 홀딩 유도)._")
    return "\n".join(lines)


def save_report(markdown: str, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return out


# ══════════════════════════════════════════════════════════
# 공개 API — 한 번에 결정 → (summary, markdown) 반환
# ══════════════════════════════════════════════════════════
def generate_report(
    decisions: Sequence[dict],
    ohlcv_dict: dict[str, pd.DataFrame],
    strategy: str,
    filter_mode: str,
    horizons: Iterable[int] = HORIZONS_DEFAULT,
) -> tuple[dict, str]:
    """결정 + OHLCV 로부터 summary 와 markdown 을 생성한다."""
    enriched = attach_forward_returns(decisions, ohlcv_dict, horizons=horizons)
    if not enriched:
        logger.warning(f"[{strategy}] LLM 필터 결정 없음 — 리포트 스킵")
        summary = {"total": 0, "blocked": 0, "confirmed": 0, "per_signal": {}, "spearman_ic": {}}
    else:
        summary = build_summary(enriched, horizons=horizons)
    markdown = format_markdown(summary, strategy, filter_mode, horizons=horizons)
    return summary, markdown


def comparison_columns(summary: dict, primary_horizon: int = 5) -> dict[str, float | int]:
    """전략 비교표에 붙일 한 줄 통계를 반환한다.

    - buy_blocks / sell_blocks: 차단 건수
    - buy_filter_alpha_{h}d / sell_filter_alpha_{h}d: 필터 알파(%)
    """
    cols: dict[str, float | int] = {}
    for sig in ("BUY", "SELL"):
        ps = summary.get("per_signal", {}).get(sig)
        key = sig.lower()
        if not ps:
            cols[f"{key}_blocks"] = 0
            cols[f"{key}_filter_alpha_{primary_horizon}d"] = float("nan")
            continue
        cols[f"{key}_blocks"] = ps["blocked"]
        alpha = ps.get("filter_alpha", {}).get(primary_horizon)
        cols[f"{key}_filter_alpha_{primary_horizon}d"] = (
            float(alpha) if alpha is not None else float("nan")
        )
    return cols
