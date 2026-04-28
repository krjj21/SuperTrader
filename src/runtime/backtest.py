"""백테스트 모드 — 팩터 + ML 타이밍 전략 비교.

main.py 에서 분리됨 (2026-04-28). 외부 진입점:
  · run_backtest(only_strategies, review, llm_filter)

내부 헬퍼:
  · _generate_rebalance_dates(start, end, freq)
  · _build_pool_history_factor_based(...) — pool cache + rolling IC 누적
  · _train_models_if_needed(ohlcv_dict)   — DT/XGB/RL 모델 보장
  · _post_backtest_slack / _trigger_codex_backtest_review
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_config


# ──────────────────────────────────────────────
# 공개 진입점
# ──────────────────────────────────────────────
def run_backtest(
    only_strategies: list[str] | None = None,
    review: bool = False,
    llm_filter: str | None = None,
) -> None:
    """팩터 + 타이밍 전략 비교 백테스트를 실행합니다."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("SuperTrader 백테스트 시작")
    logger.info("=" * 60)

    from src.data.market_data import get_universe, get_ohlcv_batch, filter_by_listing_date
    from src.factors.stock_pool import build_stock_pool
    from backtest.comparison import run_strategy_comparison
    from backtest.report import print_comparison_table

    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")

    # 1. 리밸런싱 스케줄 생성
    logger.info("Step 1: 리밸런싱 스케줄 생성")
    rebalance_dates = _generate_rebalance_dates(start, end, config.factors.rebalance_freq)
    logger.info(f"리밸런싱 날짜: {len(rebalance_dates)}회")

    # 2. 유니버스 구성 — PIT union
    logger.info("Step 2: 유니버스 구성 (PIT)")
    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        legacy = get_universe()
        if legacy.empty:
            logger.error("유니버스가 비어 있습니다")
            return
        all_codes = set(legacy["code"].tolist())
    codes = sorted(all_codes)
    logger.info(f"유니버스: {len(codes)}종목 (rebalance union)")

    # 3. OHLCV 데이터 로드
    logger.info("Step 3: OHLCV 데이터 로드")
    ohlcv_dict = get_ohlcv_batch(codes, start, end)
    logger.info(f"OHLCV 로드: {len(ohlcv_dict)}종목")

    # 3-1. IPO 이후 상장 종목 제외 (PIT meta 가 있으면 사실상 no-op)
    ohlcv_dict = filter_by_listing_date(ohlcv_dict, start)

    # 4. 종목풀 히스토리 구축
    logger.info("Step 4: 종목풀 히스토리 구축 (팩터 기반)")
    pool_history = _build_pool_history_factor_based(
        ohlcv_dict, rebalance_dates, build_stock_pool,
    )

    # 5. ML 모델 학습 (있으면)
    logger.info("Step 5: ML 모델 학습")
    model_paths = _train_models_if_needed(ohlcv_dict)

    # 6. 전략 비교 실행
    logger.info("Step 6: 전략 비교 백테스트 실행")
    comparison = run_strategy_comparison(
        ohlcv_dict, pool_history, rebalance_dates, model_paths,
        only_strategies=only_strategies,
        llm_filter=llm_filter,
    )

    if not comparison.empty:
        table_text = print_comparison_table(comparison)
        logger.info("백테스트 완료!")

        _post_backtest_slack(table_text, comparison, llm_filter=llm_filter)

        if review:
            _trigger_codex_backtest_review(comparison)
    else:
        logger.warning("백테스트 결과 없음")


# ──────────────────────────────────────────────
# Slack / Codex
# ──────────────────────────────────────────────
def _post_backtest_slack(
    table_text: str,
    comparison: pd.DataFrame,
    llm_filter: str | None = None,
) -> None:
    """백테스트 결과 비교표를 Slack으로 전송한다. 실패해도 백테스트 흐름은 계속된다."""
    try:
        from src.notification.slack_bot import SlackNotifier
        notifier = SlackNotifier()
        if not notifier.token:
            logger.info("Slack 미설정 — 전송 건너뜀")
            return

        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = f"🧪 *백테스트 결과* — {stamp}"
        filter_note = f" (LLM 필터: `{llm_filter}`)" if llm_filter else ""
        header += filter_note

        body = table_text.strip()
        if len(body) > 3500:
            body = body[:3500] + "\n...(truncated)"

        msg = f"{header}\n```\n{body}\n```"
        ok = notifier._send(msg)
        if ok:
            logger.info("백테스트 결과 Slack 전송 완료")
        else:
            logger.warning("백테스트 결과 Slack 전송 실패")
    except Exception as e:
        logger.warning(f"Slack 전송 예외: {e}")


def _trigger_codex_backtest_review(comparison: pd.DataFrame) -> None:
    """백테스트 완료 후 codex 리뷰를 트리거합니다."""
    import subprocess

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f"backtest_comparison_{stamp}.csv"
    comparison.to_csv(csv_path, encoding="utf-8")
    logger.info(f"비교 테이블 저장: {csv_path}")

    cmd = [
        sys.executable,
        "scripts/codex_review.py",
        "backtest",
        "--comparison", str(csv_path),
    ]
    logger.info(f"Codex 리뷰 실행: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        logger.warning("codex CLI 미설치 — 리뷰 단계 건너뜀")


# ──────────────────────────────────────────────
# Helpers — 풀 히스토리 / 모델 / 날짜
# ──────────────────────────────────────────────
def _forward_returns_between(
    ohlcv_dict: dict[str, pd.DataFrame],
    from_date: str,
    to_date: str,
) -> pd.Series:
    """from_date 의 close 대비 to_date 의 close 수익률을 종목별로 계산합니다."""
    fd = pd.Timestamp(from_date.replace("-", ""))
    td = pd.Timestamp(to_date.replace("-", ""))
    result: dict[str, float] = {}
    for code, df in ohlcv_dict.items():
        if "date" not in df.columns or df.empty:
            continue
        sub = df[(df["date"] >= fd) & (df["date"] <= td)]
        if len(sub) < 2:
            continue
        p0 = float(sub["close"].iloc[0])
        p1 = float(sub["close"].iloc[-1])
        if p0 > 0:
            result[code] = (p1 / p0) - 1.0
    return pd.Series(result)


def _build_pool_history_factor_based(
    ohlcv_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[str],
    build_stock_pool_fn,
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """팩터 파이프라인 기반으로 종목풀을 구성합니다 (라이브 매매와 동일).

    rolling IC(최근 ic_lookback 리밸런싱) 을 계산해 `ic_weighted` 가중이
    실제로 작동하도록 factor_report 를 누적·전달한다.

    동일 config 로 이미 빌드한 기록이 있으면 data/pool_cache/ 에서 로드한다.
    """
    if use_cache:
        from src.factors import pool_cache
        cached = pool_cache.load(expected_dates=rebalance_dates)
        if cached is not None:
            return cached

    from src.factors.validity import validate_all_factors

    config = get_config()
    ic_lookback = int(config.factors.ic_lookback)

    pool_history: dict[str, list[str]] = {}
    factor_history: dict[str, pd.DataFrame] = {}
    return_history: dict[str, pd.Series] = {}
    factor_report: pd.DataFrame | None = None
    previous_pool = None

    for i, date in enumerate(rebalance_dates):
        date_fmt = date.replace("-", "")
        logger.info(f"  종목풀 구성 {i+1}/{len(rebalance_dates)}: {date}")

        if i > 0:
            prev_date = rebalance_dates[i - 1].replace("-", "")
            if prev_date in factor_history:
                ret = _forward_returns_between(ohlcv_dict, prev_date, date_fmt)
                if not ret.empty:
                    return_history[prev_date] = ret

        if len(return_history) >= ic_lookback:
            recent = sorted(return_history.keys())[-ic_lookback:]
            rolling_fh = {d: factor_history[d] for d in recent if d in factor_history}
            rolling_rh = {d: return_history[d] for d in recent}
            if len(rolling_fh) >= max(3, ic_lookback // 2):
                try:
                    factor_report = validate_all_factors(
                        rolling_fh, rolling_rh,
                        min_ir=config.factors.min_ir,
                    )
                except Exception as e:
                    logger.warning(f"IC 리포트 계산 실패 ({date}): {e}")

        result = build_stock_pool_fn(
            date=date_fmt,
            previous_pool=previous_pool,
            ohlcv_dict=ohlcv_dict,
            factor_report=factor_report,
            return_factors=True,
        )
        if isinstance(result, tuple):
            pool, factor_df = result
            if factor_df is not None and not factor_df.empty:
                factor_history[date_fmt] = factor_df
        else:
            pool = result

        if pool.codes:
            pool_history[date] = pool.codes
            previous_pool = pool
        else:
            if previous_pool and previous_pool.codes:
                pool_history[date] = previous_pool.codes
                logger.warning(
                    f"  {date}: 팩터 계산 실패, 이전 종목풀 유지 ({len(previous_pool.codes)}종목)"
                )
            else:
                pool_history[date] = list(ohlcv_dict.keys())[:30]
                logger.warning(f"  {date}: 팩터 계산 실패, 가용 종목으로 대체")

    if use_cache and pool_history:
        try:
            from src.factors import pool_cache
            pool_cache.save(pool_history, extra_meta={"n_stocks_loaded": len(ohlcv_dict)})
        except Exception as e:
            logger.warning(f"종목풀 캐시 저장 실패: {e}")

    return pool_history


def _generate_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """리밸런싱 날짜를 생성합니다."""
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    if freq == "biweekly":
        dates = []
        current = start_dt.replace(day=1)
        while current <= end_dt:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=14)
        return dates

    dates = []
    current = start_dt.replace(day=1)
    step = 1 if freq == "monthly" else 3

    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        month = current.month + step
        year = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        current = current.replace(year=year, month=month, day=1)

    return dates


def _split_ohlcv_prefix(
    ohlcv_dict: dict[str, pd.DataFrame],
    ratio: float,
) -> dict[str, pd.DataFrame]:
    """각 종목 OHLCV 의 앞 `ratio` 비율만 남겨 반환합니다 (walk-forward 학습용)."""
    if ratio >= 1.0:
        return ohlcv_dict
    out = {}
    for code, df in ohlcv_dict.items():
        n = int(len(df) * ratio)
        if n > 60:
            out[code] = df.iloc[:n].reset_index(drop=True)
    return out


def _train_models_if_needed(ohlcv_dict: dict) -> dict[str, str]:
    """ML 모델이 없으면 학습합니다 (walk-forward prefix 만 사용)."""
    config = get_config()
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_paths = {}
    models_to_train = ["decision_tree", "xgboost", "rl"]

    train_ratio = float(getattr(config.backtest, "train_ratio", 1.0))
    train_ohlcv = _split_ohlcv_prefix(ohlcv_dict, train_ratio) if train_ratio < 1.0 else ohlcv_dict
    if train_ratio < 1.0:
        logger.info(
            f"walk-forward 학습: 전체의 {train_ratio*100:.0f}% 앞부분만 사용 "
            f"({len(train_ohlcv)}/{len(ohlcv_dict)}종목 통과)"
        )

    for model_type in models_to_train:
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        path = str(model_dir / f"{model_type}_timing{ext}")

        if Path(path).exists():
            model_paths[model_type] = path
            logger.info(f"기존 모델 사용: {model_type} ({path})")
            continue

        logger.info(f"모델 학습 시작: {model_type}")
        from src.timing.trainer import train_timing_model
        result = train_timing_model(train_ohlcv, model_type, save_path=path)

        if "error" not in result:
            model_paths[model_type] = path
            logger.info(f"모델 학습 완료: {model_type}")
        else:
            logger.warning(f"모델 학습 실패: {model_type} - {result}")

    return model_paths
