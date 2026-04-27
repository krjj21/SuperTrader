"""
SuperTrader 메인 진입점
- 백테스트 모드: 팩터 + ML 타이밍 전략 비교
- 라이브 모드: 실시간 자동매매
- 학습 모드: ML 타이밍 모델 학습
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import load_config, get_config


def setup_logging():
    config = get_config()
    logger.remove()
    logger.add(sys.stderr, level=config.logging.level)
    Path(config.logging.file).parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        config.logging.file,
        level=config.logging.level,
        rotation=config.logging.rotation,
        retention=config.logging.retention,
    )


# ═══════════════════════════════════════════════
# 백테스트 모드
# ═══════════════════════════════════════════════
def run_backtest(only_strategies: list[str] | None = None, review: bool = False, llm_filter: str | None = None):
    """팩터 + 타이밍 전략 비교 백테스트를 실행합니다."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("SuperTrader 백테스트 시작")
    logger.info("=" * 60)

    from src.data.market_data import get_universe, get_ohlcv_batch, filter_by_listing_date
    from src.factors.alpha101 import compute_all_factors
    from src.factors.stock_pool import build_stock_pool
    from backtest.comparison import run_strategy_comparison
    from backtest.report import plot_equity_comparison, print_comparison_table

    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")

    # 1. 리밸런싱 스케줄 생성 (PIT 유니버스 시드용으로 먼저 계산)
    logger.info("Step 1: 리밸런싱 스케줄 생성")
    rebalance_dates = _generate_rebalance_dates(start, end, config.factors.rebalance_freq)
    logger.info(f"리밸런싱 날짜: {len(rebalance_dates)}회")

    # 2. 유니버스 구성 — 모든 리밸런싱 날짜의 PIT 멤버십 union
    #    (폐지 종목 포함 — universe_meta.csv 가 있으면 자동, 없으면 legacy 단일 호출)
    logger.info("Step 2: 유니버스 구성 (PIT)")
    all_codes: set[str] = set()
    for d in rebalance_dates:
        u = get_universe(d.replace("-", ""))
        if not u.empty:
            all_codes.update(u["code"].tolist())
    if not all_codes:
        # meta 부재 + legacy 도 실패한 비정상 케이스
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

    # 4. 종목풀 히스토리 구축 (팩터 파이프라인 — 라이브 매매와 동일)
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


def _post_backtest_slack(table_text: str, comparison: pd.DataFrame, llm_filter: str | None = None) -> None:
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

        # Slack 은 4000자 제한. 큰 표는 잘라낸다.
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


def _trigger_codex_daily_review(model: str = "") -> None:
    """일일 리포트 이후 codex daily 리뷰를 트리거합니다."""
    import subprocess

    cmd = [sys.executable, "scripts/codex_review.py", "daily"]
    if model:
        cmd.extend(["--model", model])
    logger.info(f"Codex 일일 리뷰 실행: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        logger.warning("codex CLI 미설치 — 리뷰 단계 건너뜀")
    except Exception as e:
        logger.warning(f"Codex 일일 리뷰 실패: {e}")


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
        # YYYYMMDD 형식으로 변환
        date_fmt = date.replace("-", "")
        logger.info(f"  종목풀 구성 {i+1}/{len(rebalance_dates)}: {date}")

        # 이전 리밸런싱 기간의 forward return 을 누적
        if i > 0:
            prev_date = rebalance_dates[i - 1].replace("-", "")
            if prev_date in factor_history:
                ret = _forward_returns_between(ohlcv_dict, prev_date, date_fmt)
                if not ret.empty:
                    return_history[prev_date] = ret

        # ic_lookback 기간 이상 누적되면 rolling IC 리포트 재계산
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
            # 팩터 계산 실패 시 이전 종목풀 유지
            if previous_pool and previous_pool.codes:
                pool_history[date] = previous_pool.codes
                logger.warning(f"  {date}: 팩터 계산 실패, 이전 종목풀 유지 ({len(previous_pool.codes)}종목)")
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
    from datetime import timedelta

    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

    if freq == "biweekly":
        # 2주 간격: 시작일의 월초부터 14일 간격
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


def _split_ohlcv_prefix(ohlcv_dict: dict[str, pd.DataFrame], ratio: float) -> dict[str, pd.DataFrame]:
    """각 종목 OHLCV 의 앞 `ratio` 비율만 남겨 반환합니다 (walk-forward 학습용)."""
    if ratio >= 1.0:
        return ohlcv_dict
    out = {}
    for code, df in ohlcv_dict.items():
        n = int(len(df) * ratio)
        if n > 60:  # 최소 학습 가능 길이
            out[code] = df.iloc[:n].reset_index(drop=True)
    return out


def _train_models_if_needed(ohlcv_dict: dict) -> dict[str, str]:
    """ML 모델이 없으면 학습합니다 (walk-forward prefix 만 사용)."""
    config = get_config()
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_paths = {}
    models_to_train = ["decision_tree", "xgboost", "rl"]  # DT, XGB, RL 학습

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


# ═══════════════════════════════════════════════
# 라이브 매매 모드
# ═══════════════════════════════════════════════
def run_live():
    """실시간 자동매매를 실행합니다."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("SuperTrader 라이브 매매 시작")
    logger.info("=" * 60)

    from apscheduler.schedulers.blocking import BlockingScheduler
    from src.broker.kis_client import KISClient
    from src.broker.order import OrderManager
    from src.broker.account import AccountManager
    from src.risk.manager import RiskManager
    from src.notification.slack_bot import SlackNotifier
    from src.db.models import (
        init_db, save_runtime_status, save_signal_log, save_daily_pnl,
        save_trade, save_holding, remove_holding,
    )

    # 초기화
    init_db(config.database.path)
    client = KISClient()
    order_mgr = OrderManager(client)
    account_mgr = AccountManager(client)
    risk_mgr = RiskManager(account_mgr)
    notifier = SlackNotifier()

    # 전략 생성
    strategy = _create_live_strategy(config)

    # 팩터 기반 종목풀 구성
    from src.factors.stock_pool import build_stock_pool
    _current_pool = [None]  # mutable container for closure
    _stock_names: dict[str, str] = {}  # 종목코드 → 종목명 캐시
    validator = None

    # ── Regime Detector 컨테이너 (label 문자열 또는 None) ──
    _current_regime: list[str | None] = [None]
    if config.regime.enabled:
        try:
            from src.db.sappo_models import init_sappo_db, get_latest_regime
            init_sappo_db(config.database.path)
            prev = get_latest_regime()
            if prev is not None:
                _current_regime[0] = prev.label
                logger.info(
                    f"[REGIME] 이전 regime 복원: label={prev.label} date={prev.date} "
                    f"(lambda={config.regime.lambda_})"
                )
            else:
                logger.info("[REGIME] 이전 라벨 없음 — 첫 detect_regime_daily 까지 영향 없음")
        except Exception as e:
            logger.warning(f"[REGIME] 시작 시 regime 복원 실패: {e}")

    def sync_runtime_status() -> None:
        pool = _current_pool[0]
        save_runtime_status(
            strategy=config.strategy.name,
            pool_size=len(pool.codes) if pool else 0,
            llm_enabled=bool(validator and validator.is_enabled),
            kill_switch=risk_mgr.kill_switch_active,
            check_interval=config.schedule.check_interval_sec,
            daily_loss_limit=config.risk.daily_loss_limit_pct * 100,
            max_positions=config.risk.max_total_positions,
        )

    def rebalance_pool():
        """팩터 기반 종목풀을 리밸런싱합니다."""
        today = datetime.now().strftime("%Y%m%d")
        try:
            pool = build_stock_pool(
                today, previous_pool=_current_pool[0],
                regime_label=_current_regime[0],
            )
            if pool.codes:
                strategy.update_pool(pool.codes)
                _current_pool[0] = pool

                # 종목명 캐시 업데이트 (pykrx)
                from pykrx import stock as krx
                for code in pool.codes:
                    if code not in _stock_names:
                        try:
                            _stock_names[code] = krx.get_market_ticker_name(code)
                        except Exception:
                            _stock_names[code] = ""

                # 종목풀 JSON 저장 (대시보드용)
                import json
                pool_data = [
                    {"code": c, "name": _stock_names.get(c, ""), "score": round(pool.scores.get(c, 0), 4)}
                    for c in pool.codes
                ]
                pool_json = {
                    "date": today,
                    "count": len(pool.codes),
                    "entered": pool.entered,
                    "exited": pool.exited,
                    "stocks": pool_data,
                }
                Path("data").mkdir(exist_ok=True)
                Path("data/current_pool.json").write_text(json.dumps(pool_json, ensure_ascii=False, indent=2))

                logger.info(f"종목풀 업데이트: {len(pool.codes)}종목 (신규 {len(pool.entered)}, 퇴출 {len(pool.exited)})")
                notifier.notify_start()  # 종목풀 갱신 알림
                sync_runtime_status()
            else:
                logger.warning("종목풀이 비어 있습니다 — 기존 풀 유지")
                sync_runtime_status()
        except Exception as e:
            logger.error(f"종목풀 구성 실패: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e), "종목풀 리밸런싱")

    def _restore_pool_from_disk() -> bool:
        """data/current_pool.json 에서 이전 풀을 복원합니다.

        재시작 시 기존 풀을 유지해 pool-diff 매매 폭증을 방지.
        리밸런싱은 cron 일정(매월 1일)에만 실행.
        성공 시 True, 실패 또는 풀 이상(<10종목) 시 False.
        """
        from src.factors.stock_pool import StockPool
        from src.utils.json_io import load_json_with_fallback
        data = load_json_with_fallback("data/current_pool.json")
        if data is None:
            return False
        try:
            stocks = data.get("stocks", [])
            if len(stocks) < 10:
                logger.warning(f"저장된 풀이 {len(stocks)}종목뿐 — 재빌드")
                return False
            codes = [s["code"] for s in stocks]
            scores = {s["code"]: s.get("score", 0.0) for s in stocks}
            pool = StockPool(
                date=data.get("date", ""),
                codes=codes,
                scores=scores,
                entered=[],
                exited=[],
            )
            strategy.update_pool(pool.codes)
            _current_pool[0] = pool
            for s in stocks:
                if s.get("name"):
                    _stock_names[s["code"]] = s["name"]
            logger.info(
                f"이전 종목풀 복원: {len(pool.codes)}종목 (date={pool.date}) "
                f"— 리밸런싱은 cron 일정에서만 실행"
            )
            sync_runtime_status()
            return True
        except Exception as e:
            logger.warning(f"풀 복원 실패: {e} — 재빌드")
            return False

    # 시작 시: 기존 풀 복원 시도, 실패하면 새로 빌드
    logger.info("종목풀 복원/구성 중...")
    if not _restore_pool_from_disk():
        rebalance_pool()

    # LLM 시그널 검증기
    from src.timing.llm_validator import SignalValidator
    validator = SignalValidator()
    if validator.is_enabled:
        logger.info("LLM 시그널 검증 활성화 (Claude Haiku)")
    else:
        logger.info("LLM 시그널 검증 비활성화 — ANTHROPIC_API_KEY 미설정")

    # Notion 일일 리포터
    from src.notification.notion_reporter import NotionReporter
    notion = NotionReporter()
    if notion.is_enabled:
        logger.info("Notion 일일 리포트 활성화")
    else:
        logger.info("Notion 일일 리포트 비활성화 — NOTION_TOKEN/NOTION_DATABASE_ID 미설정")

    notifier.notify_start()
    sync_runtime_status()

    # 스케줄러
    scheduler = BlockingScheduler()

    def check_signals():
        """주기적 시그널 체크"""
        if not risk_mgr.is_trading_allowed:
            return

        try:
            balance = account_mgr.get_balance()
            risk_mgr.check_daily_loss_limit(balance.total_pnl, balance.total_eval)
            sync_runtime_status()

            # 종목명 캐시 업데이트
            for pos in balance.positions:
                if pos.stock_name and pos.stock_code not in _stock_names:
                    _stock_names[pos.stock_code] = pos.stock_name

            # 포지션 가격 스냅샷 (balance 시점) — sync_positions 에 재사용
            _pos_prices = {p.stock_code: float(p.current_price) for p in balance.positions}
            _pos_avg = {p.stock_code: float(p.avg_price) for p in balance.positions}

            def _sync_strategy_now(held: set[str]) -> None:
                """보유 종목 변경 직후 전략 내부 상태를 즉시 동기화."""
                if not hasattr(strategy, "sync_positions"):
                    return
                try:
                    strategy.sync_positions(held, _pos_prices, avg_prices=_pos_avg)
                except Exception as e:
                    logger.debug(f"sync_positions 실패: {e}")

            # ── 손절 체크 (LLM 검증 없이 즉시 실행) ──
            stop_codes = set(risk_mgr.check_stop_loss(balance.positions))
            # 손절된 종목은 같은 사이클에서 재매수 방지
            held_codes = {
                pos.stock_code for pos in balance.positions
                if pos.stock_code not in stop_codes
            }
            for pos in balance.positions:
                if pos.stock_code not in stop_codes:
                    continue
                try:
                    order = order_mgr.sell(pos.stock_code, pos.quantity, reference_price=int(pos.current_price))
                    if order.order_no:
                        notifier.notify_stop_loss(pos)
                        logger.info(
                            f"손절 매도: {pos.stock_name}({pos.stock_code}) "
                            f"{pos.pnl_pct:.2f}% / {pos.pnl:,}원"
                        )
                        save_trade(
                            stock_code=pos.stock_code,
                            stock_name=pos.stock_name,
                            side="sell",
                            quantity=pos.quantity,
                            price=int(pos.current_price),
                            order_no=order.order_no,
                            strategy=config.strategy.name,
                            signal_reason=f"stop_loss {pos.pnl_pct:.2f}%",
                        )
                        remove_holding(pos.stock_code)
                        held_codes.discard(pos.stock_code)
                        _sync_strategy_now(held_codes)
                except Exception as e:
                    logger.error(f"손절 매도 실패: {pos.stock_code} - {e}")

            # 손절이 한 건도 없었어도 한 번은 동기화 (재시작/편집 후 초기화용)
            if not stop_codes:
                _sync_strategy_now(held_codes)

            # 종목풀 내 종목에 대해 시그널 체크
            pool_codes = list(strategy._pool) if hasattr(strategy, '_pool') else []

            # 풀에서 퇴출된 보유 종목은 강제 매도 (리밸런싱 누락 방지)
            exited_codes = held_codes - set(pool_codes)
            if exited_codes:
                logger.info(f"풀 퇴출 종목 강제 매도: {len(exited_codes)}종목")
                for pos in balance.positions:
                    if pos.stock_code not in exited_codes:
                        continue
                    try:
                        order = order_mgr.sell(pos.stock_code, pos.quantity, reference_price=int(pos.current_price))
                        if order.order_no:
                            notifier.notify_order_filled(order)
                            logger.info(
                                f"풀 퇴출 매도: {pos.stock_name}({pos.stock_code}) "
                                f"{pos.quantity}주"
                            )
                            save_trade(
                                stock_code=pos.stock_code,
                                stock_name=pos.stock_name,
                                side="sell",
                                quantity=pos.quantity,
                                price=int(pos.current_price),
                                order_no=order.order_no,
                                strategy=config.strategy.name,
                                signal_reason="pool_exit",
                            )
                            remove_holding(pos.stock_code)
                            save_signal_log(
                                stock_code=pos.stock_code,
                                stock_name=pos.stock_name,
                                signal="SELL",
                                decision="확정",
                                reason="종목풀 퇴출 강제 매도",
                                signal_type="pool_exit",
                            )
                            held_codes.discard(pos.stock_code)
                            _sync_strategy_now(held_codes)
                    except Exception as e:
                        logger.error(f"풀 퇴출 매도 실패: {pos.stock_code} - {e}")
            signal_summary = {"BUY": 0, "SELL": 0, "HOLD": 0, "error": 0}
            for code in pool_codes:
                ohlcv = client.get_daily_ohlcv(code)
                if not ohlcv:
                    signal_summary["error"] += 1
                    continue

                df = pd.DataFrame(ohlcv)
                stock_name = _stock_names.get(code, "")
                try:
                    signal = strategy.generate_signal(code, df, stock_name=stock_name)
                except TypeError:
                    signal = strategy.generate_signal(code, df)
                if not signal.stock_name:
                    signal.stock_name = stock_name
                signal_summary[signal.signal.value] = signal_summary.get(signal.signal.value, 0) + 1

                if signal.is_actionable:
                    # LLM 검증: ML 시그널을 기술적 맥락에서 확인
                    confirmed, llm_reason = validator.validate_signal(
                        code, signal.stock_name, signal.signal.value,
                        signal.reason, df,
                    )
                    save_signal_log(
                        stock_code=code,
                        stock_name=signal.stock_name,
                        signal=signal.signal.value,
                        decision="확정" if confirmed else "보류",
                        reason=llm_reason,
                    )
                    if not confirmed:
                        logger.info(f"LLM 보류: {signal.stock_name} {signal.signal.value} — {llm_reason}")
                        continue

                    notifier.notify_signal(signal)

                    if signal.signal.value == "BUY":
                        # 이미 보유 중인 종목은 추가 매수하지 않음
                        if code in held_codes:
                            continue
                        qty = risk_mgr.calculate_position_size(
                            signal, balance.total_deposit,
                            balance.total_eval, len(balance.positions),
                            regime_label=_current_regime[0],
                        )
                        if qty > 0:
                            valid, reason = risk_mgr.validate_order(signal, qty, balance)
                            if valid:
                                order = order_mgr.buy(code, qty, reference_price=int(signal.price))
                                if order.order_no:
                                    notifier.notify_order_filled(order)
                                    save_trade(
                                        stock_code=code,
                                        stock_name=stock_name,
                                        side="buy",
                                        quantity=qty,
                                        price=int(signal.price),
                                        order_no=order.order_no,
                                        strategy=config.strategy.name,
                                        signal_strength=float(signal.strength),
                                        signal_reason=signal.reason or "",
                                    )
                                    # DB에 매수일 기록
                                    save_holding(
                                        stock_code=code,
                                        stock_name=stock_name,
                                        avg_price=signal.price,
                                        quantity=qty,
                                        buy_date=datetime.now().strftime("%Y%m%d"),
                                    )
                                    held_codes.add(code)
                                    _pos_prices[code] = float(signal.price)
                                    _pos_avg[code] = float(signal.price)
                                    _sync_strategy_now(held_codes)
                    else:
                        # 보유 중이면 전량 매도
                        for pos in balance.positions:
                            if pos.stock_code == code:
                                order = order_mgr.sell(code, pos.quantity, reference_price=int(pos.current_price))
                                if order.order_no:
                                    notifier.notify_order_filled(order)
                                    save_trade(
                                        stock_code=code,
                                        stock_name=pos.stock_name,
                                        side="sell",
                                        quantity=pos.quantity,
                                        price=int(pos.current_price),
                                        order_no=order.order_no,
                                        strategy=config.strategy.name,
                                        signal_strength=float(signal.strength),
                                        signal_reason=signal.reason or "",
                                    )
                                    # DB에서 매도 종목 제거
                                    remove_holding(code)
                                    held_codes.discard(code)
                                    _sync_strategy_now(held_codes)

            # 주문 체결 확인 → 체결된 주문은 확정 알림 재발송
            try:
                filled = order_mgr.check_filled()
                for f_order in filled:
                    notifier.notify_order_filled(f_order)
            except Exception as e:
                logger.warning(f"체결 확인 실패: {e}")

            summary_msg = f"BUY:{signal_summary['BUY']} SELL:{signal_summary['SELL']} HOLD:{signal_summary['HOLD']} 오류:{signal_summary['error']}"
            logger.info(f"시그널 체크 완료: {len(pool_codes)}종목 — {summary_msg}")
            save_signal_log(
                stock_code="",
                stock_name=f"{len(pool_codes)}종목 스캔",
                signal="SCAN",
                decision=f"B:{signal_summary['BUY']} S:{signal_summary['SELL']} H:{signal_summary['HOLD']}",
                reason=f"오류: {signal_summary['error']}" if signal_summary['error'] > 0 else "정상",
                signal_type="summary",
            )
            risk_mgr.reset_error_count()
            sync_runtime_status()

            # 일일 수익 스냅샷 (대시보드 Cumulative Return 차트용)
            save_daily_pnl(
                total_eval=balance.total_eval,
                total_deposit=balance.total_deposit,
                total_pnl=balance.total_pnl,
                total_pnl_pct=balance.total_pnl_pct,
                num_positions=len(balance.positions),
            )

        except Exception as e:
            logger.error(f"시그널 체크 오류: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e))

    def daily_report():
        """일일 리포트 + LLM 피드백"""
        try:
            balance = account_mgr.get_balance()
            notifier.notify_daily_report(balance)
            from src.db.models import save_daily_pnl, get_today_trades
            save_daily_pnl(
                balance.total_eval, balance.total_deposit,
                balance.total_pnl, balance.total_pnl_pct,
                len(balance.positions),
            )

            # LLM 일일 매매 피드백
            today_trades = get_today_trades()
            feedback = validator.generate_daily_feedback(
                trades=today_trades,
                positions=balance.positions,
                total_pnl=balance.total_pnl,
                total_pnl_pct=balance.total_pnl_pct,
                total_eval=balance.total_eval,
            )
            if feedback:
                notifier.notify_daily_feedback(feedback)
                logger.info(f"일일 피드백 전송 완료")

            # Notion 일일 리포트
            notion.publish_daily_report(
                balance=balance,
                trades=today_trades,
                feedback=feedback or "",
            )

            # Codex CLI 리뷰 (설정에서 활성화된 경우)
            if config.codex.enabled and config.codex.daily_review:
                _trigger_codex_daily_review(config.codex.model)

        except Exception as e:
            logger.error(f"리포트 오류: {e}")

    def weekly_retrain():
        """주간 모델 재학습 — 최신 데이터로 XGBoost를 재학습하고 성능이 개선되면 교체합니다."""
        from src.data.market_data import get_universe, get_ohlcv_batch
        from src.timing.retrain import retrain_model

        model_type = config.strategy.name.replace("factor_", "")
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        model_path = f"models/{model_type}_timing{ext}"

        logger.info(f"주간 모델 재학습 시작: {model_type}")
        try:
            # 최신 OHLCV 데이터 수집 (최근 2년)
            end = datetime.now().strftime("%Y%m%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            universe = get_universe()
            codes = universe["code"].tolist()[:100]
            ohlcv_dict = get_ohlcv_batch(codes, start, end)

            result = retrain_model(ohlcv_dict, model_type, model_path)

            if result.get("replaced"):
                # 모델이 교체되었으면 predictor 리로드
                if hasattr(strategy, 'predictor'):
                    strategy.predictor.model.load(model_path)
                    logger.info("라이브 전략에 새 모델 반영 완료")

                msg = (
                    f"모델 재학습 완료 — 교체됨\n"
                    f"• accuracy: {result['new_accuracy']:.3f}\n"
                    f"• F1: {result['new_f1']:.3f}\n"
                    f"• 학습: {result['train_samples']:,}건, 검증: {result['val_samples']:,}건\n"
                    f"• 시그널: BUY {result['signal_dist']['buy']}, "
                    f"SELL {result['signal_dist']['sell']}, "
                    f"HOLD {result['signal_dist']['hold']}"
                )
            else:
                msg = f"모델 재학습 완료 — 기존 모델 유지 (성능 개선 없음)"

            logger.info(msg)
            notifier._send(f"🔄 *{msg}*")

        except Exception as e:
            logger.error(f"모델 재학습 실패: {e}")
            risk_mgr.record_error()
            sync_runtime_status()
            notifier.notify_error(str(e), "주간 모델 재학습")

    # 스케줄 등록
    scheduler.add_job(
        check_signals, "interval",
        seconds=config.schedule.check_interval_sec,
        start_date=f"{datetime.now().strftime('%Y-%m-%d')} {config.schedule.market_open}:00",
    )
    scheduler.add_job(
        daily_report, "cron",
        hour=int(config.schedule.post_market.split(":")[0]),
        minute=int(config.schedule.post_market.split(":")[1]),
    )
    # 월간 리밸런싱 (매월 첫 거래일, 장 시작 전)
    pre_h, pre_m = config.schedule.pre_market.split(":")
    scheduler.add_job(
        rebalance_pool, "cron",
        day=config.schedule.rebalance_day,
        hour=int(pre_h), minute=int(pre_m),
    )

    # 매주 토요일 새벽 모델 재학습
    scheduler.add_job(
        weekly_retrain, "cron",
        day_of_week="sat", hour=6, minute=0,
    )

    # ── SAPPO 파이프라인: 일일 뉴스/sentiment + 주간 리포트 ──
    import subprocess

    def _run_sappo_script(script_args: list[str], tag: str, timeout: int) -> None:
        logger.info(f"[{tag}] 실행 시작: {' '.join(script_args)}")
        try:
            result = subprocess.run(
                [sys.executable, *script_args],
                check=False,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
            )
            logger.info(f"[{tag}] 완료 rc={result.returncode}")
            if result.stdout:
                logger.debug(f"[{tag} stdout] {result.stdout[-500:]}")
            if result.returncode != 0 and result.stderr:
                logger.warning(f"[{tag} stderr] {result.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            logger.error(f"[{tag}] {timeout}초 초과 — 강제 종료됨")
        except FileNotFoundError:
            logger.warning(f"[{tag}] 스크립트 없음 — 건너뜀")
        except Exception as e:
            logger.warning(f"[{tag}] 실행 실패: {e}")

    def daily_sappo_fetch():
        _run_sappo_script(["scripts/fetch_daily_news.py"], "SAPPO", timeout=900)

    def daily_market_news_fetch():
        if not config.regime.news_fetch_enabled:
            logger.debug("[REGIME] news_fetch_enabled=False — 시장 뉴스 수집 skip")
            return
        _run_sappo_script(["scripts/fetch_market_news.py"], "REGIME news", timeout=600)

    def detect_regime_daily():
        if not config.regime.enabled:
            logger.debug("[REGIME] enabled=False — detect skip")
            return
        try:
            from src.regime.detector import RegimeDetector
            from src.db.sappo_models import upsert_regime_label
            today = datetime.now().strftime("%Y%m%d")
            result = RegimeDetector().detect_today(today)
            upsert_regime_label(
                date=today,
                label=result.label,
                hmm_state=result.hmm_state,
                hmm_probs=result.hmm_probs_tuple,
                kospi_return_60d=result.kospi_return_60d,
                kospi_vol_60d=result.kospi_vol_60d,
                llm_score=result.llm_score,
                overridden_by_llm=result.overridden_by_llm,
                notes=result.notes,
            )
            prev_label = _current_regime[0]
            _current_regime[0] = result.label
            try:
                changed_msg = (
                    f" (이전: {prev_label})" if prev_label and prev_label != result.label else ""
                )
                lam_note = (
                    f" λ={config.regime.lambda_}" if config.regime.lambda_ > 0 else " (dark-launch λ=0)"
                )
                notifier.notify_info(
                    f"[REGIME] {today} → *{result.label}*{changed_msg}\n"
                    f"posts={ {k: round(v,2) for k,v in result.hmm_probs.items()} } "
                    f"ret60d={result.kospi_return_60d:+.3f} vol60d={result.kospi_vol_60d:.3f} "
                    f"llm={result.llm_score} override={result.overridden_by_llm}{lam_note}"
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[REGIME] detect_regime_daily 실패: {e}")
            try:
                notifier.notify_error(str(e), "Regime detect")
            except Exception:
                pass

    def weekly_sappo_report():
        _run_sappo_script(
            ["scripts/weekly_sappo_report.py", "--slack"], "SAPPO 주간", timeout=600
        )

    # 일일 SAPPO 수집 — post_market 이후 (16:00)
    scheduler.add_job(
        daily_sappo_fetch, "cron",
        hour=16, minute=0,
    )
    # ── Regime Detector — 매일 장 시작 전 (08:30 시장 뉴스, 08:45 detect) ──
    if config.regime.enabled:
        scheduler.add_job(
            daily_market_news_fetch, "cron",
            hour=8, minute=30, id="regime_news_fetch",
        )
        scheduler.add_job(
            detect_regime_daily, "cron",
            hour=8, minute=45, id="regime_detect",
        )
        logger.info("[REGIME] 스케줄 등록: 시장 뉴스 08:30, regime detect 08:45")
    # 주간 SAPPO 리포트 — 매주 토요일 오전 (weekly_retrain 직후 07:00)
    scheduler.add_job(
        weekly_sappo_report, "cron",
        day_of_week="sat", hour=7, minute=0,
    )

    scheduler.add_job(
        lambda: logger.info("[HEARTBEAT] scheduler alive"),
        "interval", minutes=5, id="heartbeat",
    )

    from apscheduler.events import (
        EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_MAX_INSTANCES,
    )

    _EVENT_META = {
        EVENT_JOB_ERROR: ("error", "잡 실행 실패", "scheduler job"),
        EVENT_JOB_MISSED: ("warning", "잡 실행 누락 (스케줄 시간 경과)", "scheduler miss"),
        EVENT_JOB_MAX_INSTANCES: ("warning", "잡 최대 동시 실행 초과 (이전 실행 미완료)", "scheduler max_instances"),
    }

    def _scheduler_event_listener(event):
        level, desc, ctx = _EVENT_META.get(event.code, ("warning", "알 수 없는 이벤트", "scheduler"))
        job_id = getattr(event, "job_id", "?")
        exc = getattr(event, "exception", None)
        msg = f"{desc}: {job_id}" + (f" — {exc}" if exc else "")
        getattr(logger, level)(f"[SCHEDULER] {msg}")
        notifier.notify_error(str(exc) if exc else msg, f"{ctx}: {job_id}")

    scheduler.add_listener(
        _scheduler_event_listener,
        EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES,
    )

    logger.info(f"스케줄 시작: {config.schedule.check_interval_sec}초 간격, 리밸런싱 매월 {config.schedule.rebalance_day}일, 재학습 매주 토요일, SAPPO 수집 매일 16:00, SAPPO 리포트 매주 토요일 07:00, heartbeat 5분")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 정상 종료")
    except Exception as e:
        logger.critical(f"스케줄러 비정상 종료: {e}")
        sync_runtime_status()
        notifier.notify_error(str(e), "스케줄러 크래시")
    finally:
        sync_runtime_status()
        scheduler.shutdown(wait=False)


def _create_live_strategy(config):
    """설정에 따라 전략을 생성합니다."""
    strategy_name = config.strategy.name

    if strategy_name == "factor_only":
        from src.strategy.factor_only import FactorOnlyStrategy
        return FactorOnlyStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_macd":
        from src.strategy.factor_macd import FactorMACDStrategy
        return FactorMACDStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_kdj":
        from src.strategy.factor_kdj import FactorKDJStrategy
        return FactorKDJStrategy(params=config.strategy.params.model_dump())
    elif strategy_name == "factor_rl":
        model_path = "models/rl_timing.pt"
        from src.strategy.factor_rl import FactorRLStrategy
        return FactorRLStrategy(model_path, config.strategy.params.model_dump())
    elif strategy_name == "factor_hybrid":
        from src.strategy.factor_hybrid import FactorHybridStrategy
        return FactorHybridStrategy(
            ml_model_path="models/xgboost_timing.pkl",
            rl_model_path="models/rl_timing.pt",
            params=config.strategy.params.model_dump(),
        )
    elif strategy_name.startswith("factor_"):
        model_type = strategy_name.replace("factor_", "")
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        model_path = f"models/{model_type}_timing{ext}"
        from src.strategy.factor_ml import FactorMLStrategy
        return FactorMLStrategy(model_type, model_path, config.strategy.params.model_dump())
    else:
        raise ValueError(f"알 수 없는 전략: {strategy_name}")


# ═══════════════════════════════════════════════
# 학습 모드
# ═══════════════════════════════════════════════
def run_train(model_type: str):
    """ML 타이밍 모델을 학습합니다."""
    config = get_config()
    logger.info(f"모델 학습: {model_type}")

    from src.data.market_data import get_universe, get_ohlcv_batch
    from src.timing.trainer import train_timing_model

    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")

    universe = get_universe(end)
    codes = universe["code"].tolist()[:100]

    ohlcv_dict = get_ohlcv_batch(codes, start, end)

    ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
    save_path = f"models/{model_type}_timing{ext}"

    result = train_timing_model(ohlcv_dict, model_type, save_path=save_path)
    logger.info(f"학습 결과: {result}")


# ═══════════════════════════════════════════════
# 재학습 모드
# ═══════════════════════════════════════════════
def run_retrain(model_type: str):
    """최신 데이터로 모델을 재학습하고 성능이 개선되면 교체합니다."""
    config = get_config()
    logger.info(f"모델 재학습: {model_type}")

    from src.data.market_data import get_universe, get_ohlcv_batch
    from src.timing.retrain import retrain_model

    # 최근 2년 데이터
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")

    universe = get_universe()
    codes = universe["code"].tolist()[:100]
    ohlcv_dict = get_ohlcv_batch(codes, start, end)

    ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
    model_path = f"models/{model_type}_timing{ext}"

    result = retrain_model(ohlcv_dict, model_type, model_path)

    if result.get("replaced"):
        print(f"\n  모델 교체 완료!")
        print(f"  accuracy: {result['new_accuracy']:.3f}")
        print(f"  F1: {result['new_f1']:.3f}")
        print(f"  학습: {result['train_samples']:,}건, 검증: {result['val_samples']:,}건")
        dist = result['signal_dist']
        print(f"  시그널 분포: BUY {dist['buy']}, SELL {dist['sell']}, HOLD {dist['hold']}")
    else:
        print(f"\n  기존 모델 유지 (성능 개선 없음)")
        if "new_accuracy" in result:
            print(f"  새 모델 accuracy: {result['new_accuracy']:.3f}, F1: {result['new_f1']:.3f}")
    print()


# ═══════════════════════════════════════════════
# 계좌 현황 조회
# ═══════════════════════════════════════════════
def run_status():
    """모의투자 계좌 현황을 간편하게 출력합니다."""
    config = get_config()

    from src.broker.kis_client import KISClient
    from src.broker.account import AccountManager

    print()
    env_label = "🏦 모의투자" if config.kis.is_virtual else "🏦 실전투자"
    print(f"  {env_label} 계좌 현황")
    print(f"  {'─' * 50}")

    try:
        client = KISClient()
        account_mgr = AccountManager(client)
        summary = account_mgr.get_balance()
    except Exception as e:
        print(f"  ❌ 계좌 조회 실패: {e}")
        return

    pnl_sign = "+" if summary.total_pnl >= 0 else ""
    pnl_color = "\033[91m" if summary.total_pnl < 0 else "\033[92m"
    reset = "\033[0m"

    print(f"  총 평가금액  : {summary.total_eval:>15,}원")
    print(f"  예수금       : {summary.total_deposit:>15,}원")
    print(f"  평가손익     : {pnl_color}{pnl_sign}{summary.total_pnl:>14,}원 ({pnl_sign}{summary.total_pnl_pct:.2f}%){reset}")
    print(f"  보유 종목    : {len(summary.positions)}개")
    print()

    if summary.positions:
        # 헤더
        print(f"  {'종목명':<12} {'수량':>6} {'평균가':>10} {'현재가':>10} {'손익':>12} {'수익률':>8}")
        print(f"  {'─' * 62}")

        for pos in summary.positions:
            ps = "+" if pos.pnl >= 0 else ""
            pc = "\033[91m" if pos.pnl < 0 else "\033[92m"
            print(
                f"  {pos.stock_name:<12} {pos.quantity:>5}주 "
                f"{pos.avg_price:>9,}  {pos.current_price:>9,}  "
                f"{pc}{ps}{pos.pnl:>10,}  {ps}{pos.pnl_pct:>6.1f}%{reset}"
            )

        print()
    else:
        print("  보유 종목 없음")
        print()


# ═══════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SuperTrader - Factor + ML Timing Trading System")
    parser.add_argument(
        "mode", choices=["backtest", "live", "train", "retrain", "status"],
        help="실행 모드: backtest(백테스트), live(실매매), train(모델학습), retrain(재학습), status(계좌현황)",
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["decision_tree", "xgboost", "lightgbm", "lstm", "transformer", "rl"],
        help="학습할 모델 타입 (train 모드용)",
    )
    parser.add_argument(
        "--config", type=str, default="config/settings.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--strategy", type=str, nargs="+", default=None,
        help="백테스트할 전략 (예: --strategy factor_rl)",
    )
    parser.add_argument(
        "--factor-module", type=str, default=None,
        choices=["alpha101", "alpha158", "both"],
        help="팩터 모듈 선택 (기본: config 설정 사용)",
    )
    parser.add_argument(
        "--review", action="store_true",
        help="완료 후 codex CLI 로 자동 리뷰 (backtest 모드)",
    )
    parser.add_argument(
        "--llm-filter", type=str, default=None,
        choices=["mock", "real"],
        help="백테스트에 LLM 검증 필터 적용 (mock=규칙기반, real=Claude API). 원신호 vs 필터 적용 비교용",
    )

    args = parser.parse_args()

    # 설정 로드
    load_config(args.config)
    if args.factor_module:
        get_config().factors.factor_module = args.factor_module
    setup_logging()

    if args.mode == "backtest":
        run_backtest(only_strategies=args.strategy, review=args.review, llm_filter=args.llm_filter)
    elif args.mode == "live":
        run_live()
    elif args.mode == "train":
        run_train(args.model)
    elif args.mode == "retrain":
        run_retrain(args.model)
    elif args.mode == "status":
        run_status()


if __name__ == "__main__":
    main()
