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
def run_backtest():
    """팩터 + 타이밍 전략 비교 백테스트를 실행합니다."""
    config = get_config()
    logger.info("=" * 60)
    logger.info("SuperTrader 백테스트 시작")
    logger.info("=" * 60)

    from src.data.market_data import get_universe, get_ohlcv_batch
    from src.factors.alpha101 import compute_all_factors
    from backtest.comparison import run_strategy_comparison
    from backtest.report import plot_equity_comparison, print_comparison_table

    start = config.backtest.start_date.replace("-", "")
    end = config.backtest.end_date.replace("-", "")

    # 1. 유니버스 구성
    logger.info("Step 1: 유니버스 구성")
    universe = get_universe()
    if universe.empty:
        logger.error("유니버스가 비어 있습니다")
        return
    codes = universe["code"].tolist()
    logger.info(f"유니버스: {len(codes)}종목")

    # 2. OHLCV 데이터 로드 (상위 50종목으로 빠른 테스트)
    logger.info("Step 2: OHLCV 데이터 로드")
    n_stocks = min(50, len(codes))
    ohlcv_dict = get_ohlcv_batch(codes[:n_stocks], start, end)
    logger.info(f"OHLCV 로드: {len(ohlcv_dict)}종목")

    # 3. 리밸런싱 날짜 생성 (월초)
    logger.info("Step 3: 리밸런싱 스케줄 생성")
    rebalance_dates = _generate_rebalance_dates(start, end, config.factors.rebalance_freq)
    logger.info(f"리밸런싱 날짜: {len(rebalance_dates)}회")

    # 4. 종목풀 히스토리 구축 (OHLCV 기반 모멘텀 팩터)
    logger.info("Step 4: 종목풀 히스토리 구축")
    pool_history = _build_pool_history_from_ohlcv(
        ohlcv_dict, rebalance_dates, config.factors.top_n,
    )

    # 5. ML 모델 학습 (있으면)
    logger.info("Step 5: ML 모델 학습")
    model_paths = _train_models_if_needed(ohlcv_dict)

    # 6. 전략 비교 실행
    logger.info("Step 6: 전략 비교 백테스트 실행")
    comparison = run_strategy_comparison(
        ohlcv_dict, pool_history, rebalance_dates, model_paths,
    )

    if not comparison.empty:
        print_comparison_table(comparison)
        logger.info("백테스트 완료!")
    else:
        logger.warning("백테스트 결과 없음")


def _build_pool_history_from_ohlcv(
    ohlcv_dict: dict[str, pd.DataFrame],
    rebalance_dates: list[str],
    top_n: int = 30,
) -> dict[str, list[str]]:
    """OHLCV 데이터에서 모멘텀 기반으로 종목풀을 구성합니다."""
    pool_history = {}
    available_codes = list(ohlcv_dict.keys())

    for date in rebalance_dates:
        dt = pd.Timestamp(date)
        scores = {}

        for code, df in ohlcv_dict.items():
            # 기준일까지의 데이터
            mask = df["date"] <= dt
            df_until = df[mask]
            if len(df_until) < 60:
                continue

            close = df_until["close"]
            # 모멘텀 점수: 6개월 수익률 - 1개월 수익률
            if len(close) >= 120:
                ret_6m = close.iloc[-1] / close.iloc[-120] - 1
                ret_1m = close.iloc[-1] / close.iloc[-20] - 1
                scores[code] = ret_6m - ret_1m
            elif len(close) >= 60:
                ret_3m = close.iloc[-1] / close.iloc[-60] - 1
                scores[code] = ret_3m

        if scores:
            sorted_codes = sorted(scores, key=lambda x: scores[x], reverse=True)
            pool_codes = sorted_codes[:min(top_n, len(sorted_codes))]
        else:
            pool_codes = available_codes[:top_n]

        pool_history[date] = pool_codes
        logger.info(f"  {date}: {len(pool_codes)}종목 선정")

    return pool_history


def _generate_rebalance_dates(start: str, end: str, freq: str) -> list[str]:
    """리밸런싱 날짜를 생성합니다."""
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt = datetime.strptime(end, "%Y%m%d")

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


def _train_models_if_needed(ohlcv_dict: dict) -> dict[str, str]:
    """ML 모델이 없으면 학습합니다."""
    config = get_config()
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_paths = {}
    models_to_train = ["decision_tree", "xgboost", "rl"]  # DT, XGB, RL 학습

    for model_type in models_to_train:
        ext = ".pkl" if model_type in ("decision_tree", "xgboost", "lightgbm") else ".pt"
        path = str(model_dir / f"{model_type}_timing{ext}")

        if Path(path).exists():
            model_paths[model_type] = path
            logger.info(f"기존 모델 사용: {model_type} ({path})")
            continue

        logger.info(f"모델 학습 시작: {model_type}")
        from src.timing.trainer import train_timing_model
        result = train_timing_model(ohlcv_dict, model_type, save_path=path)

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
    from src.db.models import init_db

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

    def rebalance_pool():
        """팩터 기반 종목풀을 리밸런싱합니다."""
        today = datetime.now().strftime("%Y%m%d")
        try:
            pool = build_stock_pool(today, previous_pool=_current_pool[0])
            if pool.codes:
                strategy.update_pool(pool.codes)
                _current_pool[0] = pool
                logger.info(f"종목풀 업데이트: {len(pool.codes)}종목 (신규 {len(pool.entered)}, 퇴출 {len(pool.exited)})")
                notifier.notify_start()  # 종목풀 갱신 알림
            else:
                logger.warning("종목풀이 비어 있습니다 — 기존 풀 유지")
        except Exception as e:
            logger.error(f"종목풀 구성 실패: {e}")
            notifier.notify_error(str(e), "종목풀 리밸런싱")

    # 시작 시 즉시 종목풀 구성
    logger.info("팩터 기반 종목풀 구성 중...")
    rebalance_pool()

    # LLM 시그널 검증기
    from src.timing.llm_validator import SignalValidator
    validator = SignalValidator()
    if validator.is_enabled:
        logger.info("LLM 시그널 검증 활성화 (Claude Haiku)")
    else:
        logger.info("LLM 시그널 검증 비활성화 — ANTHROPIC_API_KEY 미설정")

    notifier.notify_start()

    # 스케줄러
    scheduler = BlockingScheduler()

    def check_signals():
        """주기적 시그널 체크"""
        if not risk_mgr.is_trading_allowed:
            return

        try:
            balance = account_mgr.get_balance()
            risk_mgr.check_daily_loss_limit(balance.total_pnl, balance.total_eval)

            # 보유 종목 코드 set
            held_codes = {pos.stock_code for pos in balance.positions}

            # 종목풀 내 종목에 대해 시그널 체크
            for code in strategy._pool if hasattr(strategy, '_pool') else []:
                ohlcv = client.get_daily_ohlcv(code)
                if not ohlcv:
                    continue

                df = pd.DataFrame(ohlcv)
                signal = strategy.generate_signal(code, df)

                if signal.is_actionable:
                    # LLM 검증: ML 시그널을 기술적 맥락에서 확인
                    confirmed, llm_reason = validator.validate_signal(
                        code, signal.stock_name, signal.signal.value,
                        signal.reason, df,
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
                        )
                        if qty > 0:
                            valid, reason = risk_mgr.validate_order(signal, qty, balance)
                            if valid:
                                order = order_mgr.buy(code, qty)
                                if order.order_no:
                                    notifier.notify_order_filled(order)
                    else:
                        # 보유 중이면 전량 매도
                        for pos in balance.positions:
                            if pos.stock_code == code:
                                order = order_mgr.sell(code, pos.quantity)
                                if order.order_no:
                                    notifier.notify_order_filled(order)

            risk_mgr.reset_error_count()

        except Exception as e:
            logger.error(f"시그널 체크 오류: {e}")
            risk_mgr.record_error()
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
    # 매월 리밸런싱 (장 시작 전)
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

    logger.info(f"스케줄 시작: {config.schedule.check_interval_sec}초 간격, 리밸런싱 매월 {config.schedule.rebalance_day}일, 재학습 매주 토요일")
    scheduler.start()


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

    args = parser.parse_args()

    # 설정 로드
    load_config(args.config)
    setup_logging()

    if args.mode == "backtest":
        run_backtest()
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
