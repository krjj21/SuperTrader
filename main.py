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
    models_to_train = ["decision_tree", "xgboost"]  # 기본적으로 DT, XGB 학습

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

            # 종목풀 내 종목에 대해 시그널 체크
            for code in strategy._pool if hasattr(strategy, '_pool') else []:
                ohlcv = client.get_daily_ohlcv(code)
                if not ohlcv:
                    continue

                df = pd.DataFrame(ohlcv)
                signal = strategy.generate_signal(code, df)

                if signal.is_actionable:
                    notifier.notify_signal(signal)

                    if signal.signal.value == "BUY":
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
        """일일 리포트"""
        try:
            balance = account_mgr.get_balance()
            notifier.notify_daily_report(balance)
            from src.db.models import save_daily_pnl
            save_daily_pnl(
                balance.total_eval, balance.total_deposit,
                balance.total_pnl, balance.total_pnl_pct,
                len(balance.positions),
            )
        except Exception as e:
            logger.error(f"리포트 오류: {e}")

    # 스케줄 등록
    scheduler.add_job(
        check_signals, "interval",
        seconds=config.schedule.check_interval_sec,
        start_date=f"{datetime.now().strftime('%Y-%m-%d')} {config.schedule.market_open}",
    )
    scheduler.add_job(
        daily_report, "cron",
        hour=int(config.schedule.post_market.split(":")[0]),
        minute=int(config.schedule.post_market.split(":")[1]),
    )

    logger.info(f"스케줄 시작: {config.schedule.check_interval_sec}초 간격")
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
# CLI
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SuperTrader - Factor + ML Timing Trading System")
    parser.add_argument(
        "mode", choices=["backtest", "live", "train"],
        help="실행 모드: backtest(백테스트), live(실매매), train(모델학습)",
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["decision_tree", "xgboost", "lightgbm", "lstm", "transformer"],
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


if __name__ == "__main__":
    main()
