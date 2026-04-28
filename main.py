"""SuperTrader 메인 진입점 (CLI dispatcher).

5가지 모드를 src/runtime/* 의 분리된 모듈로 위임한다:
  · backtest  → src/runtime/backtest.run_backtest
  · live      → src/runtime/live.run_live
  · train     → src/runtime/training.run_train
  · retrain   → src/runtime/training.run_retrain
  · status    → src/runtime/status.run_status
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.config import load_config, get_config


def setup_logging() -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SuperTrader - Factor + ML Timing Trading System",
    )
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
        from src.runtime.backtest import run_backtest
        run_backtest(
            only_strategies=args.strategy,
            review=args.review,
            llm_filter=args.llm_filter,
        )
    elif args.mode == "live":
        from src.runtime.live import run_live
        run_live()
    elif args.mode == "train":
        from src.runtime.training import run_train
        run_train(args.model)
    elif args.mode == "retrain":
        from src.runtime.training import run_retrain
        run_retrain(args.model)
    elif args.mode == "status":
        from src.runtime.status import run_status
        run_status()


if __name__ == "__main__":
    main()
