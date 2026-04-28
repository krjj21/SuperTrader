"""학습 / 재학습 모드 — main.py 에서 분리됨 (2026-04-28).

  · run_train(model_type)  — 새 모델 학습 (in-sample backtest 윈도)
  · run_retrain(model_type) — 최근 3년 데이터로 후보 학습 → F1/Sharpe 게이트 통과 시 교체
"""
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger

from src.config import get_config


def run_train(model_type: str) -> None:
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


def run_retrain(model_type: str) -> None:
    """최신 데이터로 모델을 재학습하고 성능이 개선되면 교체합니다.

    학습 윈도: 최근 3년 (1095일). 5년(1825) 시도했으나 batch 88K 가 GPU 6GB
    한계 초과 OOM (2026-04-28 사고). 3년이 mid-cap 학습 데이터 안정 균형.
    """
    config = get_config()
    logger.info(f"모델 재학습: {model_type}")

    from src.data.market_data import get_universe, get_ohlcv_batch
    from src.timing.retrain import retrain_model

    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=1095)).strftime("%Y%m%d")

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
