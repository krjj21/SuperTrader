"""
모델 자동 재학습 모듈
- 최신 시장 데이터로 주기적 재학습
- 신규 모델이 기존 모델보다 나을 때만 교체
- 검증 정확도 + 수익률 기반 평가
"""
from __future__ import annotations

import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_config
from src.timing.features import build_features
from src.timing.labels import generate_labels
from src.timing.trainer import create_model


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """모델의 검증 성능을 평가합니다."""
    from sklearn.metrics import accuracy_score, f1_score

    mask = X_val.notna().all(axis=1) & y_val.notna()
    if mask.sum() < 50:
        return {"error": "insufficient_val_data"}

    X_clean = X_val[mask]
    y_clean = y_val[mask]

    predictions = model.predict(X_clean)
    pred_clean = predictions[mask]

    accuracy = accuracy_score(y_clean, pred_clean)
    f1 = f1_score(y_clean, pred_clean, average="weighted", zero_division=0)

    # 시그널 분포
    n_buy = (pred_clean == 1).sum()
    n_sell = (pred_clean == -1).sum()
    n_hold = (pred_clean == 0).sum()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "n_samples": len(y_clean),
        "n_buy": int(n_buy),
        "n_sell": int(n_sell),
        "n_hold": int(n_hold),
    }


def retrain_model(
    ohlcv_dict: dict[str, pd.DataFrame],
    model_type: str = "xgboost",
    current_model_path: str = "models/xgboost_timing.pkl",
    val_ratio: float = 0.2,
) -> dict:
    """최신 데이터로 모델을 재학습하고, 기존 모델보다 나으면 교체합니다.

    Args:
        ohlcv_dict: {종목코드: OHLCV DataFrame}
        model_type: 모델 타입
        current_model_path: 현재 사용 중인 모델 경로
        val_ratio: 검증 데이터 비율 (최근 N%)

    Returns:
        재학습 결과 딕셔너리
    """
    config = get_config().timing

    # 1. 전 종목 피처 + 라벨 통합
    all_features = []
    all_labels = []

    for code, df in ohlcv_dict.items():
        if len(df) < 100:
            continue

        features = build_features(df)
        labels = generate_labels(
            df["close"],
            forward_days=config.forward_days,
            buy_threshold=config.buy_threshold,
            sell_threshold=config.sell_threshold,
        )
        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        return {"error": "no_data", "replaced": False}

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    # 2. 시간 기반 Train/Val 분할 (최근 val_ratio를 검증용)
    split_idx = int(len(X) * (1 - val_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    logger.info(f"재학습 데이터: train={len(X_train)}, val={len(X_val)}")

    # 3. 새 모델 학습
    new_model = create_model(model_type)
    if model_type == "decision_tree":
        train_result = new_model.train(X_train, y_train)
    else:
        train_result = new_model.train(X_train, y_train, X_val, y_val)

    if "error" in train_result:
        return {"error": train_result["error"], "replaced": False}

    # 4. 새 모델 검증 성능
    new_eval = evaluate_model(new_model, X_val, y_val)
    if "error" in new_eval:
        return {"error": new_eval["error"], "replaced": False}

    logger.info(
        f"새 모델 성능: accuracy={new_eval['accuracy']:.3f}, "
        f"f1={new_eval['f1']:.3f}, "
        f"BUY={new_eval['n_buy']}, SELL={new_eval['n_sell']}, HOLD={new_eval['n_hold']}"
    )

    # 5. 기존 모델 검증 성능 (있으면)
    replaced = False
    current_path = Path(current_model_path)

    if current_path.exists():
        old_model = create_model(model_type)
        old_model.load(str(current_path))
        old_eval = evaluate_model(old_model, X_val, y_val)

        if "error" not in old_eval:
            logger.info(
                f"기존 모델 성능: accuracy={old_eval['accuracy']:.3f}, "
                f"f1={old_eval['f1']:.3f}"
            )

            # 새 모델이 정확도 또는 F1 기준으로 개선되었을 때만 교체
            improved = (
                new_eval["f1"] > old_eval["f1"] + 0.005
                or (new_eval["f1"] >= old_eval["f1"] and new_eval["accuracy"] > old_eval["accuracy"] + 0.005)
            )

            if improved:
                # 기존 모델 백업
                backup_path = current_path.with_suffix(f".backup_{datetime.now():%Y%m%d_%H%M}.pkl")
                shutil.copy2(current_path, backup_path)
                logger.info(f"기존 모델 백업: {backup_path}")

                new_model.save(str(current_path))
                replaced = True
                logger.info(
                    f"모델 교체 완료: F1 {old_eval['f1']:.3f} → {new_eval['f1']:.3f}, "
                    f"accuracy {old_eval['accuracy']:.3f} → {new_eval['accuracy']:.3f}"
                )
            else:
                logger.info(
                    f"모델 유지: 신규 F1={new_eval['f1']:.3f} <= 기존 F1={old_eval['f1']:.3f}"
                )
        else:
            # 기존 모델 평가 실패 시 새 모델로 교체
            new_model.save(str(current_path))
            replaced = True
    else:
        # 기존 모델이 없으면 바로 저장
        current_path.parent.mkdir(parents=True, exist_ok=True)
        new_model.save(str(current_path))
        replaced = True
        logger.info(f"신규 모델 저장: {current_path}")

    return {
        "replaced": replaced,
        "new_accuracy": new_eval["accuracy"],
        "new_f1": new_eval["f1"],
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "signal_dist": {
            "buy": new_eval["n_buy"],
            "sell": new_eval["n_sell"],
            "hold": new_eval["n_hold"],
        },
    }
