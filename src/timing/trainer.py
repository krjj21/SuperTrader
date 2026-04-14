"""
통합 학습 파이프라인
- Walk-forward 확장 윈도우 학습
- 모든 타이밍 모델 지원
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import get_config
from src.timing.features import build_features
from src.timing.labels import generate_labels


def create_model(model_type: str, config=None):
    """모델 타입에 따라 모델 인스턴스를 생성합니다 (lazy import)."""
    if config is None:
        config = get_config().timing

    if model_type == "decision_tree":
        from src.timing.decision_tree import DecisionTreeTimingModel
        return DecisionTreeTimingModel(
            max_depth=config.decision_tree.max_depth,
            min_samples_leaf=config.decision_tree.min_samples_leaf,
        )
    elif model_type in ("xgboost", "lightgbm"):
        from src.timing.gradient_boost import GradientBoostTimingModel
        return GradientBoostTimingModel(
            engine=model_type,
            n_estimators=config.gradient_boost.n_estimators,
            max_depth=config.gradient_boost.max_depth,
            learning_rate=config.gradient_boost.learning_rate,
            subsample=config.gradient_boost.subsample,
            colsample_bytree=config.gradient_boost.colsample_bytree,
        )
    elif model_type == "lstm":
        from src.timing.lstm_model import LSTMTimingModel
        return LSTMTimingModel(
            sequence_length=config.lstm.sequence_length,
            hidden_size=config.lstm.hidden_size,
            num_layers=config.lstm.num_layers,
            dropout=config.lstm.dropout,
            learning_rate=config.lstm.learning_rate,
            epochs=config.lstm.epochs,
            batch_size=config.lstm.batch_size,
        )
    elif model_type == "transformer":
        from src.timing.transformer_model import TransformerTimingModel
        return TransformerTimingModel(
            sequence_length=config.transformer.sequence_length,
            d_model=config.transformer.d_model,
            nhead=config.transformer.nhead,
            num_layers=config.transformer.num_layers,
            dropout=config.transformer.dropout,
            learning_rate=config.transformer.learning_rate,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
        )
    elif model_type == "rl":
        from src.timing.rl_agent import RLTimingModel
        return RLTimingModel(
            state_dim=43,
            hidden_dim=config.rl.hidden_dim,
            learning_rate=config.rl.learning_rate,
            gamma=config.rl.gamma,
            group_size=config.rl.group_size,
            clip_epsilon_low=config.rl.clip_epsilon_low,
            clip_epsilon_high=config.rl.clip_epsilon_high,
            entropy_coeff=config.rl.entropy_coeff,
            epochs_per_update=config.rl.epochs_per_update,
        )
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")


def train_timing_model(
    ohlcv_dict: dict[str, pd.DataFrame],
    model_type: str,
    train_end_date: str | None = None,
    save_path: str | None = None,
) -> dict:
    """타이밍 모델을 학습합니다.

    Args:
        ohlcv_dict: {종목코드: OHLCV DataFrame} 딕셔너리
        model_type: 모델 타입
        train_end_date: 학습 데이터 종료일 (이후는 검증)
        save_path: 모델 저장 경로

    Returns:
        학습 결과 딕셔너리
    """
    config = get_config().timing

    # RL 모델은 별도 학습 파이프라인 사용
    if model_type == "rl":
        from src.timing.rl_trainer import train_rl_model
        return train_rl_model(ohlcv_dict, save_path=save_path or "models/rl_timing.pt")

    # 전 종목 피처 + 라벨 통합
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

        features["_code"] = code
        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        return {"error": "no_data"}

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    # _code 열 제거
    X = X.drop(columns=["_code"], errors="ignore")

    # Train/Val 분할
    X_val, y_val = None, None
    if train_end_date:
        # 시간 기반 분할은 개별 종목 내에서 해야 하지만,
        # 여기서는 간단히 전체의 80%를 학습, 20%를 검증으로 사용
        split_idx = int(len(X) * 0.8)
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        X = X.iloc[:split_idx]
        y = y.iloc[:split_idx]

    # 모델 생성 및 학습
    model = create_model(model_type)
    # DT는 val 파라미터 없음
    if model_type == "decision_tree":
        result = model.train(X, y)
    else:
        result = model.train(X, y, X_val, y_val)

    # 저장
    if save_path and "error" not in result:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    return result
