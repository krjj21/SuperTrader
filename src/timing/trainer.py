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
from src.timing.labels import generate_labels, generate_forward_returns


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

    # RL 모델은 별도 학습 파이프라인 사용 (+ SAPPO 옵션)
    if model_type == "rl":
        from src.timing.rl_trainer import train_rl_model
        rl_cfg = get_config().timing.rl
        sentiment_lambda = float(getattr(rl_cfg, "sentiment_lambda", 0.0))
        sentiment_source = str(getattr(rl_cfg, "sentiment_source", "off"))
        sentiment_map: dict[str, dict[str, float]] | None = None
        if sentiment_lambda > 0.0 and sentiment_source != "off":
            sentiment_map = _load_sentiment_map(ohlcv_dict.keys(), sentiment_source)
            if not sentiment_map:
                logger.warning(
                    f"sentiment_source={sentiment_source} 요청이지만 sentiment 데이터 없음 — "
                    f"baseline PPO (λ=0) 로 폴백"
                )
                sentiment_lambda = 0.0
                sentiment_source = "off"
        return train_rl_model(
            ohlcv_dict,
            save_path=save_path or "models/rl_timing.pt",
            sentiment_lambda=sentiment_lambda,
            sentiment_map=sentiment_map,
            sentiment_source=sentiment_source,
        )

    # ── 비-RL 모델 (DT/XGBoost/LightGBM/LSTM/Transformer) ──
    # 전 종목 피처 + 라벨 통합 (Transformer 는 forward_returns 도 함께)
    all_features = []
    all_labels = []
    all_forward_returns = []

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
        # Momentum Transformer cost-aware loss 입력용 raw 미래 수익률
        forward_returns = generate_forward_returns(
            df["close"], forward_days=config.forward_days,
        )

        features["_code"] = code
        all_features.append(features)
        all_labels.append(labels)
        all_forward_returns.append(forward_returns)

    if not all_features:
        return {"error": "no_data"}

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    fr = pd.concat(all_forward_returns, ignore_index=True)

    # _code 열 제거
    X = X.drop(columns=["_code"], errors="ignore")

    # Train/Val 분할
    X_val, y_val = None, None
    fr_train = fr
    if train_end_date:
        # 시간 기반 분할은 개별 종목 내에서 해야 하지만,
        # 여기서는 간단히 전체의 80%를 학습, 20%를 검증으로 사용
        split_idx = int(len(X) * 0.8)
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]
        X = X.iloc[:split_idx]
        y = y.iloc[:split_idx]
        fr_train = fr.iloc[:split_idx]

    # 모델 생성 및 학습
    model = create_model(model_type)
    # DT는 val 파라미터 없음, Transformer 는 forward_returns 추가 전달
    if model_type == "decision_tree":
        result = model.train(X, y)
    elif model_type == "transformer":
        # Momentum Transformer cost-aware loss 활성화
        result = model.train(X, y, X_val, y_val, forward_returns=fr_train)
    else:
        result = model.train(X, y, X_val, y_val)

    # 저장
    if save_path and "error" not in result:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)

    return result


def _load_sentiment_map(codes, source: str) -> dict[str, dict[str, float]]:
    """sentiment DB 에서 {code: {date: score}} 로드."""
    try:
        from src.db.sappo_models import init_sappo_db, get_sappo_session, SentimentScore
        init_sappo_db("data/trading.db")
    except Exception as e:
        logger.warning(f"SAPPO DB 초기화 실패: {e}")
        return {}

    session = get_sappo_session()
    try:
        rows = (
            session.query(SentimentScore)
            .filter(SentimentScore.stock_code.in_(list(codes)))
            .all()
        )
    finally:
        session.close()

    if not rows:
        return {}
    m: dict[str, dict[str, float]] = {}
    for r in rows:
        m.setdefault(r.stock_code, {})[r.date] = float(r.score)
    logger.info(f"sentiment_map 로드: {len(m)} 종목, 총 {sum(len(v) for v in m.values())} 점수")
    return m
