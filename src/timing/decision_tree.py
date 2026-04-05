"""
Decision Tree 타이밍 모델
- sklearn DecisionTreeClassifier
- Walk-forward 검증
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from loguru import logger


class DecisionTreeTimingModel:
    """Decision Tree 기반 매매 타이밍 모델"""

    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 20):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.model: DecisionTreeClassifier | None = None
        self.feature_names: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """모델을 학습합니다.

        Args:
            X: 피처 DataFrame
            y: 라벨 Series (1, 0, -1)

        Returns:
            학습 결과 딕셔너리
        """
        # NaN 제거
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            logger.warning(f"학습 데이터 부족: {len(X_clean)}")
            return {"error": "insufficient_data"}

        self.feature_names = X_clean.columns.tolist()
        self.model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=42,
        )
        self.model.fit(X_clean.values, y_clean.values)

        # 학습 성과
        train_pred = self.model.predict(X_clean.values)
        accuracy = accuracy_score(y_clean.values, train_pred)

        # 피처 중요도
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"DT 학습 완료: accuracy={accuracy:.3f}, samples={len(X_clean)}")
        return {
            "accuracy": accuracy,
            "n_samples": len(X_clean),
            "top_features": top_features,
        }

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """예측을 수행합니다.

        Returns:
            예측 라벨 Series (1, 0, -1)
        """
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X_aligned = X[self.feature_names].copy()
        mask = X_aligned.notna().all(axis=1)

        predictions = pd.Series(0, index=X.index, dtype=int)
        if mask.any():
            pred = self.model.predict(X_aligned[mask].values)
            predictions[mask] = pred

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """클래스별 확률을 반환합니다."""
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X_aligned = X[self.feature_names].copy()
        mask = X_aligned.notna().all(axis=1)

        proba = pd.DataFrame(0.0, index=X.index, columns=self.model.classes_)
        if mask.any():
            p = self.model.predict_proba(X_aligned[mask].values)
            proba.loc[mask] = p

        return proba

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "feature_names": self.feature_names}, f)
        logger.info(f"DT 모델 저장: {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info(f"DT 모델 로드: {path}")
