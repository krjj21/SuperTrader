"""
Gradient Boosting 타이밍 모델
- XGBoost / LightGBM
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from loguru import logger


class GradientBoostTimingModel:
    """XGBoost/LightGBM 기반 매매 타이밍 모델"""

    def __init__(
        self,
        engine: str = "xgboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
    ):
        self.engine = engine
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = None
        self.feature_names: list[str] = []

    def _create_model(self, n_classes: int):
        if self.engine == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1,
            )
        else:  # lightgbm
            import lightgbm as lgb
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="multiclass" if n_classes > 2 else "binary",
                num_class=n_classes if n_classes > 2 else None,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict:
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 100:
            return {"error": "insufficient_data"}

        self.feature_names = X_clean.columns.tolist()

        # 라벨을 0-indexed로 변환 (-1, 0, 1) → (0, 1, 2)
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = y_clean.map(label_map)
        n_classes = y_mapped.nunique()

        self.model = self._create_model(n_classes)
        self._label_map = label_map
        self._inv_label_map = {v: k for k, v in label_map.items()}

        fit_params = {}
        if X_val is not None and y_val is not None:
            val_mask = X_val.notna().all(axis=1) & y_val.notna()
            if val_mask.any():
                if self.engine == "xgboost":
                    fit_params["eval_set"] = [(X_val[val_mask][self.feature_names].values, y_val[val_mask].map(label_map).values)]
                    fit_params["verbose"] = False

        self.model.fit(X_clean.values, y_mapped.values, **fit_params)

        train_pred = self.model.predict(X_clean.values)
        accuracy = accuracy_score(y_mapped.values, train_pred)

        # 피처 중요도
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        logger.info(f"{self.engine} 학습 완료: accuracy={accuracy:.3f}, samples={len(X_clean)}")
        return {"accuracy": accuracy, "n_samples": len(X_clean), "top_features": top_features}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X_aligned = X[self.feature_names].copy()
        mask = X_aligned.notna().all(axis=1)
        predictions = pd.Series(0, index=X.index, dtype=int)

        if mask.any():
            pred = self.model.predict(X_aligned[mask].values)
            predictions[mask] = pd.Series(pred, index=X_aligned[mask].index).map(self._inv_label_map)

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X_aligned = X[self.feature_names].copy()
        mask = X_aligned.notna().all(axis=1)
        proba = pd.DataFrame(0.0, index=X.index, columns=[-1, 0, 1])

        if mask.any():
            p = self.model.predict_proba(X_aligned[mask].values)
            for i, cls in enumerate(sorted(self._inv_label_map.values())):
                proba.loc[mask, cls] = p[:, i]

        return proba

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model, "feature_names": self.feature_names,
                "engine": self.engine, "label_map": self._label_map,
                "inv_label_map": self._inv_label_map,
            }, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.engine = data["engine"]
        self._label_map = data["label_map"]
        self._inv_label_map = data["inv_label_map"]
