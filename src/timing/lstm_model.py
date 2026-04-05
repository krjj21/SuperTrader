"""
LSTM 타이밍 모델 (PyTorch)
- 시퀀스 입력 (20일 윈도우)
- 3클래스 분류 (BUY, HOLD, SELL)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class TimeSeriesDataset(Dataset):
    """시계열 시퀀스 데이터셋"""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        y_label = self.y[idx + self.seq_length - 1]
        return x_seq, y_label


class LSTMClassifier(nn.Module):
    """LSTM 분류기"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)


class LSTMTimingModel:
    """LSTM 기반 매매 타이밍 모델"""

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: LSTMClassifier | None = None
        self.feature_names: list[str] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = np.nanmean(X, axis=0)
            self._std = np.nanstd(X, axis=0)
            self._std[self._std < 1e-10] = 1.0
        return (X - self._mean) / self._std

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict:
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask].values.astype(np.float32)
        y_clean = y[mask].values

        if len(X_clean) < self.sequence_length + 50:
            return {"error": "insufficient_data"}

        self.feature_names = X.columns.tolist()

        # 라벨 매핑
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_map[v] for v in y_clean])

        # 정규화
        X_norm = self._normalize(X_clean, fit=True)
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        # 데이터셋
        dataset = TimeSeriesDataset(X_norm, y_mapped, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 모델
        self.model = LSTMClassifier(
            input_size=X_clean.shape[1],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # 클래스 가중치
        class_counts = np.bincount(y_mapped, minlength=3).astype(float) + 1
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5,
        )

        best_loss = float("inf")
        patience_count = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 15:
                    logger.info(f"LSTM Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, loss={avg_loss:.4f}")

        logger.info(f"LSTM 학습 완료: best_loss={best_loss:.4f}, samples={len(X_clean)}")
        return {"best_loss": best_loss, "n_samples": len(X_clean)}

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")

        X_aligned = X[self.feature_names].copy()
        X_vals = X_aligned.values.astype(np.float32)
        X_norm = self._normalize(X_vals)
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        inv_label_map = {0: -1, 1: 0, 2: 1}
        predictions = pd.Series(0, index=X.index, dtype=int)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.sequence_length - 1, len(X_norm)):
                seq = torch.FloatTensor(
                    X_norm[i - self.sequence_length + 1:i + 1]
                ).unsqueeze(0).to(self.device)
                output = self.model(seq)
                pred_class = output.argmax(dim=1).item()
                predictions.iloc[i] = inv_label_map[pred_class]

        return predictions

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "feature_names": self.feature_names,
            "mean": self._mean, "std": self._std,
            "config": {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "input_size": len(self.feature_names),
            },
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self._mean = data["mean"]
        self._std = data["std"]
        cfg = data["config"]
        self.model = LSTMClassifier(
            input_size=cfg["input_size"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(data["model_state"])
