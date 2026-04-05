"""
Transformer 타이밍 모델 (PyTorch)
- Multi-head attention으로 팩터 간 상호작용 포착
- 시퀀스 입력 (30일 윈도우)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

from src.timing.lstm_model import TimeSeriesDataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # 마지막 시점
        x = self.dropout(x)
        return self.fc(x)


class TransformerTimingModel:
    """Transformer 기반 매매 타이밍 모델"""

    def __init__(
        self,
        sequence_length: int = 30,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.0005,
        epochs: int = 100,
        batch_size: int = 64,
    ):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model: TransformerClassifier | None = None
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
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_map[v] for v in y_clean])

        X_norm = self._normalize(X_clean, fit=True)
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        dataset = TimeSeriesDataset(X_norm, y_mapped, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = TransformerClassifier(
            input_size=X_clean.shape[1],
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        class_counts = np.bincount(y_mapped, minlength=3).astype(float) + 1
        class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

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

            scheduler.step()
            avg_loss = total_loss / len(loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= 15:
                    logger.info(f"Transformer Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"Transformer 학습 완료: best_loss={best_loss:.4f}, samples={len(X_clean)}")
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
                "d_model": self.d_model, "nhead": self.nhead,
                "num_layers": self.num_layers, "dropout": self.dropout,
                "input_size": len(self.feature_names),
            },
        }, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.feature_names = data["feature_names"]
        self._mean = data["mean"]
        self._std = data["std"]
        cfg = data["config"]
        self.model = TransformerClassifier(
            input_size=cfg["input_size"], d_model=cfg["d_model"],
            nhead=cfg["nhead"], num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(data["model_state"])
