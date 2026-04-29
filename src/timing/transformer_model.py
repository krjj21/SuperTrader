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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from loguru import logger

from src.timing.lstm_model import TimeSeriesDataset


class PnLAwareDataset(Dataset):
    """시계열 시퀀스 + (선택적) forward_return 함께 반환.

    Momentum Transformer paper (Wood et al. 2021) 의 transaction-cost-aware
    loss 입력. forward_returns 가 None 이면 기존 TimeSeriesDataset 와 동일 동작
    (sample_weight = 1, no transaction cost penalty).
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_length: int,
        forward_returns: np.ndarray | None = None,
    ):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_length = seq_length
        if forward_returns is not None:
            # NaN → 0 (학습 영향 없게)
            fr = np.nan_to_num(forward_returns, nan=0.0).astype(np.float32)
            self.fr = torch.FloatTensor(fr)
        else:
            self.fr = None

    def __len__(self):
        return len(self.X) - self.seq_length + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        target_idx = idx + self.seq_length - 1
        y_label = self.y[target_idx]
        if self.fr is not None:
            fr_val = self.fr[target_idx]
        else:
            fr_val = torch.tensor(0.0)
        return x_seq, y_label, fr_val


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
        # Momentum Transformer (Wood et al. 2021) — transaction-cost-aware loss
        pnl_alpha: float = 5.0,           # PnL emphasis: weight = 1 + alpha · |forward_return|
        trade_cost: float = 0.004,        # 한국 마찰비용 commission + tax (왕복 0.4%)
        cost_lambda: float = 1.0,         # transaction cost 페널티 가중
    ):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.pnl_alpha = float(pnl_alpha)
        self.trade_cost = float(trade_cost)
        self.cost_lambda = float(cost_lambda)
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
        forward_returns: pd.Series | None = None,
    ) -> dict:
        """Transformer 학습.

        Args:
            forward_returns: 미래 수익률 raw 값 (Momentum Transformer paper 의 PnL-weighted
                loss 입력). None 이면 기존 표준 CE loss 사용 (backward compat).
        """
        mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[mask].values.astype(np.float32)
        y_clean = y[mask].values

        # forward_returns 도 같은 mask 적용
        if forward_returns is not None:
            fr_clean = forward_returns[mask].values.astype(np.float32)
        else:
            fr_clean = None

        if len(X_clean) < self.sequence_length + 50:
            return {"error": "insufficient_data"}

        self.feature_names = X.columns.tolist()
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = np.array([label_map[v] for v in y_clean])

        X_norm = self._normalize(X_clean, fit=True)
        X_norm = np.nan_to_num(X_norm, nan=0.0)

        # forward_returns 있으면 PnL-aware dataset, 없으면 기존 dataset
        if fr_clean is not None:
            dataset = PnLAwareDataset(X_norm, y_mapped, self.sequence_length, fr_clean)
            cost_aware = True
        else:
            dataset = TimeSeriesDataset(X_norm, y_mapped, self.sequence_length)
            cost_aware = False
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

        # cost-aware 모드면 reduction='none' (per-sample weighting), 아니면 표준 CE
        if cost_aware:
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_loss = float("inf")
        patience_count = 0

        if cost_aware:
            logger.info(
                f"Transformer cost-aware loss 활성: pnl_alpha={self.pnl_alpha}, "
                f"trade_cost={self.trade_cost}, cost_lambda={self.cost_lambda}"
            )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in loader:
                if cost_aware:
                    batch_x, batch_y, batch_fr = batch
                    batch_fr = batch_fr.to(self.device)
                else:
                    batch_x, batch_y = batch
                    batch_fr = None
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)

                if cost_aware:
                    # ── Momentum Transformer (Wood et al. 2021) cost-aware loss ──
                    # 1) per-sample CE
                    ce_per = criterion(output, batch_y)
                    # 2) PnL emphasis: |forward_return| 큰 sample 에 더 가중
                    pnl_weight = 1.0 + self.pnl_alpha * batch_fr.abs()
                    weighted_ce = (ce_per * pnl_weight).mean()
                    # 3) Transaction cost penalty: BUY/SELL 예측 확률 × 비용
                    softmax_out = torch.softmax(output, dim=-1)
                    # label_map: {-1:0, 0:1, 1:2} → SELL=0, HOLD=1, BUY=2
                    trade_prob = softmax_out[:, 0] + softmax_out[:, 2]
                    cost_pen = self.cost_lambda * self.trade_cost * trade_prob.mean()
                    loss = weighted_ce + cost_pen
                else:
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

        loss_type = "cost-aware" if cost_aware else "standard CE"
        logger.info(
            f"Transformer 학습 완료 ({loss_type}): "
            f"best_loss={best_loss:.4f}, samples={len(X_clean)}"
        )
        return {
            "best_loss": best_loss,
            "n_samples": len(X_clean),
            "cost_aware": cost_aware,
        }

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
