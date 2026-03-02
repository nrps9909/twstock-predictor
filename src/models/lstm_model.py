"""LSTM 時間序列預測模型

Input:  (batch, seq_len=60, features=N) — 過去 60 個交易日的特徵
Output: (batch, predict_days) — 未來 N 日預測報酬率
"""

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import settings

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Temporal attention over LSTM hidden states"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) — attention-weighted sum
        """
        # (batch, seq_len, 1)
        scores = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        # (batch, hidden_size)
        context = (weights * lstm_out).sum(dim=1)
        return context


class StockLSTM(nn.Module):
    """雙層 LSTM + FC 預測模型"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        use_attention: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        if use_attention:
            self.attention = AttentionLayer(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, output_size)
        """
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            # Attention-weighted context over all time steps
            context = self.attention(lstm_out)
        else:
            # 取最後一個時間步
            context = lstm_out[:, -1, :]

        return self.fc(context)


class LSTMPredictor:
    """LSTM 模型訓練/預測封裝"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
        lr: float = 1e-3,
        use_attention: bool = False,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StockLSTM(
            input_size, hidden_size, num_layers, dropout, output_size,
            use_attention=use_attention,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def _normalize(
        self, X: np.ndarray, fit: bool = False
    ) -> np.ndarray:
        """Z-score 正規化"""
        if fit:
            # (features,) — 在 seq 和 sample 維度上計算
            self.scaler_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            self.scaler_std = X.reshape(-1, X.shape[-1]).std(axis=0)
            self.scaler_std[self.scaler_std == 0] = 1  # 避免除零
        return (X - self.scaler_mean) / self.scaler_std

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> dict:
        """訓練模型（含 early stopping）

        Args:
            X_train: (samples, seq_len, features)
            y_train: (samples,) or (samples, output_size)
            patience: early stopping 容忍輪數（val loss 不改善即停止）

        Returns:
            {"train_loss": [...], "val_loss": [...]}
        """
        X_train = self._normalize(X_train, fit=True)
        if X_val is not None:
            X_val = self._normalize(X_val)

        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(-1) if y_train.ndim == 1 else torch.FloatTensor(y_train),
        )
        # Bug 1 fix: shuffle=False — 時間序列不可隨機排列 mini-batch
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        history = {"train_loss": [], "val_loss": []}

        # Bug 4 fix: Early stopping with best model checkpoint
        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = self.criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            # Validation + early stopping
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val, already_normalized=True)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.6f)",
                        epoch + 1, best_val_loss,
                    )
                    break

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                logger.info(msg)

        # Restore best model if early stopping was active
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("已恢復最佳模型 (val_loss=%.6f)", best_val_loss)

        return history

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        already_normalized: bool = False,
    ) -> float:
        """評估模型，回傳 MSE"""
        if not already_normalized:
            X = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = torch.FloatTensor(y).unsqueeze(-1) if y.ndim == 1 else torch.FloatTensor(y)
            y_t = y_t.to(self.device)
            pred = self.model(X_t)
            loss = self.criterion(pred, y_t)
        return loss.item()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測

        Args:
            X: (samples, seq_len, features) or (1, seq_len, features)

        Returns:
            (samples, output_size)
        """
        X = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            pred = self.model(X_t)
        return pred.cpu().numpy()

    def save(self, path: str | Path):
        """儲存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
            },
            path,
        )
        logger.info("LSTM 模型已儲存至 %s", path)

    def load(self, path: str | Path):
        """載入模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler_mean = checkpoint["scaler_mean"]
        self.scaler_std = checkpoint["scaler_std"]
        logger.info("LSTM 模型已載入自 %s", path)
