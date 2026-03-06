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

from src.models.ensemble import direction_accuracy as _direction_accuracy

logger = logging.getLogger(__name__)


class DirectionAwareLoss(nn.Module):
    """Huber loss + direction penalty for prediction-direction alignment.

    When pred and target have opposite signs, adds a penalty proportional
    to direction_weight. This aligns the training objective with the
    quality gate's direction_accuracy metric.
    """

    def __init__(self, delta: float = 2.0, direction_weight: float = 0.2):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.dw = direction_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reg_loss = self.huber(pred, target)
        wrong_dir = (pred * target < 0).float().mean()
        return reg_loss + self.dw * wrong_dir


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
    """Dual-task LSTM: regression head + classification head.

    The classification head predicts direction {up, down, neutral} alongside
    the regression head. This dual-task approach improves robustness in
    extreme market conditions (2024-2025 research consensus).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
        use_attention: bool = False,
        use_classification: bool = True,
        n_classes: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.use_classification = use_classification

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        if use_attention:
            self.attention = AttentionLayer(hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)

        # Regression head (return prediction)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

        # Classification head (direction prediction: up/neutral/down)
        if use_classification:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, n_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            If use_classification: (regression_out, class_logits)
            Else: regression_out only
        """
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            context = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]

        context = self.layer_norm(context)
        reg_out = self.fc(context)

        if self.use_classification:
            cls_logits = self.classifier(context)
            return reg_out, cls_logits

        return reg_out


class LSTMPredictor:
    """LSTM 模型訓練/預測封裝 (supports dual-task: regression + classification)"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size: int = 1,
        lr: float = 5e-4,
        use_attention: bool = False,
        use_classification: bool = True,
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_classification = use_classification
        self.model = StockLSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout,
            output_size,
            use_attention=use_attention,
            use_classification=use_classification,
        ).to(self.device)
        self.lr = lr
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.criterion = DirectionAwareLoss(delta=2.0, direction_weight=0.2)
        self.cls_criterion = nn.CrossEntropyLoss() if use_classification else None
        self.cls_weight = 0.3  # weight for classification loss in dual-task
        self.target_scale = 100.0  # work in percentage space
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score 正規化"""
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            # (features,) — 在 seq 和 sample 維度上計算
            self.scaler_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
            self.scaler_std = X.reshape(-1, X.shape[-1]).std(axis=0)
            self.scaler_std[self.scaler_std == 0] = 1  # 避免除零
        return (X - self.scaler_mean) / self.scaler_std

    @staticmethod
    def _make_direction_labels(y: np.ndarray, threshold: float = 0.003) -> np.ndarray:
        """Convert regression targets to 3-class direction labels.

        Classes: 0=down, 1=neutral, 2=up (threshold=±0.3% after scaling).
        """
        labels = np.ones(len(y), dtype=np.int64)  # default neutral
        y_flat = y.flatten()
        labels[y_flat > threshold] = 2  # up
        labels[y_flat < -threshold] = 0  # down
        return labels

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 15,
    ) -> dict:
        """訓練模型（含 early stopping + dual-task classification）

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

        # Generate direction labels BEFORE scaling (in return space)
        cls_train = None
        cls_val = None
        if self.use_classification:
            cls_train = self._make_direction_labels(y_train)
            if y_val is not None:
                cls_val = self._make_direction_labels(y_val)

        # Scale targets to percentage space for better gradient flow
        y_train = y_train * self.target_scale
        if y_val is not None:
            y_val = y_val * self.target_scale

        tensors = [
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(-1)
            if y_train.ndim == 1
            else torch.FloatTensor(y_train),
        ]
        if cls_train is not None:
            tensors.append(torch.LongTensor(cls_train))
        dataset = TensorDataset(*tensors)
        # Shuffle mini-batch order (each sequence preserves internal temporal order)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": []}
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr * 0.01
        )

        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_data in loader:
                if self.use_classification and len(batch_data) == 3:
                    X_batch, y_batch, cls_batch = batch_data
                    cls_batch = cls_batch.to(self.device)
                else:
                    X_batch, y_batch = batch_data[:2]
                    cls_batch = None
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)

                if self.use_classification and isinstance(output, tuple):
                    reg_pred, cls_logits = output
                    reg_loss = self.criterion(reg_pred, y_batch)
                    cls_loss = (
                        self.cls_criterion(cls_logits, cls_batch)
                        if cls_batch is not None
                        else 0.0
                    )
                    loss = reg_loss + self.cls_weight * cls_loss
                else:
                    pred = output if not isinstance(output, tuple) else output[0]
                    loss = self.criterion(pred, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["train_loss"].append(avg_loss)

            # Validation + early stopping (composite: val_loss - 0.1 * direction_acc)
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(
                    X_val, y_val, already_normalized=True, targets_prescaled=True
                )
                history["val_loss"].append(val_loss)

                # Composite metric: lower is better (penalize wrong direction)
                val_dir = self.evaluate_directional(
                    X_val, y_val, already_normalized=True
                )
                composite = val_loss - 0.1 * val_dir["direction_acc"]

                if composite < best_val_loss:
                    best_val_loss = composite
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping at epoch %d (best val_loss=%.6f)",
                        epoch + 1,
                        best_val_loss,
                    )
                    break

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.6f}"
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
        targets_prescaled: bool = False,
    ) -> float:
        """評估模型，回傳 loss (in scaled space)"""
        if not already_normalized:
            X = self._normalize(X)
        y_scaled = y if targets_prescaled else y * self.target_scale
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            y_t = (
                torch.FloatTensor(y_scaled).unsqueeze(-1)
                if y_scaled.ndim == 1
                else torch.FloatTensor(y_scaled)
            )
            y_t = y_t.to(self.device)
            output = self.model(X_t)
            pred = output[0] if isinstance(output, tuple) else output
            loss = self.criterion(pred, y_t)
        return loss.item()

    def evaluate_directional(
        self,
        X: np.ndarray,
        y: np.ndarray,
        already_normalized: bool = False,
    ) -> dict:
        """Evaluate MSE + direction accuracy + naive baseline comparison"""
        if not already_normalized:
            X = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            output = self.model(X_t)
            reg_out = output[0] if isinstance(output, tuple) else output
            pred = reg_out.cpu().numpy().flatten() / self.target_scale

        y_flat = y.flatten()
        mse = float(np.mean((pred - y_flat) ** 2))
        naive_mse = float(np.mean(y_flat**2))  # Predict 0 (hold)
        direction_acc = _direction_accuracy(pred, y_flat)

        result = {
            "mse": mse,
            "naive_mse": naive_mse,
            "direction_acc": direction_acc,
            "beats_naive": mse < naive_mse,
        }

        # Classification accuracy if dual-task
        if isinstance(output, tuple) and len(output) == 2:
            cls_logits = output[1]
            cls_pred = cls_logits.argmax(dim=1).cpu().numpy()
            true_cls = self._make_direction_labels(y_flat)
            result["cls_accuracy"] = float(np.mean(cls_pred == true_cls))

        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測

        Args:
            X: (samples, seq_len, features) or (1, seq_len, features)

        Returns:
            (samples, output_size) — in original return space
        """
        X = self._normalize(X)
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            output = self.model(X_t)
            reg_out = output[0] if isinstance(output, tuple) else output
        return reg_out.cpu().numpy() / self.target_scale

    def save(self, path: str | Path):
        """儲存模型（含架構 metadata + 訓練時間）"""
        from datetime import date as _date

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "target_scale": self.target_scale,
                "architecture": {
                    "input_size": self.model.lstm.input_size,
                    "hidden_size": self.model.hidden_size,
                    "num_layers": self.model.num_layers,
                    "use_attention": self.model.use_attention,
                    "use_classification": self.model.use_classification,
                },
                "trained_at": _date.today().isoformat(),
            },
            path,
        )
        logger.info("LSTM 模型已儲存至 %s", path)

    def load(self, path: str | Path):
        """載入模型（backward compatible with pre-dual-task checkpoints）"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Auto-detect architecture from checkpoint
        arch = checkpoint.get("architecture", {})
        saved_input_size = arch.get("input_size")
        saved_use_cls = arch.get("use_classification", False)

        if saved_input_size and (
            saved_input_size != self.model.lstm.input_size
            or saved_use_cls != self.model.use_classification
        ):
            logger.info(
                "Rebuilding LSTM: checkpoint input_size=%d cls=%s, current=%d cls=%s",
                saved_input_size,
                saved_use_cls,
                self.model.lstm.input_size,
                self.model.use_classification,
            )
            self.use_classification = saved_use_cls
            self.model = StockLSTM(
                input_size=saved_input_size,
                hidden_size=arch.get("hidden_size", self.model.hidden_size),
                num_layers=arch.get("num_layers", self.model.num_layers),
                use_attention=arch.get("use_attention", self.model.use_attention),
                use_classification=saved_use_cls,
            ).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters())
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler_mean = checkpoint["scaler_mean"]
        self.scaler_std = checkpoint["scaler_std"]
        self.target_scale = checkpoint.get("target_scale", 100.0)
        logger.info(
            "LSTM 模型已載入自 %s (input_size=%d, cls=%s)",
            path,
            self.model.lstm.input_size,
            self.model.use_classification,
        )
