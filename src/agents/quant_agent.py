"""量化分析 Agent

彙整 ML 模型預測結果，不需 LLM。純數值運算。
"""

import logging

import numpy as np

from src.agents.base import (
    AgentMessage, AgentRole, BaseAgent, MarketContext, Signal,
)

logger = logging.getLogger(__name__)


class QuantAgent(BaseAgent):
    """量化模型彙整 Agent（不需 LLM）"""

    def __init__(self):
        super().__init__(AgentRole.QUANT)

    async def analyze(self, context: MarketContext) -> AgentMessage:
        """彙整所有 ML 模型的預測結果"""
        predictions = context.model_predictions

        # 從各模型取得預測報酬率
        lstm_pred = predictions.get("lstm_return", 0.0)
        xgb_pred = predictions.get("xgb_return", 0.0)
        tft_pred = predictions.get("tft_return", 0.0)
        ensemble_pred = predictions.get("ensemble_return", 0.0)

        # 模型一致性分析
        preds = [p for p in [lstm_pred, xgb_pred, tft_pred, ensemble_pred] if p != 0]
        if not preds:
            return AgentMessage(
                sender=self.role,
                content={"error": "無模型預測結果"},
                signal=Signal.HOLD,
                confidence=0.0,
                reasoning="無可用的模型預測",
            )

        avg_pred = np.mean(preds)
        std_pred = np.std(preds) if len(preds) > 1 else 0
        agreement = 1.0 - min(std_pred / (abs(avg_pred) + 1e-8), 1.0)

        # 根據預測報酬率生成訊號
        if avg_pred > 0.03:
            signal = Signal.STRONG_BUY
        elif avg_pred > 0.01:
            signal = Signal.BUY
        elif avg_pred < -0.03:
            signal = Signal.STRONG_SELL
        elif avg_pred < -0.01:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # 信心度 = 預測強度 × 模型一致性
        confidence = min(abs(avg_pred) / 0.05, 1.0) * agreement

        content = {
            "avg_prediction": float(avg_pred),
            "prediction_std": float(std_pred),
            "model_agreement": float(agreement),
            "individual_predictions": {
                "lstm": float(lstm_pred),
                "xgboost": float(xgb_pred),
                "tft": float(tft_pred),
                "ensemble": float(ensemble_pred),
            },
            "direction_consensus": "bullish" if avg_pred > 0 else "bearish",
        }

        return AgentMessage(
            sender=self.role,
            content=content,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning=(
                f"模型平均預測報酬 {avg_pred:.2%}，"
                f"一致性 {agreement:.0%}，"
                f"std={std_pred:.4f}"
            ),
        )
