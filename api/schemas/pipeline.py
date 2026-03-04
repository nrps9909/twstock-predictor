"""一鍵預測 Pipeline 相關模型"""

from pydantic import BaseModel
from typing import Any


class PipelineEvent(BaseModel):
    step: str
    status: str  # "running", "done", "error", "skipped"
    progress: int  # 0-100
    message: str = ""
    data: dict[str, Any] | None = None


class PipelineRequest(BaseModel):
    force_retrain: bool = False
    epochs: int = 50
