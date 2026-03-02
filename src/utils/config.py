"""設定管理模組"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 載入 .env
_project_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """集中管理所有設定值"""

    # 專案根目錄
    PROJECT_ROOT: Path = _project_root

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", f"sqlite:///{_project_root / 'data' / 'twstock.db'}"
    )

    # FinMind
    FINMIND_TOKEN: str = os.getenv("FINMIND_TOKEN", "")
    FINMIND_BASE_URL: str = "https://api.finmindtrade.com/api/v4/data"

    # Firecrawl
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")

    # LLM
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # 預設參數
    DEFAULT_LOOKBACK_DAYS: int = 90
    DEFAULT_PREDICT_DAYS: int = 5
    LSTM_SEQ_LEN: int = 60


settings = Settings()
