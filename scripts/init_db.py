#!/usr/bin/env python3
"""初始化資料庫 — 建立所有資料表"""

import sys
from pathlib import Path

# 確保能 import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db.database import init_db


def main():
    print("初始化資料庫...")
    init_db()
    print("資料庫初始化完成！")


if __name__ == "__main__":
    main()
