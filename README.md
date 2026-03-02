# twstock-predictor

**台股 AI 量化交易系統 v0.2**

> 資料抓取 → Triple Barrier 標籤 → 特徵篩選 → ML 集成預測 → HMM 狀態偵測 → 規則引擎決策 → 硬性風控 → 自動交易

核心原則：**LLM 是研究助手，不是交易員。ML 做預測，規則做風控。**

---

## 系統架構

```
資料層
  FinMind API ──→ 日K線、三大法人、融資融券
  Firecrawl   ──→ PTT 股票板、鉅亨網新聞
  Claude Haiku ──→ 結構化情緒提取
        │
        ▼
特徵工程（SHAP/MI 篩選 43 → 15-20 維）
        │
        ├──→ Triple Barrier 標籤（ATR 動態障礙）
        │    取代 naive pct_change(5).shift(-5)
        │
        ├──→ 樣本唯一性權重（Average Uniqueness）
        │
        ▼
ML 模型層（LSTM + XGBoost + TFT）
        │
        ├──→ Purged Walk-Forward CV（purging + embargo）
        │
        ├──→ HMM 3-State 狀態偵測 ──→ 動態權重分配
        │    Bull(1.0x) / Sideways(0.5x) / Bear(0.3x)
        │
        ▼
信號產生（ML 信號 + 信心區間）
        │
        ├──→ LLM Agent（情緒分析 + 多空辯論）← 僅供參考，佔 30%
        │
        ▼
規則引擎決策（ML 信號 70% + Agent 建議 30%）
        │
        ├──→ 硬性風控（LLM 無法覆蓋）
        │        ├── 1/4 Kelly 倉位上限
        │        ├── ATR Trailing Stop（追蹤停損）
        │        ├── 最大回撤 15% 熔斷
        │        └── 倉位 / 停損 / 風險報酬比限制
        │
        ▼
執行 + 即時監控 + 記憶回饋
```

---

## 核心模組

### 1. 資料抓取 (`src/data/`)

| 模組 | 來源 | 內容 |
|------|------|------|
| `stock_fetcher.py` | FinMind API | 日K線 OHLCV、三大法人買賣超、融資融券 |
| `sentiment_crawler.py` | PTT、鉅亨網 | 社群文章、新聞標題 |
| `news_crawler.py` | 鉅亨網、工商、經濟日報 | 財經新聞 |

- 指數退避重試（1s → 2s → 4s），區分暫時性 / 永久性錯誤
- Token Bucket 速率限制
- 情緒提取優先 Claude Haiku，regex 做 fallback

### 2. 特徵工程 (`src/analysis/`)

**43 維原始特徵：**

```
價格 (8)     │ close, open, high, low, volume, return_1d/5d/20d
技術 (17)    │ SMA(5/20/60), RSI, KD, MACD, BIAS, BB, OBV, ADX
情緒 (5)     │ sentiment_score, sentiment_ma5, change, post_volume, bullish_ratio
籌碼 (5)     │ foreign/trust/dealer_buy_sell, margin/short_balance
波動率 (3)   │ realized_vol_5d/20d, Parkinson high-low vol
微結構 (2)   │ volume_ratio_5d, spread_proxy
日曆 (3)     │ day_of_week, month, is_settlement
```

**特徵篩選（43 → 15-20 維）：**

- **Mutual Information** — 預設方法，衡量特徵與 target 的非線性相關
- **SHAP** — 基於輕量 XGBoost 的 Shapley 重要性（需安裝 `shap`）
- 自動移除相關係數 > 0.95 的共線特徵

**Triple Barrier 標籤** (`src/analysis/labels.py`)：

取代 naive `pct_change(5).shift(-5)`，三重障礙同時設定止盈、停損、到期：

| 障礙 | 觸發條件 | 標籤 |
|------|---------|------|
| 上障礙 | 價格觸及 entry + ATR × multiplier | 正報酬率 |
| 下障礙 | 價格觸及 entry - ATR × multiplier | 負報酬率 |
| 時間障礙 | max_holding 天後未觸及任何障礙 | 到期時實際報酬率 |

搭配 **樣本唯一性權重**（Average Uniqueness）傳入 XGBoost `sample_weight`，降低重疊標籤的影響。

### 3. ML 模型層 (`src/models/`)

| 模型 | 擅長 | 架構 |
|------|------|------|
| LSTM + Attention | 時間序列長期依賴 | 雙層 LSTM → Temporal Attention → FC |
| XGBoost | 表格特徵、缺失值穩健 | 500 樹 / depth=6 / L1+L2 正則 / 樣本權重 |
| TFT | 多步預測、特徵自動選擇 | Temporal Fusion Transformer |

**集成策略：**
- Weighted Ensemble（inverse MSE 動態權重）
- Stacking Ensemble（Ridge 做 meta-learner）
- 幾何複利 `np.cumprod(1+r)` + Log-space 信心區間

**Purged Walk-Forward CV** (`PurgedTimeSeriesSplit`)：

```
Train ─────────── [purge] [embargo] ── Test ──────
                   10 days   5 days

purge: 移除訓練集末尾與測試集標籤重疊的樣本（≥ max_holding）
embargo: purge 後額外安全間隔，防止序列相關洩漏
```

**HMM 市場狀態偵測** (`HMMStateDetector`)：

3-state Gaussian HMM，觀測 [daily_return, realized_volatility]：

| 狀態 | 信號縮放 | 行為 |
|------|---------|------|
| Bull（牛市） | × 1.0 | 正常執行信號 |
| Sideways（盤整） | × 0.5 | 信號強度減半 |
| Bear（熊市） | × 0.3 | 大幅降低信號，低強度 buy → hold |

核心價值：**告訴你何時不要交易**。

### 4. Multi-Agent 決策層 (`src/agents/`)

參考 [TradingAgents](https://arxiv.org/abs/2412.20138) + [FinMem](https://arxiv.org/abs/2311.11340) (AAAI 2024)。

**架構改進：LLM Agent 降級為「顧問」角色。**

```
Phase 1（並行）：4 個分析師 Agent 提供觀點
    ├── 技術面 Agent (Claude Haiku)  → RSI、MACD、均線解讀
    ├── 情緒面 Agent (Claude Sonnet) → PTT 輿論、新聞風向
    ├── 基本面 Agent (Claude Haiku)  → 法人動向、融資融券
    └── 量化面 Agent (無 LLM)        → ML 模型預測彙整

Phase 2：研究員 Agent (Claude Sonnet) — 僅供參考
    → Bull vs Bear 多空辯論 → 綜合建議

Phase 3：規則引擎決策（取代 LLM 直接決策）
    → ML 信號 × 70% + Agent 建議 × 30%
    → HMM 市場狀態調整
    → 閾值決策: > 0.25 → buy / < -0.25 → sell / else → hold

Phase 4：硬性風控（LLM 無法覆蓋）
    → 熔斷檢查 → 倉位限制 → Trailing Stop → 風險報酬比

Phase 5：記憶更新 + 層級轉移
```

**為什麼降級 LLM？**

| 論文 | 結論 |
|------|------|
| FINSABER (KDD 2026) | LLM 策略優勢在更廣泛股票和更長期限下完全消失 |
| TradeTrap (2025) | LLM agent 記憶攻擊成功率 77.97% |
| StockBench (2025) | 最好的 LLM agent 僅比 buy-and-hold 多賺 1.9% |

### 5. 風控層 (`src/risk/`)

**硬性風控（LLM 無法覆蓋）：**

| 規則 | 限制 | 說明 |
|------|------|------|
| 最大單一倉位 | 20% | 任何來源都不可超過 |
| 最多持倉數 | 5 檔 | |
| 停損幅度 | ≤ 8% | |
| 風險報酬比 | ≥ 2:1 | |
| Kelly 倉位 | 1/4 Kelly | 保守版 Kelly criterion |
| 最大回撤熔斷 | 15% | 觸發後禁止一切買入，需手動重置 |

**ATR Trailing Stop（追蹤停損）：**

```
進場 → 設定初始停損 = entry - 2.5 × ATR
       │
       ├→ 價格上漲 → 停損跟隨上移（只升不降）
       │              new_stop = highest - 2.5 × ATR
       │
       └→ 價格跌破停損 → 觸發出場，禁止加倉
```

**記憶層級轉移（FinMem 升級）：**

| 條件 | 動作 |
|------|------|
| 勝率 ≥ 60%（3+ 筆） | 短期記憶模式 → 升級到長期記憶 |
| 連續虧損 3 次 | 相關模式標記為不可靠，降低參考權重 |

### 6. 回測引擎 (`src/backtest/`)

- 事件驅動，含滑價 + 台股交易成本（手續費 0.1425% × 2.8 折 + 證交稅 0.3%）
- 績效指標：Sharpe ratio、最大回撤、勝率、獲利因子、月報酬、權益曲線

### 7. 即時排程 (`src/pipeline/`)

| 時間 | 任務 |
|------|------|
| 08:30 盤前 | 資料更新 + Agent 分析 → 推播通知 |
| 09:00–13:30 盤中 | 每 30 分鐘監控 Trailing Stop / 止盈 |
| 14:00 盤後 | 結算、記憶層級轉移 |
| 20:00 晚間 | 情緒爬蟲、模型再訓練 |

### 8. 前端 (`app/`)

Streamlit 三頁式儀表板：

| 頁面 | 功能 |
|------|------|
| 技術分析 | K 線圖 + 技術指標 + 買賣訊號 |
| 情緒分析 | PTT/新聞情緒趨勢 + 籌碼分析 |
| 走勢預測 | ML 預測 + 信心區間 + Agent 辯論 + 回測 |

---

## 快速開始

### 環境需求

- Python ≥ 3.12
- 約 2GB 磁碟空間（PyTorch + 依賴）

### 安裝

```bash
git clone https://github.com/your-username/twstock-predictor.git
cd twstock-predictor

# 建議使用 uv 管理
uv sync

# 或 pip
pip install -e .

# 可選：SHAP 特徵重要性分析
pip install -e ".[analysis]"
```

### 設定

```bash
cp .env.example .env
```

編輯 `.env`：

```env
# 必要
FINMIND_TOKEN=       # 免費註冊: https://finmindtrade.com/
ANTHROPIC_API_KEY=   # Claude API

# 選填
FIRECRAWL_API_KEY=   # 新聞爬蟲
OPENAI_API_KEY=      # Embedding fallback
DATABASE_URL=sqlite:///./data/twstock.db
```

### 啟動

```bash
# 前端
streamlit run app/main.py

# 即時管線（含排程）
python -m src.pipeline.realtime
```

### 訓練模型

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer(stock_id="2330")

# 完整訓練（Triple Barrier + 特徵篩選 + HMM）
results = trainer.train(
    start_date="2022-01-01",
    end_date="2025-12-31",
    use_triple_barrier=True,       # Triple Barrier 標籤
    max_features=20,               # 特徵篩選至 20 維
    feature_selection_method="mutual_info",
)

# Purged Walk-Forward 驗證
cv_results = trainer.walk_forward_validate(
    start_date="2022-01-01",
    end_date="2025-12-31",
    n_splits=5,
    purge_days=10,     # >= max_holding
    embargo_days=5,
)

# 預測（含 HMM 狀態偵測）
prediction = trainer.predict(
    start_date="2025-10-01",
    end_date="2025-12-31",
)
print(prediction.signal, prediction.signal_strength)
print(prediction.market_state)  # HMM 狀態
```

---

## 技術棧

| 類別 | 技術 |
|------|------|
| ML | PyTorch 2.2+, XGBoost 2.0+, scikit-learn 1.4+, pytorch-forecasting |
| 市場狀態 | hmmlearn 0.3+ (Gaussian HMM) |
| 波動率 | arch 7.0+ (GARCH) |
| LLM | Anthropic Claude (Haiku 快速 / Sonnet 深度) |
| Agent 編排 | LangGraph 0.2+, 自建 async Agent 框架 |
| 資料 | FinMind API, Firecrawl, pandas 2.2+ |
| 資料庫 | SQLAlchemy 2.0+ / SQLite (6 張表) |
| 前端 | Streamlit 1.35+ / Plotly 5.22+ |
| 排程 | APScheduler 3.10+ |
| 通知 | LINE Notify, Telegram Bot |
| 最佳化 | CVXPY 1.5+ (Mean-Variance) |

---

## 專案結構

```
twstock-predictor/
├── app/                          # Streamlit 前端
│   ├── main.py                   #   入口
│   ├── pages/                    #   三頁式 UI
│   └── components/               #   圖表、側邊欄元件
├── src/                          # 核心邏輯
│   ├── analysis/                 # 特徵工程
│   │   ├── features.py           #   43 維特徵 + SHAP/MI 篩選
│   │   ├── labels.py             #   Triple Barrier 標籤 + 樣本權重
│   │   ├── technical.py          #   17 種技術指標
│   │   ├── sentiment.py          #   情緒分析
│   │   └── llm_features.py       #   LLM Embedding + PCA 降維
│   ├── models/                   # ML 模型
│   │   ├── lstm_model.py         #   LSTM + Attention
│   │   ├── xgboost_model.py      #   XGBoost（含樣本權重）
│   │   ├── tft_model.py          #   Temporal Fusion Transformer
│   │   ├── ensemble.py           #   集成預測 + HMM 狀態偵測
│   │   ├── trainer.py            #   訓練管線 + Purged Walk-Forward CV
│   │   ├── backtester.py         #   回測引擎
│   │   └── data_module.py        #   PyTorch DataLoader
│   ├── agents/                   # Multi-Agent 系統
│   │   ├── orchestrator.py       #   DAG 編排 + 規則引擎 (70/30)
│   │   ├── researcher_agent.py   #   多空辯論（Claude Sonnet）
│   │   ├── trader_agent.py       #   交易建議（降級為顧問）
│   │   ├── risk_agent.py         #   風控規則
│   │   ├── memory.py             #   三層記憶 + 層級轉移
│   │   ├── technical_agent.py    #   技術面觀點
│   │   ├── sentiment_agent.py    #   情緒面觀點
│   │   ├── fundamental_agent.py  #   基本面觀點
│   │   ├── quant_agent.py        #   量化面觀點
│   │   └── base.py               #   基礎類別 + 資料結構
│   ├── risk/                     # 風險管理
│   │   ├── manager.py            #   Kelly / ATR Trailing Stop / 熔斷
│   │   └── portfolio.py          #   多檔持倉追蹤
│   ├── data/                     # 資料抓取
│   ├── db/                       # SQLAlchemy ORM (6 張表)
│   ├── backtest/                 # 回測框架
│   ├── pipeline/                 # APScheduler 即時管線
│   ├── portfolio/                # Mean-Variance 最佳化
│   ├── monitoring/               # LINE / Telegram 通知
│   └── utils/                    # 設定、常數、重試
├── models/                       # 訓練好的模型檔
├── data/                         # SQLite 資料庫
├── .env.example                  # 環境變數範本
└── pyproject.toml                # 依賴定義
```

---

## 資料庫 Schema

| 表 | 用途 |
|----|------|
| `StockPrice` | 日 K 線 OHLCV + 三大法人 + 融資融券 |
| `SentimentRecord` | 社群情緒（來源、標題、分數、關鍵字） |
| `Prediction` | ML 預測記錄（含信心區間） |
| `BacktestResult` | Walk-Forward 回測結果 |
| `AgentMemory` | 三層記憶（短期/長期 + embedding） |
| `TradeJournal` | 交易日誌（完整推理鏈 + 事後檢討） |

---

## 設計決策與研究基礎

### 為什麼不讓 LLM 直接做交易決策？

| 研究 | 發現 |
|------|------|
| FINSABER (KDD 2026, arXiv 2505.07078) | 20 年回測：LLM 優勢在更廣泛股票和更長期限下**完全消失** |
| StockBench (arXiv 2510.02209) | 最好的 LLM agent 僅比 buy-and-hold 多賺 1.9% |
| TradeTrap (arXiv 2512.02261) | LLM agent 記憶攻擊成功率 77.97% |
| 2026 偏誤審查 (arXiv 2602.14233) | 164 篇論文中沒有任何一種偏誤被超過 28% 的論文討論 |

> "LLMs are NOT great at inventing alpha — they remove friction around the work that surrounds it."
> — Ilya Navogitsyn, Dataconomy 2026

### 為什麼用 HMM？

多項獨立研究驗證 HMM 狀態切換的價值：
- Sharpe 從 0.67 → 1.05（NIFTY 50, 2018-2024）
- 核心價值：告訴你**何時不要交易**

### 為什麼用 Triple Barrier？

- Marcos López de Prado, *Advances in Financial Machine Learning* (2018)
- 自適應市場波動率（ATR-based 動態障礙）
- 搭配 Purged CV 才能產生可信的回測結果

### 實際績效期望

- 散戶 AI 系統實盤 Sharpe：**0.8 – 1.5**
- 超過 2.0 幾乎確定是過擬合
- 實盤表現通常比回測降 30-50%

---

## 開發

```bash
# 安裝開發依賴
pip install -e ".[dev]"

# 測試
pytest tests/

# 型別檢查（選用）
mypy src/
```

---

## 支援的股票

預設追蹤台股前 10 大權值股：

| 代碼 | 名稱 |
|------|------|
| 2330 | 台積電 |
| 2317 | 鴻海 |
| 2382 | 廣達 |
| 2454 | 聯發科 |
| 2881 | 富邦金 |
| 2882 | 國泰金 |
| 2303 | 聯電 |
| 3711 | 日月光投控 |
| 2308 | 台達電 |
| 2412 | 中華電 |

可在 `src/utils/constants.py` 中新增。

---

## License

MIT
