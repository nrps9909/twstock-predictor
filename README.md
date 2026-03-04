# twstock-predictor

**台股 AI 量化分析系統 v3.0**

> 20 因子評分 → HMM 體制偵測 → ML 集成預測 → LLM 敘事 → 硬性風控 → 即時串流

核心原則：**演算法做評分，ML 做預測，LLM 做萃取與敘事，規則做風控。**

---

## 系統架構

```
┌──────────────────────────────────────────────────────┐
│          Phase 1: 數據收集 (並行)                      │
│  ├─ 股票 OHLCV + 三大法人 (T86/DB)                    │
│  ├─ 技術指標 (TechnicalAnalyzer)                      │
│  ├─ 月營收 (FinMind)                                  │
│  ├─ 全球市場 (yfinance: SOX, TSM, EWT)                │
│  ├─ 總經數據 (yfinance: VIX, USD/TWD, TNX, XLI, SPY) │
│  └─ 新聞/情緒原始資料 (DB)                             │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│          Phase 2: 特徵萃取 (並行)                      │
│  ├─ 20 因子計算 (全部演算法，score 0-1)                 │
│  ├─ HMM 3-state 體制偵測 → regime 權重                 │
│  ├─ ML 模型預測 (LSTM + XGBoost)                      │
│  └─ LLM 情緒萃取 (Claude Haiku，1 次呼叫)              │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│          Phase 3: 多因子評分                            │
│  ├─ HMM 體制調整因子權重 (REGIME_MULTIPLIERS)           │
│  ├─ 加權評分: total_score = Σ(factor × weight)         │
│  ├─ 信號判定: strong_buy/buy/hold/sell/strong_sell     │
│  └─ 信心度 = agreement × strength × coverage           │
│              × freshness × risk_discount               │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│          Phase 4: LLM 敘事生成 (Claude Sonnet)          │
│  ├─ 輸入: 因子分數 + regime + ML 預測 + 情緒           │
│  └─ 輸出: 展望 / 驅動因子 / 風險 / 催化劑 / 關鍵價位   │
│  └─ Fallback: 演算法推理 (LLM 不可用時)                 │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│          Phase 5: 風險控制 + 部位建議                    │
│  ├─ 信號 → 行動映射 + 部位大小 (0-20%)                 │
│  ├─ 迴路斷路器 (drawdown > 15%)                        │
│  ├─ ATR 追蹤止損                                       │
│  └─ 體制轉換減倉 (bull→bear: 強制減倉)                  │
└─────────────────────┬────────────────────────────────┘
                      │
┌─────────────────────▼────────────────────────────────┐
│          Phase 6: 儲存 + 警報                           │
│  ├─ 存入 DB (MarketScanResult + PipelineResult)        │
│  ├─ 警報生成 (signal_change, strong_signal, etc.)      │
│  └─ SSE 串流完成事件                                    │
└──────────────────────────────────────────────────────┘
```

### LLM 使用策略

| 用途 | 模型 | 呼叫次數 |
|------|------|---------|
| 情緒特徵萃取 | Claude Haiku | 1 次 |
| 敘事生成 | Claude Sonnet | 1 次 |
| **總計** | | **2 次** |

技術分析、籌碼分析全部由演算法（20 因子系統）完成——比 LLM 更準確、更快、更可靠。

---

## 20 因子評分引擎

短期因子 (39%):
| 因子 | 權重 | 數據來源 |
|------|------|---------|
| foreign_flow | 11% | T86 三大法人 |
| technical_signal | 8% | RSI/KD/MACD/均線 |
| short_momentum | 7% | 5日動量 |
| trust_flow | 5% | 投信買賣超 |
| volume_anomaly | 4% | 量比 |
| margin_sentiment | 4% | 融資融券 |

中期因子 (32%):
| 因子 | 權重 | 數據來源 |
|------|------|---------|
| trend_momentum | 7% | 20日趨勢 |
| revenue_momentum | 4% | FinMind 月營收 |
| institutional_sync | 4% | 外資+投信同步 |
| volatility_regime | 4% | 波動率 |
| news_sentiment | 3% | LLM 情緒萃取 |
| global_context | 3% | SOX/TSM/EWT |
| margin_quality | 4% | yfinance 季財報 |
| sector_rotation | 3% | 產業輪動 |

長期因子 (29%):
| 因子 | 權重 | 數據來源 |
|------|------|---------|
| ml_ensemble | 7% | LSTM+XGBoost |
| fundamental_value | 6% | P/E + 營收 |
| liquidity_quality | 4% | 流動性 |
| macro_risk | 4% | VIX/匯率/利率 |
| export_momentum | 4% | EWT ETF |
| us_manufacturing | 4% | XLI/SPY |

HMM 3-state 體制偵測動態調整所有因子權重：
- **Bull**: 短期因子加權，積極交易
- **Sideways**: 均衡權重
- **Bear**: 長期因子加權，保守避險

---

## ML 模型

| 模型 | 擅長 | 架構 |
|------|------|------|
| LSTM + Attention | 時間序列依賴 | 雙層 LSTM → Temporal Attention → FC |
| XGBoost | 表格特徵 | 500 樹 / depth=6 / L1+L2 / 樣本權重 |
| TFT (選配) | 多步預測 | Temporal Fusion Transformer |

- **Triple Barrier 標籤** (ATR-based 動態障礙)
- **PurgedTimeSeriesSplit + CPCV** 防止前看偏誤
- **Stacking Ensemble** (Ridge meta-learner)
- **Meta-Labeling** (GBM) 校準信心度

---

## 風控系統

| 規則 | 限制 | 說明 |
|------|------|------|
| 最大單一倉位 | 20% | 硬性上限 |
| 最大回撤熔斷 | 15% | 觸發後禁止交易 |
| ATR Trailing Stop | 2.5×ATR | 只升不降 |
| 體制轉換減倉 | 自動 | bull→bear 強制減倉 |
| 停損幅度 | ≤ 8% | |
| 風險報酬比 | ≥ 2:1 | |

---

## 技術棧

| 類別 | 技術 |
|------|------|
| 後端 | FastAPI + uvicorn (SSE 串流) |
| 前端 | Next.js 15 + TailwindCSS + Recharts |
| ML | PyTorch, XGBoost, scikit-learn, hmmlearn |
| LLM | Anthropic Claude (Haiku + Sonnet) |
| 資料 | FinMind API, yfinance, TWSE T86 API |
| 資料庫 | SQLAlchemy 2.x + SQLite |
| 排程 | APScheduler (15:00 掃描, 15:30 管線, 16:00 IC) |
| 最佳化 | CVXPY (Mean-Variance) |

---

## 快速開始

### 環境需求

- Python >= 3.12
- Node.js >= 18 (前端)

### 安裝

```bash
git clone https://github.com/nrps9909/twstock-predictor.git
cd twstock-predictor

# Python (uv 或 pip)
uv sync
# 或
pip install -e .

# 前端
cd web && npm install && cd ..
```

### 設定

```bash
cp .env.example .env
```

```env
# 必要
FINMIND_TOKEN=       # https://finmindtrade.com/
ANTHROPIC_API_KEY=   # Claude API

# 選填
DATABASE_URL=sqlite:///./data/twstock.db
```

### 啟動

```bash
# 方法 1: 一鍵啟動 (Windows)
start_dev.bat

# 方法 2: 手動
# 後端 API
uvicorn api.main:app --reload --port 8000

# 前端
cd web && npm run dev

# Streamlit (舊版前端)
streamlit run app/main.py --server.port 8501
```

### 測試

```bash
.venv/Scripts/python -m pytest tests/ -v
```

---

## 專案結構

```
twstock-predictor/
├── api/                           # FastAPI 後端
│   ├── main.py                    #   入口 + 排程
│   ├── routers/                   #   10 個 API 端點
│   │   ├── market.py              #   市場掃描/分析/推薦/管線
│   │   ├── alerts.py              #   警報系統
│   │   └── ...                    #   其他端點
│   └── services/                  #   業務邏輯
│       ├── stock_analysis_service.py  # 統一 6 階段管線
│       ├── market_service.py      #   20 因子評分引擎
│       ├── auto_pipeline_service.py   # 每日自動管線
│       ├── alert_service.py       #   警報生成 (5 類型)
│       └── market_intel_service.py    # 市場情報
├── web/                           # Next.js 前端
│   ├── app/                       #   頁面 (市場/歷史)
│   └── components/                #   分析/圖表/佈局元件
├── src/                           # 核心邏輯
│   ├── agents/                    #   Agent 系統
│   │   ├── narrative_agent.py     #   LLM 情緒萃取 + 敘事生成
│   │   ├── orchestrator.py        #   管線入口 (向後相容)
│   │   ├── risk_agent.py          #   風控規則
│   │   └── base.py                #   基礎類別
│   ├── analysis/                  #   特徵工程
│   │   ├── features.py            #   43 維特徵 + SHAP/MI 篩選
│   │   ├── labels.py              #   Triple Barrier 標籤
│   │   ├── technical.py           #   技術指標
│   │   └── drift.py               #   PSI 特徵漂移偵測
│   ├── models/                    #   ML 模型
│   │   ├── ensemble.py            #   集成預測 + HMM 偵測
│   │   ├── trainer.py             #   訓練管線
│   │   ├── meta_label.py          #   Meta-labeling
│   │   └── cpcv.py                #   CPCV 交叉驗證
│   ├── risk/                      #   風險管理
│   │   └── manager.py             #   ATR Stop + 熔斷
│   ├── data/                      #   資料抓取
│   │   ├── twse_scanner.py        #   T86 三大法人
│   │   └── sentiment_crawler.py   #   PTT/新聞爬蟲
│   ├── db/                        #   SQLAlchemy ORM
│   ├── pipeline/                  #   排程 + 自動重訓練
│   └── monitoring/                #   週報/月報
├── tests/                         # 200+ 測試
├── models/                        # 模型檔 (.pt/.json)
├── data/                          # SQLite DB
└── docs/                          # 架構文件
    └── architecture.md            # 詳細架構說明
```

---

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/api/market/scan` | 市場掃描 (SSE 串流) |
| GET | `/api/market/overview` | 市場總覽 |
| GET | `/api/market/recommendations` | 推薦排名 |
| POST | `/api/market/analyze/{stock_id}` | 個股深度分析 (SSE) |
| GET | `/api/market/pipeline` | 管線結果查詢 |
| POST | `/api/market/pipeline/{stock_id}` | 觸發管線 |
| POST | `/api/market/pipeline/batch` | 批次管線 |
| GET | `/api/market/intel/*` | 市場情報 |
| GET | `/api/market/factor-ic` | 因子 IC 追蹤 |
| GET/POST | `/api/alerts/*` | 警報管理 |

---

## 資料庫 Schema

| 表 | 用途 |
|----|------|
| `StockPrice` | OHLCV + 三大法人 + 融資融券 (as_of_date) |
| `SentimentRecord` | 新聞/PTT 情緒 (as_of_date) |
| `MarketScanResult` | 掃描結果 + 20 因子 + 信心度 |
| `PipelineResult` | 管線分析完整結果 |
| `FactorICRecord` | 因子 IC 追蹤 (Spearman) |
| `Alert` | 警報 (5 類型, 3 嚴重度) |
| `Prediction` | ML 預測記錄 |
| `AgentMemory` | Agent 記憶 |
| `DelistedStock` | 下市股票 (防存活偏誤) |

---

## 設計決策

### 為什麼統一管線取代多 Agent 辯論？

| 原架構 | 問題 |
|--------|------|
| 4 個 LLM Agent 平行分析 | 6-8 次 LLM 呼叫，成本高、延遲大 |
| 多輪 Bull/Bear 辯論 | LLM 對技術/籌碼分析不如演算法 |
| 雙軌信號系統 | Pipeline vs Scan 可能矛盾 |

| 新架構 | 改進 |
|--------|------|
| 20 因子演算法評分 | 客觀、快速、可回測 |
| 2 次 LLM 呼叫 | 成本降 70%，延遲降 60% |
| 單一管線 | 一致的信號系統 |

### 研究基礎

| 研究 | 發現 |
|------|------|
| FINSABER (KDD 2026) | LLM 策略優勢在長期回測中消失 |
| StockBench (2025) | 最好的 LLM agent 僅比 buy-and-hold 多 1.9% |
| TradeTrap (2025) | LLM agent 記憶攻擊成功率 77.97% |

> LLM 適合做特徵萃取和可解釋性報告，不適合直接做交易決策。

---

## 支援的股票

預設追蹤 ~80 支台股，涵蓋 8 個產業：

| 產業 | 代表 |
|------|------|
| 半導體 | 2330 台積電, 2454 聯發科, 3711 日月光 |
| 電子 | 2317 鴻海, 2382 廣達, 2308 台達電 |
| 金融 | 2881 富邦金, 2882 國泰金, 2886 兆豐金 |
| 電信 | 2412 中華電, 3045 台灣大 |
| 傳產 | 1301 台塑, 2002 中鋼 |
| 航運 | 2603 長榮, 2609 陽明 |
| 生技 | 6446 藥華藥, 4743 合一 |
| 綠能 | 6669 緯穎, 3661 世芯-KY |

可在 `api/services/market_service.py` 的 `STOCK_SECTOR` 中新增。

---

## License

MIT
