# 預測模型架構（統一管線 v3.0）

統一後只有**一條管線**，6 個階段，LLM 只呼叫 **2 次**（之前 6-8 次）。

---

## 全局流程圖

```
analyze_stock("2330")
│
├── Phase 1: 數據收集 (並行, ~5s)
│   ├── DB 股價 OHLCV (120 天)
│   ├── T86 三大法人 (TWSEScanner)
│   ├── 技術指標 (TechnicalAnalyzer)
│   ├── FinMind 月營收
│   ├── yfinance 全球 (SOX/TSM/EWT) ←── 每日快取
│   ├── yfinance 總經 (VIX/USD-TWD/TNX/XLI/SPY) ←── 每日快取
│   └── DB 新聞/PTT 情緒 (14 天)
│
├── Phase 2: 特徵萃取 (並行, ~3s)
│   ├── HMM 3-state 體制偵測 → bull/bear/sideways
│   ├── ML 模型預測 (LSTM+XGBoost) → ml_scores
│   └── LLM #1: 情緒萃取 (Haiku) → sentiment_score + key_themes
│
├── Phase 3: 20 因子評分 (~0.1s)
│   ├── 計算 20 個因子 (score 0~1)
│   ├── 體制調整權重 (REGIME_MULTIPLIERS)
│   ├── 缺失數據重分配
│   ├── 加權平均 → total_score (0~1)
│   ├── 訊號判定: strong_buy/buy/hold/sell/strong_sell
│   └── 信心度 = (agreement×0.30 + strength×0.30 + coverage×0.25
│                  + freshness×0.15) × risk_discount
│
├── Phase 4: 敘事生成 (~2s)
│   └── LLM #2: 分析報告 (Sonnet) → outlook/drivers/risks/catalysts
│       └── Fallback: _algorithm_narrative() (純演算法推理)
│
├── Phase 5: 風控 + 部位 (~0.1s)
│   ├── signal → action 映射
│   ├── confidence → position_size (0~20%)
│   ├── Meta-label 校準 (if available)
│   ├── 迴路斷路器 (drawdown > 15%)
│   ├── ATR Trailing Stop
│   ├── 體制轉換減倉 (bull→bear)
│   └── 單一部位上限 20%
│
└── Phase 6: 儲存 + 警報 (~0.2s)
    ├── MarketScanResult (DB)
    ├── PipelineResult (DB)
    ├── FactorICRecord (DB)
    └── Alert 偵測 (5 種)
```

---

## 一、20 因子評分引擎（核心）

這是整個系統的**主引擎**，位於 `api/services/market_service.py` 的 `score_stock()`。

### 因子清單

| # | 因子 | 權重 | 時間層 | 來源 | 子組件 |
|---|------|------|--------|------|--------|
| 1 | `foreign_flow` | 11% | 短期 | T86/DB | net_normalized(40%) + consecutive(20%) + acceleration(20%) + anomaly_z(20%) |
| 2 | `technical_signal` | 8% | 短期 | TechnicalAnalyzer | signal(40%) + adx(20%) + ma_alignment(20%) + obv_divergence(20%) |
| 3 | `short_momentum` | 7% | 短期 | 價格計算 | return_1d/3d/5d(60%) + bias(40%) |
| 4 | `trust_flow` | 5% | 短期 | T86/DB | net_normalized(40%) + consecutive(30%) + acceleration(30%) |
| 5 | `volume_anomaly` | 4% | 短期 | 價格計算 | expansion(50%) + consistency(30%) + obv_trend(20%) |
| 6 | `margin_sentiment` | 4% | 短期 | DB融資券 | margin_trend(50%) + utilization(30%) + short_ratio(20%) **反向** |
| 7 | `trend_momentum` | 7% | 中期 | 價格計算 | return_20d/60d(40%) + ma_alignment(30%) + adx(30%) |
| 8 | `revenue_momentum` | 4% | 中期 | FinMind | yoy(50%) + yoy_accel(30%) + mom(20%) |
| 9 | `institutional_sync` | 4% | 中期 | T86/DB | foreign_trust_sync(40%) + direction(30%) + divergence(30%) |
| 10 | `volatility_regime` | 4% | 中期 | 價格計算 | low_vol_premium(40%) + compression(30%) + bb_percentile(30%) |
| 11 | `news_sentiment` | 3% | 中期 | DB+LLM | source_weighted(40%) + momentum(30%) + engagement(30%) |
| 12 | `global_context` | 3% | 中期 | yfinance | sox_score(60%) + tsm_score(40%) |
| 13 | `margin_quality` | 4% | 中期 | yfinance季報 | gross_margin_trend(60%) + operating_margin_level(40%) |
| 14 | `sector_rotation` | 3% | 中期 | STOCK_SECTOR | flow_vs_market(50%) + return_vs_market(30%) + breadth(20%) |
| 15 | `ml_ensemble` | 7% | 長期 | LSTM+XGBoost | raw_ml_score(100%) |
| 16 | `fundamental_value` | 6% | 長期 | yfinance | pe_ratio(40%) + roe(35%) + dividend_yield(25%) |
| 17 | `liquidity_quality` | 4% | 長期 | 價格計算 | avg_volume(50%) + stability(25%) + spread_proxy(25%) |
| 18 | `macro_risk` | 4% | 長期 | yfinance | vix(40%) + fx_trend(30%) + yield_change(30%) |
| 19 | `export_momentum` | 4% | 長期 | yfinance EWT | ewt_20d(50%) + ewt_60d(30%) + relative_strength(20%) |
| 20 | `us_manufacturing` | 4% | 長期 | yfinance XLI | xli_return(40%) + xli_spy_ratio(40%) + xli_sma200(20%) |

**合計：短期 39% / 中期 32% / 長期 29% = 100%**

### HMM 體制動態權重

3-state Gaussian HMM（`src/models/ensemble.py` `HMMStateDetector`）用 TSMC 報酬率訓練：

```
觀察值: [daily_return × 100, realized_volatility × 100]
狀態分配: 依 mean_return 排序 → 最低=bear, 中間=sideways, 最高=bull
```

每個體制有 20 個乘數（`REGIME_MULTIPLIERS`），例如：

| 因子 | bull 乘數 | bear 乘數 | sideways 乘數 |
|------|----------|----------|-------------|
| short_momentum | **1.3** | 0.5 | 0.8 |
| margin_sentiment | 0.8 | **1.5** | 1.0 |
| volatility_regime | 0.7 | **1.5** | 1.2 |
| technical_signal | 1.1 | 0.8 | **1.3** |

計算流程：

```python
raw_weight[i] = BASE_WEIGHTS[i] × REGIME_MULTIPLIERS[regime][i]
final_weight[i] = raw_weight[i] / Σ(available raw_weights)  # 正規化
# 缺失因子的權重自動重分配給可用因子
```

### 信心度計算

```
raw_confidence = agreement×0.30 + strength×0.30 + coverage×0.25 + freshness×0.15

agreement  = max(bullish_count, bearish_count) / total_available
strength   = |total_score - 0.5| × 2
coverage   = Σ(base_weights for available factors)
freshness  = weighted avg of factor freshness scores

confidence = raw_confidence × risk_discount (floor 0.3)
```

風險折扣觸發條件：

- 年化波動率 > 60%: ×0.70
- 年化波動率 > 40%: ×0.85
- 均量 < 200 張: ×0.60
- 均量 < 500 張: ×0.80
- 融資 5 日暴增 > 10%: ×0.85
- P/E > 80 或 < 0: ×0.75

---

## 二、ML 模型（因子 #15，佔 7%）

ML 是 20 因子之一，不是獨立決策路線。位於 `src/models/trainer.py`。

### 訓練流程

```
1. 特徵工程
   ├── OHLCV 原始 + 技術指標 (ta 庫)
   ├── 情緒指標 (sentiment_score, bullish_ratio)
   ├── 波動率特徵 (realized_vol, parkinson_vol)
   └── 微結構 (volume_ratio, spread_proxy)

2. Triple Barrier 標籤 (ATR-based)
   ├── upper_barrier = entry + 2×ATR
   ├── lower_barrier = entry - 2×ATR
   └── max_holding = 10 days
   → 三分類: +1 (突破上方) / -1 (突破下方) / 0 (超時)

3. 特徵選擇 (MI or SHAP, max 20 features)
   └── 重要性存入 FeatureImportanceRecord 表

4. 資料分割 (Purged Time-Series Split)
   ├── train 60% / val 20% / test 20%
   └── purge_gap = max_holding + embargo (防洩漏)

5. LSTM 訓練 (seq_len=60, early stopping)
   └── Input: (batch, 60, n_features) → Output: (batch, 5) 報酬預測

6. XGBoost 訓練 (tabular)
   └── sample_weight = average uniqueness weights

7. Ensemble 權重校準
   └── 根據 validation MSE 動態調整 lstm_weight/xgb_weight
       (default 0.6/0.4, 按 inverse MSE 更新)

8. HMM 擬合 (3-state on returns)

9. (選用) TFT 訓練 → StackingEnsemble (Ridge meta-learner)

10. (選用) Meta-Labeling
    ├── GradientBoosting + Isotonic Calibration
    ├── P(primary model correct) → bet_size
    └── Kelly-inspired: size = max_position × (2p-1) if p>0.5
```

### 預測流程

```python
# 在 stock_analysis_service._predict_ml() 中
trainer = ModelTrainer(stock_id)
trainer.load_models()  # LSTM .pt + XGBoost .json

result = trainer.predict(start_date, end_date)
# → PredictionResult(signal, signal_strength, predicted_prices, ...)

# 信號映射為 ml_ensemble 因子分數:
signal_score_map = {
    "strong_buy": 0.9, "buy": 0.75,
    "hold": 0.5,
    "sell": 0.25, "strong_sell": 0.1,
}
```

### EnsemblePredictor 預測邏輯

```python
ensemble_returns = lstm_weight × lstm_pred + xgb_weight × xgb_pred
predicted_prices = current_price × cumprod(1 + ensemble_returns)

# 信號生成 (波動率校準)
threshold = max(volatility_multiplier × recent_std, 0.005)
if total_return > threshold:  signal = "buy"
elif total_return < -threshold: signal = "sell"
else: signal = "hold"

# HMM 狀態調整
signal_strength *= state_scale[regime]  # bull=1.0, sideways=0.5, bear=0.3
if regime == "bear" and signal == "buy" and strength < 0.5:
    signal = "hold"  # 熊市降級弱買為持有
```

---

## 三、LLM 使用策略（精簡為 2 次）

位於 `src/agents/narrative_agent.py`。

| | 呼叫 #1 | 呼叫 #2 |
|--|---------|---------|
| **階段** | Phase 2 特徵萃取 | Phase 4 敘事生成 |
| **模型** | Claude Haiku (快/便宜) | Claude Sonnet (品質) |
| **輸入** | 新聞+PTT+法人+全球 | 20 因子分數+regime+ML+信心 |
| **輸出** | `sentiment_score` + `key_themes` + `contrarian_flag` + `geopolitical_risk` | `outlook` + `key_drivers` + `risks` + `catalysts` + `key_levels` |
| **用途** | 增強 `news_sentiment` 因子 | 人類可讀分析報告 |
| **Fallback** | 使用 DB 平均情緒分數 | `_algorithm_narrative()` 演算法推理 |

### 被移除的 LLM 呼叫

- ~~TechnicalAgent (Haiku ×1)~~ — 被 20 因子演算法取代
- ~~FundamentalAgent (Haiku ×1)~~ — 被 20 因子演算法取代
- ~~SentimentAgent 4 層分析 (Haiku ×1)~~ — 精簡為結構化萃取
- ~~QuantAgent (Haiku ×1)~~ — 被 ML 因子取代
- ~~ResearcherAgent 多輪辯論 (Haiku ×2-4)~~ — 被多因子共識度取代
- ~~Trader/綜合判斷 (Sonnet ×1)~~ — 被規則引擎取代

---

## 四、風控系統

位於 `src/risk/manager.py`，是**硬性規則**，LLM 和評分都無法覆蓋。

```
風控層級:

1. 迴路斷路器
   └── 最大回撤 > 15% → 禁止一切買入

2. 部位上限
   └── 單一股票 ≤ 20% 資金

3. ATR Trailing Stop
   ├── 初始停損 = entry_price - 2.5×ATR
   ├── 只能上移，不能下移
   └── 觸發後禁止加碼

4. 體制轉換
   ├── bull → bear (severity=1.0): 全部減倉 50%
   ├── sideways → bear (severity=0.7): 減倉 50%
   └── bear → bull (severity=0.5): 可增加曝險

5. Kelly 倉位 (1/4 Kelly)
   └── f* = (p×b - q) / b × 0.25

6. 部位建議
   └── position_size = min(confidence × 0.20, 0.20)
       × meta_label_adjust (if available)
       × regime_discount (bear = 0.5)
```

---

## 五、數據流向總結

```
外部數據源                    內部處理                     輸出
─────────                    ────────                     ────
TWSE T86 API ──────┐
StockFetcher ──────┤
FinMind 月營收 ────┤    ┌─────────────────┐
yfinance 全球 ─────┼───→│  20 因子計算     │
yfinance 總經 ─────┤    │  (純演算法)      │───→ total_score
DB 情緒資料 ───────┤    └────────┬────────┘        (0~1)
                   │             │                    │
                   │    ┌────────▼────────┐           │
Claude Haiku ──────┼───→│  HMM 體制偵測   │───→ regime ──→ 權重調整
                   │    └─────────────────┘           │
                   │                                  │
LSTM+XGBoost ──────┼──────────────────────────→ ml_ensemble 因子 (7%)
                   │                                  │
                   │    ┌─────────────────┐           │
                   └───→│  Phase 3 評分    │←──────────┘
                        │  Σ(factor×weight) │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
Claude Sonnet ─────────→│  Phase 4 敘事    │───→ 分析報告 (JSON)
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Phase 5 風控    │───→ action + position_size
                        │  (硬性規則)      │
                        └────────┬────────┘
                                 │
                        ┌────────▼────────┐
                        │  Phase 6 儲存    │───→ MarketScanResult
                        │  + Alert 偵測   │     PipelineResult
                        └─────────────────┘     FactorICRecord
```

---

## 六、入口點

| 入口 | 觸發方式 | 備註 |
|------|---------|------|
| `POST /api/market/analyze/{stock_id}` | 前端個股分析按鈕 | SSE 串流，主要入口 |
| `POST /api/market/scan` | 全市場掃描 | 批次 40 支，用相同 `score_stock()` |
| `run_daily_pipeline()` | 排程器 15:30 | 逐支呼叫統一管線 |
| `trigger_reanalysis()` | 高嚴重度警報自動觸發 | 重跑統一管線 |
| `AgentOrchestrator.run_analysis()` | 向後相容接口 | 委派給統一管線 |

---

## 七、關鍵檔案對照

| 檔案 | 行數 | 職責 |
|------|------|------|
| `api/services/stock_analysis_service.py` | ~500 | 統一 6 階段管線入口 |
| `api/services/market_service.py` | ~2300 | 20 因子計算 + `score_stock()` + 市場掃描 |
| `src/agents/narrative_agent.py` | ~230 | LLM 情緒萃取 + 敘事生成 (2 次呼叫) |
| `src/models/ensemble.py` | ~600 | HMM + EnsemblePredictor + StackingEnsemble |
| `src/models/trainer.py` | ~700 | LSTM/XGBoost 訓練 + Triple Barrier + CPCV |
| `src/risk/manager.py` | ~340 | ATR Trailing Stop + 迴路斷路器 + Kelly |
| `src/models/meta_label.py` | ~175 | Meta-labeling 倉位校準 |
| `src/agents/orchestrator.py` | ~300 | 向後相容入口 (委派給統一管線) |
| `api/services/auto_pipeline_service.py` | ~140 | 每日批次管線 (使用統一管線) |
| `api/routers/market.py` | ~150 | FastAPI 10 個端點 |

---

## 八、DB 模型

| 表名 | 用途 |
|------|------|
| `StockPrice` | 日 OHLCV + 法人買賣超 + 融資券 |
| `SentimentRecord` | 新聞/PTT/鉅亨/Yahoo 情緒 |
| `MarketScanResult` | 20 因子掃描結果 (含 `factor_details` JSON) |
| `PipelineResult` | 每日管線結果 (signal/confidence/reasoning) |
| `FactorICRecord` | 因子有效性追蹤 (Spearman IC) |
| `Alert` | 5 種警報 (signal_change/strong_signal/institutional_surge/sync_buy/score_jump) |
| `FeatureImportanceRecord` | ML 特徵重要性 (MI/SHAP) |
| `Prediction` | 模型預測歷史 |
| `BacktestResult` | 回測結果 |
| `TradeJournal` | 交易日誌 |
| `AgentMemory` | Agent 記憶系統 |
| `DelistedStock` | 下市股票 (存活者偏差校正) |
